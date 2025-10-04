# allele_ot_sklearn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import math
import numpy as np


# ----------------------- Minimal IUPAC -> 4-bit masks ---------------------- #
class _IUPAC:
    """Internal: fast IUPAC→bitmask encoder (A=1, C=2, G=4, T=8; N=15; -/.=0)."""
    LUT = np.zeros(256, dtype=np.uint8)
    for k, v in {
        "-": 0b0000, ".": 0b0000, "N": 0b1111,
        "A": 0b0001, "C": 0b0010, "G": 0b0100, "T": 0b1000,
        "R": 0b0101, "Y": 0b1010, "S": 0b0110, "W": 0b1001,
        "K": 0b1100, "M": 0b0011, "B": 0b1110, "D": 0b1101,
        "H": 0b1011, "V": 0b0111,
    }.items():
        LUT[ord(k)] = v
        LUT[ord(k.lower())] = v

    @staticmethod
    def encode_mask(x: np.ndarray) -> np.ndarray:
        """Encode IUPAC chars to uint8 bitmasks of length-4 alphabet."""
        a = np.asarray(x)
        u = np.char.upper(a.astype("U1"))
        b = np.char.encode(u, "ascii").view(np.uint8)
        return _IUPAC.LUT[b].reshape(a.shape)

    @staticmethod
    def mask_to_prob(mask_vec: np.ndarray) -> np.ndarray:
        """Convert 1D mask vector (L,) to per-locus allele probabilities (L,4).

        Homozygote → 1 on that base; heterozygote → 0.5+0.5; multi-allelic → uniform
        over present bits; missing ('-','.') → row of NaNs.

        Args:
            mask_vec: 1D uint8 bitmask per locus.

        Returns:
            (L, 4) float32 array with rows summing to 1 (or NaN row if missing).
        """
        m = mask_vec.astype(np.uint8, copy=False)
        L = m.size
        P = np.zeros((L, 4), dtype=np.float32)
        for j, bit in enumerate((1, 2, 4, 8)):
            P[:, j] = (m & bit) != 0
        denom = P.sum(axis=1, keepdims=True)
        miss = denom[:, 0] == 0
        P[~miss] /= denom[~miss]
        P[miss] = np.nan
        return P


# --------------------------- Sinkhorn OT helpers --------------------------- #
def _sinkhorn_emd(p: np.ndarray, q: np.ndarray, C: np.ndarray, eps: float,
                  n_iter: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized entropic OT (Sinkhorn) for many (4→4) problems.

    Args:
        p: (..., 4) source distributions (rows sum to 1).
        q: (..., 4) target distributions (rows sum to 1).
        C: (4, 4) ground cost matrix.
        eps: Entropic regularization (smaller → sharper).
        n_iter: Number of Sinkhorn iterations.

    Returns:
        (pi, u, v): transport plans (...,4,4) and dual scalings (...,4), (...,4).
    """
    K = np.exp(-C / max(eps, 1e-8)).astype(np.float64)  # (4,4)
    Kb = np.broadcast_to(K, p.shape[:-1] + (4, 4))
    u = np.ones_like(p, dtype=np.float64) / 4.0
    v = np.ones_like(q, dtype=np.float64) / 4.0
    for _ in range(n_iter):
        Ku = np.einsum("...ij,...j->...i", Kb, v)
        u = p / np.clip(Ku, 1e-12, None)
        Kv = np.einsum("...ij,...i->...j", Kb, u)
        v = q / np.clip(Kv, 1e-12, None)
    pi = u[..., :, None] * Kb * v[..., None, :]
    return pi, u, v

def _emd_cost(pi: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Compute <pi,C> over trailing axes."""
    return np.einsum("...ij,ij->...", pi, C, dtype=np.float64)


# ----------------------------- Main Estimator ----------------------------- #
@dataclass
class AlleleOTEstimator:
    """Scikit-learn–style Optimal-Transport estimator for unlinked SNPs.

    This estimator learns per-population per-locus allele-frequency profiles
    from IUPAC genotypes and exposes distances/assignments via a familiar API.

    Args:
        eps: Entropic regularization for Sinkhorn (typ. 0.03–0.2).
        n_iter: Sinkhorn iterations (30–100 is fine for 4x4).
        cost_mode: Ground cost between alleles:
            - "hamming": 0 on match, 1 otherwise.
            - "ts_tv": transitions cheaper (A<->G, C<->T = 0.6).
        min_locus_callrate: Minimum non-missing fraction required to keep a locus.
        laplace: Pseudocount per allele when estimating pop frequencies.
        allele_order: Order of the allele alphabet (default A,C,G,T).

    Attributes:
        classes_: Sorted unique population labels.
        C_: (4,4) ground cost matrix.
        loci_mask_: Boolean mask for loci retained after QC.
        pop_freqs_: Array of shape (n_classes, L_kept, 4) with per-locus freqs.

    Notes:
        - X is (n_samples, L) with single-char IUPAC codes.
        - Unlinked SNP assumption: locuswise independent 4×4 OT problems.
    """

    eps: float = 0.07
    n_iter: int = 60
    cost_mode: Literal["hamming", "ts_tv"] = "ts_tv"
    min_locus_callrate: float = 0.6
    laplace: float = 0.5
    allele_order: Tuple[str, str, str, str] = ("A", "C", "G", "T")

    # Fitted members
    classes_: Optional[np.ndarray] = None
    C_: Optional[np.ndarray] = None
    loci_mask_: Optional[np.ndarray] = None
    pop_freqs_: Optional[np.ndarray] = None  # (K, Lk, 4)

    # ------------------------------ Utilities ------------------------------ #
    def _build_cost(self) -> np.ndarray:
        C = np.ones((4, 4), dtype=np.float64)
        np.fill_diagonal(C, 0.0)
        if self.cost_mode == "hamming":
            return C
        # transitions discounted
        idx = {b: i for i, b in enumerate(self.allele_order)}
        for a, b in (("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")):
            C[idx[a], idx[b]] = 0.6
        return C

    # ------------------------------- Fitting ------------------------------- #
    def fit(self, X: np.ndarray, y: Iterable[str]) -> "AlleleOTEstimator":
        """Fit population allele-frequency profiles.

        Args:
            X: (n_samples, L) array of IUPAC genotypes.
            y: Iterable of length n_samples with population labels.

        Returns:
            self
        """
        X = np.asarray(X)
        n, L = X.shape
        y = np.asarray(list(y))
        if y.shape[0] != n:
            raise ValueError("y must match number of rows in X.")

        self.classes_ = np.unique(y)
        self.C_ = self._build_cost()

        # Encode and convert individuals to per-locus allele distributions.
        masks = _IUPAC.encode_mask(X)  # (n, L)
        indiv = np.empty((n, L, 4), dtype=np.float32)
        for i in range(n):
            indiv[i] = _IUPAC.mask_to_prob(masks[i])

        # Locus QC: keep loci with callrate >= threshold (any population)
        is_obs = ~np.isnan(indiv).any(axis=2)  # (n, L)
        callrate = is_obs.mean(axis=0)         # (L,)
        self.loci_mask_ = callrate >= float(self.min_locus_callrate)
        if not self.loci_mask_.any():
            raise ValueError("No loci pass call-rate filter.")

        # Estimate per-pop frequencies with Laplace smoothing.
        Lk = int(self.loci_mask_.sum())
        K = self.classes_.size
        pop_freqs = np.empty((K, Lk, 4), dtype=np.float64)

        for k, pop in enumerate(self.classes_):
            idx = np.where(y == pop)[0]
            P = indiv[idx][:, self.loci_mask_, :]             # (n_k, Lk, 4)
            valid = ~np.isnan(P).any(axis=2)                  # (n_k, Lk)
            counts = np.nansum(np.where(valid[..., None], P, 0.0), axis=0)  # (Lk,4)
            counts += self.laplace
            pop_freqs[k] = counts / counts.sum(axis=1, keepdims=True)

        self.pop_freqs_ = pop_freqs
        return self

    # --------------------------- Core distances ---------------------------- #
    def _satd_one(self, geno_row: np.ndarray, per_locus: bool = False):
        """Compute SATD distances from one individual to each population."""
        if self.pop_freqs_ is None or self.C_ is None or self.loci_mask_ is None:
            raise RuntimeError("Estimator not fitted.")
        mask = _IUPAC.encode_mask(np.asarray(geno_row))
        P = _IUPAC.mask_to_prob(mask)[self.loci_mask_]  # (Lk,4)
        valid = ~np.isnan(P).any(axis=1)
        P = P[valid]                                     # (Lk*,4)
        out = np.empty(self.classes_.size, dtype=np.float64)
        out_loci = None
        if per_locus:
            out_loci = np.full((self.classes_.size, self.loci_mask_.sum()), np.nan, dtype=np.float64)

        for k in range(self.classes_.size):
            Q = self.pop_freqs_[k][valid]               # (Lk*,4)
            pi, _, _ = _sinkhorn_emd(P, Q, self.C_, self.eps, self.n_iter)
            costs = _emd_cost(pi, self.C_)              # (Lk*,)
            out[k] = float(np.mean(costs))
            if per_locus:
                full = np.full(self.loci_mask_.sum(), np.nan, dtype=np.float64)
                full[valid] = costs
                out_loci[k] = full
        return out if not per_locus else (out, out_loci)

    # -------------------------- SK-style interface -------------------------- #
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute SATD distances to each population for a batch of individuals.

        Args:
            X: (n_samples, L) IUPAC array.

        Returns:
            (n_samples, n_classes) array of mean per-locus OT costs (lower is closer).
        """
        X = np.asarray(X)
        d = np.empty((X.shape[0], self.classes_.size), dtype=np.float64)
        for i in range(X.shape[0]):
            d[i] = self._satd_one(X[i])
        return d

    def predict_proba(self, X: np.ndarray, temperature: float = 0.3) -> np.ndarray:
        """Soft mixture weights over populations from SATD via softmax.

        Args:
            X: (n_samples, L) IUPAC array.
            temperature: Softmax temperature applied to negative distances.

        Returns:
            (n_samples, n_classes) array of probabilities per population.
        """
        D = self.transform(X)  # distances
        logits = -D / max(float(temperature), 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        W = np.exp(logits)
        W /= W.sum(axis=1, keepdims=True)
        return W

    def predict(self, X: np.ndarray, temperature: float = 0.3) -> np.ndarray:
        """Hard assignment to the most likely population by soft weights.

        Args:
            X: (n_samples, L) IUPAC array.
            temperature: Temperature used in `predict_proba`.

        Returns:
            (n_samples,) array of predicted class labels.
        """
        W = self.predict_proba(X, temperature=temperature)
        idx = np.argmax(W, axis=1)
        return self.classes_[idx]

    def kneighbors(self, X: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k closest populations per individual by SATD.

        Args:
            X: (n_samples, L) IUPAC array.
            k: Number of nearest populations to return.

        Returns:
            (indices, distances):
                indices → (n_samples, k) indices into `classes_`.
                distances → (n_samples, k) corresponding SATD values.
        """
        D = self.transform(X)
        k = min(k, self.classes_.size)
        idx = np.argpartition(D, kth=k-1, axis=1)[:, :k]
        # sort the k neighbors per row
        row = np.arange(D.shape[0])[:, None]
        order = np.argsort(D[row, idx], axis=1)
        idx_sorted = idx[row, order]
        dist_sorted = D[row, idx_sorted]
        return idx_sorted, dist_sorted

    # ------------------- Population geometry & barycenters ------------------ #
    def pairwise_population_distances(self) -> np.ndarray:
        """Pairwise SATD among fitted populations.

        Returns:
            (K, K) symmetric matrix of mean per-locus SATD.
        """
        if self.pop_freqs_ is None or self.C_ is None:
            raise RuntimeError("Estimator not fitted.")
        K = self.classes_.size
        D = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            for j in range(i, K):
                P, Q = self.pop_freqs_[i], self.pop_freqs_[j]
                pi, _, _ = _sinkhorn_emd(P, Q, self.C_, self.eps, self.n_iter)
                c = _emd_cost(pi, self.C_)
                D[i, j] = D[j, i] = float(np.mean(c))
        return D

    def barycenter(self, weights: Dict[str, float], n_iter: int = 30) -> np.ndarray:
        """OT barycenter (per-locus allele-frequency profile) across populations.

        Args:
            weights: Mapping {class_label: weight}. Auto-normalized.
            n_iter: Number of fixed-point iterations.

        Returns:
            (L_kept, 4) allele-frequency barycenter.
        """
        if self.pop_freqs_ is None or self.C_ is None or self.classes_ is None:
            raise RuntimeError("Estimator not fitted.")
        w = np.array([weights.get(c, 0.0) for c in self.classes_], dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("At least one positive weight required.")
        w /= w.sum()

        Lk = self.pop_freqs_.shape[1]
        B = self.pop_freqs_.mean(axis=0)  # (Lk,4)
        Kmat = np.exp(-self.C_ / max(self.eps, 1e-8))

        for _ in range(n_iter):
            logs = np.zeros_like(B)
            for alpha, k in zip(w, range(self.classes_.size)):
                Q = self.pop_freqs_[k]
                u = np.ones_like(B) / 4.0
                for _ in range(10):
                    Ku = (Kmat[None, :, :] * u[..., None, :]).sum(axis=1)
                    v = Q / np.clip(Ku, 1e-12, None)
                    Kv = (Kmat[None, :, :] * v[..., None, :]).sum(axis=2)
                    u = B / np.clip(Kv, 1e-12, None)
                T = u[..., :, None] * Kmat[None, :, :] * v[..., None, :]
                logs += alpha * np.log(np.clip(T.sum(axis=2), 1e-12, None))
            B = np.exp(logs)
            B /= B.sum(axis=1, keepdims=True)
        return B

    # ------------------------------ Plotting -------------------------------- #
    def plot_population_mds(self, ax=None):
        """Plot classical MDS (Torgerson) of the population SATD matrix.

        Args:
            ax: Optional matplotlib Axes. If None, creates a new figure.

        Returns:
            (fig, ax, emb): Figure, Axes, and (K,2) embedding array.
        """
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise ImportError("matplotlib is required for plotting.") from e

        D = self.pairwise_population_distances()
        # Double-centering for classical MDS
        K = D.shape[0]
        J = np.eye(K) - np.ones((K, K))/K
        B = -0.5 * J @ (D**2) @ J
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        w = np.clip(eigvals[idx][:2], 0, None)
        V = eigvecs[:, idx][:, :2]
        emb = V * np.sqrt(w + 1e-12)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.scatter(emb[:, 0], emb[:, 1])
        for i, label in enumerate(self.classes_):
            ax.text(emb[i, 0], emb[i, 1], str(label))
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
        ax.set_title("Population geometry (SATD MDS)")
        return fig, ax, emb

    def plot_barycenter_heatmap(self, B: np.ndarray, max_loci: int = 50, ax=None):
        """Plot a small heatmap of a barycenter’s allele freqs (subset of loci).

        Args:
            B: (L_kept, 4) barycenter allele-frequency matrix.
            max_loci: Number of loci to display (top rows).
            ax: Optional matplotlib Axes.

        Returns:
            (fig, ax)
        """
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise ImportError("matplotlib is required for plotting.") from e

        L = min(max_loci, B.shape[0])
        M = B[:L]  # (L,4)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        im = ax.imshow(M, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(4))
        ax.set_xticklabels(list(self.allele_order))
        ax.set_ylabel("Locus (subset)")
        ax.set_title("Barycenter allele frequencies (subset)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig, ax
