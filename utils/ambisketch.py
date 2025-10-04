from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple

import numpy as np


# -------------------------- Minimal IUPAC encoder -------------------------- #
class _IUPAC:
    """Tiny 4-bit encoder used internally: A=1, C=2, G=4, T=8; N=15; '-'/'.'->0."""
    LUT = np.array([0] * 256, dtype=np.uint8)
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
    def encode(x: np.ndarray | Iterable[str]) -> np.ndarray:
        a = np.asarray(x)
        u = np.char.upper(a.astype("U1"))
        b = np.char.encode(u, "ascii").view(np.uint8)
        return _IUPAC.LUT[b].reshape(a.shape)


# ------------------------- Ambiguity-aware sketcher ------------------------ #
SketchMode = Literal["simhash", "minhash"]
AxisSel = Literal["rows", "cols"]


@dataclass
class AmbiSketch:
    """Ambiguity-aware LSH sketches for 2D IUPAC genotype matrices.

    Two complementary sketch families:
      • "simhash"  — Signed Random Projections (SRP) over allele features ≈ cosine
      • "minhash"  — One-permutation MinHash over allele features ≈ Jaccard

    Features are the set of (position, base) pairs present in a row/column,
    where the base is one of {A,C,G,T} and "present" means the IUPAC mask
    includes that allele. Heterozygotes contribute multiple features; homozygotes
    can optionally be dose-weighted (2×) for SRP.

    Args:
        n_hashes: Number of hash bits (SimHash) or permutations (MinHash).
        mode: "simhash" or "minhash".
        axis: Build sketches for "rows" (samples) or "cols" (loci).
        random_state: Seed or Generator for reproducibility.
        weight_homozygote: If True and mode="simhash", AA/CC/GG/TT contribute 2×.

    Methods:
        fit(X): Build sketches for X (array of IUPAC chars).
        similarity(i, j): Approximate similarity between two items by index.
        nearest_neighbors(q, k): Find top-k approximate neighbors of item q.
        pairwise_sim(): Dense matrix of approximate similarities (sketch space).

    Notes:
        • Input X must be shape (n_rows, n_cols) with dtype str/'U1'/'S1'.
        • Works streaming-friendly; does NOT expand to 4× one-hot in memory.
        • For truly massive data, compute sketches by chunks of rows/cols.

    """

    n_hashes: int = 128
    mode: SketchMode = "simhash"
    axis: AxisSel = "rows"
    random_state: Optional[int | np.random.Generator] = None
    weight_homozygote: bool = True

    # Learned/created state after fit()
    _sketch: Optional[np.ndarray] = None  # (n_items, n_hashes) int/uint
    _n_items: int = 0
    _shape: Tuple[int, int] = (0, 0)
    _rng: Optional[np.random.Generator] = None
    # Hash seeds for SRP/MinHash
    _h_seeds: Optional[np.ndarray] = None  # (n_hashes,) uint64
    _perm_a: Optional[np.ndarray] = None   # (n_hashes,) uint64  (minhash)
    _perm_b: Optional[np.ndarray] = None   # (n_hashes,) uint64  (minhash)

    # ------------------------------ Utilities ------------------------------ #
    def _get_rng(self) -> np.random.Generator:
        if isinstance(self.random_state, np.random.Generator):
            return self.random_state
        return np.random.default_rng(self.random_state)

    @staticmethod
    def _hash64(x: np.ndarray, seed: int) -> np.ndarray:
        """Fast 64-bit mix (splitmix-style) for deterministic pseudorandomness."""
        z = (x + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(0xFFFFFFFFFFFFFFFF)
        return z ^ (z >> np.uint64(31))

    # Represent feature key as uint64: (position << 2) | base_id among {A=0,C=1,G=2,T=3}
    @staticmethod
    def _feature_keys(mask_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (keys, weights) for one row/col mask vector.

        We generate one feature per present allele at each position.
        """
        m = mask_row.astype(np.uint8, copy=False)
        if m.ndim != 1:
            m = m.ravel()

        # Positions with each base bit present
        pos = np.arange(m.size, dtype=np.uint64)
        keys_list = []
        w_list = []

        # For each base, collect positions where bit is set
        for base_id, bit, w2 in ((0, 1, "A"), (1, 2, "C"), (2, 4, "G"), (3, 8, "T")):
            idx = np.nonzero((m & bit) != 0)[0]
            if idx.size == 0:
                continue
            # Pack (pos, base) into 64-bit key
            k = (pos[idx] << np.uint64(2)) | np.uint64(base_id)
            keys_list.append(k)
            w_list.append(idx)  # store indices for weighting later

        if not keys_list:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.float32)

        keys = np.concatenate(keys_list)
        # Optional dosage weight: +1 extra if mask is homozygote for that base
        w = np.ones(keys.size, dtype=np.float32)
        # homozygote: exactly one bit set (popcount==1)
        if True:
            # vectorized popcount on nibble via table
            POP = np.array([0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4], dtype=np.uint8)
            pc = POP[m]
            # map back per-feature; a feature belongs to index idx in each base list
            cursor = 0
            for lst_idx in w_list:
                # homozygous at those positions?
                homo = (pc[lst_idx] == 1)
                if homo.any():
                    # add +1 if homozygote and user requested dose-weighting
                    if self.weight_homozygote:
                        w[cursor:cursor + lst_idx.size][homo] += 1.0
                cursor += lst_idx.size

        return keys, w

    # ------------------------------ Fit ------------------------------------ #
    def fit(self, X: np.ndarray) -> "AmbiSketch":
        """Build sketches for a 2D array of IUPAC nucleotide codes.

        Args:
            X: Array of shape (n_rows, n_cols), dtype str/'U1'/'S1'.

        Returns:
            self

        Raises:
            ValueError: if mode is invalid or X is not 2D.
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_rows, n_cols).")

        self._rng = self._get_rng()
        self._shape = X.shape
        n_rows, n_cols = X.shape

        # Encode to 4-bit masks once
        M = _IUPAC.encode(X)

        # We sketch either rows or columns; unify by iterating 1D mask vectors
        if self.axis == "rows":
            n_items = n_rows
            iterator = (M[i, :] for i in range(n_rows))
        elif self.axis == "cols":
            n_items = n_cols
            iterator = (M[:, j] for j in range(n_cols))
        else:
            raise ValueError("axis must be 'rows' or 'cols'.")

        self._n_items = n_items

        if self.mode == "simhash":
            self._build_simhash(iterator, n_items)
        elif self.mode == "minhash":
            self._build_minhash(iterator, n_items)
        else:
            raise ValueError("mode must be 'simhash' or 'minhash'.")

        return self

    # ------------------------------ SimHash --------------------------------- #
    def _build_simhash(self, iterator, n_items: int) -> None:
        """Signed random projections over feature universe—bit signature per item."""
        # Seeds per hash bit
        self._h_seeds = self._rng.integers(0, np.iinfo(np.uint64).max, size=self.n_hashes, dtype=np.uint64)
        # Accumulator for signs > 0
        bits = np.zeros((n_items, self.n_hashes), dtype=np.uint8)

        for i, vec in enumerate(iterator):
            # For each item, compute signed projection per hash over present features
            k, w = self._feature_keys(vec)
            if k.size == 0:
                # all zeros
                bits[i, :] = 0
                continue
            # For each hash, derive ±1 sign per feature via 64-bit hashing
            # sign = +1 if lowest bit of hash is 1, else -1; accumulate with weights
            # Vectorized: hash all features for each seed by broadcasting
            # (n_hashes, n_features)
            hs = self._hash64(k[None, :], self._h_seeds[:, None])
            signs = ((hs & 1) * 2 - 1).astype(np.int8)  # {0,1} -> {-1,+1}
            proj = (signs * w[None, :]).sum(axis=1)     # (n_hashes,)
            bits[i, :] = (proj > 0).astype(np.uint8)

        self._sketch = bits  # (n_items, n_hashes)

    # ------------------------------ MinHash --------------------------------- #
    def _build_minhash(self, iterator, n_items: int) -> None:
        """One-permutation MinHash (with densification) over feature sets."""
        # Universal hashing parameters per permutation
        self._perm_a = self._rng.integers(1, np.iinfo(np.uint64).max, size=self.n_hashes, dtype=np.uint64)
        self._perm_b = self._rng.integers(0, np.iinfo(np.uint64).max, size=self.n_hashes, dtype=np.uint64)
        P = np.uint64(0xFFFFFFFFFFFFFFFF)  # operate mod 2^64 wrap

        mins = np.full((n_items, self.n_hashes), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)

        for i, vec in enumerate(iterator):
            k, _ = self._feature_keys(vec)
            if k.size == 0:
                continue
            # hash: (a * k + b) mod 2^64, then take min per hash
            # broadcast: (n_hashes, n_features)
            h = (self._perm_a[:, None] * k[None, :] + self._perm_b[:, None]) & P
            mins[i, :] = h.min(axis=1)

        self._sketch = mins  # (n_items, n_hashes) uint64

    # --------------------------- Similarities / KNN ------------------------- #
    def similarity(self, i: int, j: int) -> float:
        """Approximate similarity between two sketched items.

        Returns:
            float in [0,1] for both modes:
                - simhash: fraction of equal bits ≈ cosine similarity
                - minhash: fraction of equal minima ≈ Jaccard similarity
        """
        S = self._sketch
        if S is None:
            raise RuntimeError("Call fit(X) before similarity().")
        a, b = S[i], S[j]
        if self.mode == "simhash":
            # Hamming agreement over n_hashes
            eq = (a == b).sum()
            return float(eq) / float(self.n_hashes)
        else:
            return float((a == b).sum()) / float(self.n_hashes)

    def pairwise_sim(self) -> np.ndarray:
        """Dense pairwise similarity among all items, using the sketches."""
        if self._sketch is None:
            raise RuntimeError("Call fit(X) first.")
        S = self._sketch
        n = S.shape[0]
        out = np.ones((n, n), dtype=np.float32)
        # Vectorized Hamming agreement using bit tricks
        if self.mode == "simhash":
            # Use uint8; equality -> sum across columns
            for i in range(n):
                eq = (S[i] == S).sum(axis=1)
                out[i] = eq / self.n_hashes
        else:
            for i in range(n):
                eq = (S[i] == S).sum(axis=1)
                out[i] = eq / self.n_hashes
        return out

    def nearest_neighbors(self, q: int, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k neighbors by sketch similarity.

        Args:
            q: Query item index.
            k: Number of neighbors (excluding q).

        Returns:
            (idx, sim): neighbor indices and their approximate similarities.
        """
        if self._sketch is None:
            raise RuntimeError("Call fit(X) first.")
        sims = self.pairwise_sim()[q]
        sims[q] = -1.0
        idx = np.argpartition(-sims, k)[:k]
        return idx[np.argsort(-sims[idx])], sims[idx][np.argsort(-sims[idx])]
