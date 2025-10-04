from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, "np.typing.NDArray[np.float64]"]
AxisSel = Union[int, Tuple[int, int]]
CIKind = Literal["percentile", "bca"]
NanPolicy = Literal["propagate", "omit"]


@dataclass
class Bootstrap2D:
    """General-purpose 2D bootstrap with stratification, blocks, and BCa CIs.

    This class resamples a 2D array along rows and/or columns and evaluates
    arbitrary statistics. It includes efficient vectorized index generation,
    optional stratification, weighted sampling, (circular) block bootstrap
    for correlated data, and confidence intervals (percentile, BCa).

    Args:
        axis: Which axes to resample. 0 = rows, 1 = cols, (0,1) = both independently.
        replace: Sample with replacement (True) or without (False).
        random_state: Seed or Generator for reproducibility.
        nan_policy: How to handle NaNs produced by `stat_fn`. If "omit", NaNs are
            removed before computing CIs; "propagate" keeps them in the bootstrap
            distribution.

    Notes:
        - Stratification and weights are specified per-axis in the sampling calls.
        - For BCa intervals, jackknife is performed along each resampled axis.
        - When `axis=(0,1)`, row and column resampling are independent per replicate.
        - For large n_boot, prefer `apply()` which streams resamples to save memory.

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> X = rng.normal(size=(100, 20))
        >>> bs = Bootstrap2D(axis=0, random_state=42)
        >>> def stat_mean(x: np.ndarray) -> float:
        ...     return float(np.nanmean(x))
        >>> est, (lo, hi), dist = bs.bootstrap(X, stat_mean, n_boot=2000, ci=("bca", 95))
        >>> round(est, 3), round(lo, 3), round(hi, 3)
        (0.012, -0.121, 0.142)
    """

    axis: AxisSel = 0
    replace: bool = True
    random_state: Optional[Union[int, np.random.Generator]] = None
    nan_policy: NanPolicy = "propagate"

    # ---------------------------- RNG utilities ---------------------------- #
    def _rng(self) -> np.random.Generator:
        if isinstance(self.random_state, np.random.Generator):
            return self.random_state
        return np.random.default_rng(self.random_state)

    # ------------------------- Public sampling API ------------------------- #
    def sample_indices(
        self,
        X: np.ndarray,
        n_boot: int,
        *,
        row_weights: Optional[np.ndarray] = None,
        col_weights: Optional[np.ndarray] = None,
        row_strata: Optional[np.ndarray] = None,
        col_strata: Optional[np.ndarray] = None,
        row_block: Optional[int] = None,
        col_block: Optional[int] = None,
        circular: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate bootstrap index matrices for rows/cols.

        This is vectorized and returns integer arrays of shape:
            rows_idx: (n_boot, n_rows_out) or None
            cols_idx: (n_boot, n_cols_out) or None

        For simple bootstrap, n_rows_out == n_rows and n_cols_out == n_cols.
        For block bootstrap, the output length is ceil(n / block_size) * block_size
        (then truncated back to n to keep shape consistent).

        Args:
            X: Input array, shape (n_rows, n_cols).
            n_boot: Number of bootstrap replicates.
            row_weights: Probabilities for row sampling (sum to 1).
            col_weights: Probabilities for column sampling (sum to 1).
            row_strata: Integer stratum labels per row; sampling preserves counts.
            col_strata: Integer stratum labels per column.
            row_block: Block size for row-wise block bootstrap (>=1).
            col_block: Block size for column-wise block bootstrap (>=1).
            circular: Whether to wrap blocks circularly.

        Returns:
            (rows_idx, cols_idx): Each is an int array or None if that axis is not sampled.
        """
        rng = self._rng()
        n_rows, n_cols = X.shape

        rows_idx = None
        cols_idx = None

        def _simple_indices(n_items: int, weights: Optional[np.ndarray]) -> np.ndarray:
            if self.replace:
                # Vectorized multinomial-style draws via choice
                if weights is not None:
                    return rng.choice(n_items, size=(n_boot, n_items), replace=True, p=weights)
                return rng.integers(0, n_items, size=(n_boot, n_items))
            else:
                # Without replacement: sample permutations per replicate
                # Efficiently: generate (n_boot, n_items) random keys and argsort
                keys = rng.random(size=(n_boot, n_items))
                idx = np.argsort(keys, axis=1)
                # Take the first n_items (all) — for symmetry with with-replacement
                return idx

        def _stratified_indices(labels: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
            # Preserve per-stratum counts exactly; allow optional weights within stratum
            labels = np.asarray(labels)
            uniques, counts = np.unique(labels, return_counts=True)
            # Preallocate output
            out = np.empty((n_boot, labels.size), dtype=np.int64)
            start = 0
            for s, c in zip(uniques, counts):
                stratum_idx = np.flatnonzero(labels == s)
                if weights is not None:
                    p = weights[stratum_idx]
                    p = p / p.sum()
                else:
                    p = None
                if self.replace:
                    draw = rng.choice(stratum_idx, size=(n_boot, c), replace=True, p=p)
                else:
                    # Without replacement inside stratum
                    if c > stratum_idx.size:
                        raise ValueError("Stratum size smaller than requested sample without replacement.")
                    # Permute each replicate, then take first c
                    keys = rng.random(size=(n_boot, stratum_idx.size))
                    perm = stratum_idx[np.argsort(keys, axis=1)[:, :c]]
                    draw = perm
                out[:, start:start + c] = draw
                start += c
            # Shuffle columns so strata blocks are not grouped
            perm_keys = rng.random(size=(n_boot, labels.size))
            out = out[np.arange(n_boot)[:, None], np.argsort(perm_keys, axis=1)]
            return out

        def _block_indices(n_items: int, block: int) -> np.ndarray:
            # Circular (moving) block bootstrap: pick starting positions, then
            # take contiguous blocks; concatenate until length >= n_items, then trim.
            if block <= 0:
                raise ValueError("block must be >= 1")
            n_blocks = int(np.ceil(n_items / block))
            # Start positions for each block in each replicate
            starts = rng.integers(0, n_items if circular else (n_items - block + 1),
                                  size=(n_boot, n_blocks))
            # Build indices
            idx = np.empty((n_boot, n_blocks * block), dtype=np.int64)
            base = np.arange(block, dtype=np.int64)[None, None, :]
            # idx[r, b, k] = (starts[r, b] + k) % n_items (if circular)
            stepped = (starts[:, :, None] + base)
            if circular:
                stepped %= n_items
            else:
                # clamp to max start if not circular (ensure valid range)
                stepped = np.minimum(stepped, n_items - 1)
            idx = stepped.reshape(n_boot, -1)[:, :n_items]
            return idx

        # Decide per-axis plans
        sample_rows = (self.axis == 0) or (isinstance(self.axis, tuple) and 0 in self.axis)
        sample_cols = (self.axis == 1) or (isinstance(self.axis, tuple) and 1 in self.axis)

        if sample_rows:
            if row_strata is not None:
                rows_idx = _stratified_indices(np.asarray(row_strata), row_weights)
            elif row_block is not None:
                rows_idx = _block_indices(n_rows, int(row_block))
            else:
                rows_idx = _simple_indices(n_rows, row_weights)

        if sample_cols:
            if col_strata is not None:
                cols_idx = _stratified_indices(np.asarray(col_strata), col_weights)
            elif col_block is not None:
                cols_idx = _block_indices(n_cols, int(col_block))
            else:
                cols_idx = _simple_indices(n_cols, col_weights)

        return rows_idx, cols_idx

    def iter_resamples(
        self,
        X: np.ndarray,
        n_boot: int,
        **sample_kwargs,
    ) -> Iterable[np.ndarray]:
        """Yield bootstrap resamples one-by-one (memory efficient).

        Args:
            X: Input array (n_rows, n_cols).
            n_boot: Number of bootstrap replicates.
            **sample_kwargs: Passed to `sample_indices`.

        Yields:
            Resampled arrays, shape (n_rows, n_cols).
        """
        rows_idx, cols_idx = self.sample_indices(X, n_boot, **sample_kwargs)
        for b in range(n_boot):
            xr = X
            if rows_idx is not None:
                xr = xr[rows_idx[b], :]
            if cols_idx is not None:
                xr = xr[:, cols_idx[b]]
            yield xr

    # ---------------------------- CI helpers -------------------------------- #
    @staticmethod
    def _percentile_ci(dist: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray]:
        q_lo = (100.0 - level) / 2.0
        q_hi = 100.0 - q_lo
        lo = np.nanpercentile(dist, q_lo, axis=0)
        hi = np.nanpercentile(dist, q_hi, axis=0)
        return lo, hi

    @staticmethod
    def _jackknife_stats(
        X: np.ndarray,
        stat_fn: Callable[[np.ndarray], np.ndarray],
        axis: AxisSel,
    ) -> np.ndarray:
        """Leave-one-out along the specified axis/axes (independently)."""
        if isinstance(axis, tuple):
            # Jackknife rows, then columns independently; concatenate results
            jk_rows = Bootstrap2D._jackknife_stats(X, stat_fn, 0) if 0 in axis else None
            jk_cols = Bootstrap2D._jackknife_stats(X, stat_fn, 1) if 1 in axis else None
            if jk_rows is not None and jk_cols is not None:
                return np.concatenate([jk_rows, jk_cols], axis=0)
            return jk_rows if jk_rows is not None else jk_cols

        n = X.shape[axis]
        out = []
        for i in range(n):
            slc = [slice(None), slice(None)]
            slc[axis] = np.r_[0:i, i + 1:n]  # drop i
            Xi = X[tuple(slc)]
            out.append(np.asarray(stat_fn(Xi)))
        return np.stack(out, axis=0)  # (n, k)

    @staticmethod
    def _bca_ci(
        theta_hat: np.ndarray,
        boot: np.ndarray,
        jack: np.ndarray,
        level: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """BCa interval from bootstrap and jackknife stats.

        Implements:
            z0 = Phi^{-1}( (#(boot < theta_hat))/B )
            a  = skewness from jackknife influence values
            Adjusted percentiles: Phi(z0 + (z0 + z_{alpha})/(1 - a*(z0 + z_{alpha})))

        Handles vector-valued statistics; broadcasting across components.
        """
        from math import sqrt
        # Convert to arrays with shape (B, k)
        boot = np.asarray(boot)
        theta_hat = np.asarray(theta_hat)
        B = boot.shape[0]

        # z0 bias-correction
        less = np.sum(boot < theta_hat, axis=0)
        frac = np.clip(less / B, 1e-10, 1 - 1e-10)
        z0 = _norm_ppf(frac)

        # Acceleration from jackknife
        jack = np.asarray(jack)  # (n, k)
        jack_mean = np.nanmean(jack, axis=0)
        num = np.nansum((jack_mean - jack) ** 3, axis=0)
        den = 6.0 * (np.nansum((jack_mean - jack) ** 2, axis=0) ** 1.5 + 1e-15)
        a = num / (den + 1e-20)

        # Target quantiles
        alpha = (100.0 - level) / 100.0 / 2.0
        zalpha_lo = _norm_ppf(alpha)
        zalpha_hi = _norm_ppf(1 - alpha)

        def _adj(zalpha):
            num = z0 + zalpha
            den = 1.0 - a * num
            return _norm_cdf(z0 + num / (den + 1e-20))

        q_lo = _adj(zalpha_lo)
        q_hi = _adj(zalpha_hi)

        lo = np.nanquantile(boot, q_lo, axis=0, method="linear")
        hi = np.nanquantile(boot, q_hi, axis=0, method="linear")
        return lo, hi

    # ----------------------------- Main API --------------------------------- #
    def apply(
        self,
        X: np.ndarray,
        stat_fn: Callable[[np.ndarray], ArrayLike],
        n_boot: int,
        *,
        ci: Tuple[CIKind, float] | None = None,
        sample_kwargs: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Apply `stat_fn` over bootstrap resamples (streaming).

        Args:
            X: Input array (n_rows, n_cols).
            stat_fn: Callable mapping (array) -> scalar or 1D array (k,).
            n_boot: Number of bootstrap replicates.
            ci: Optional tuple (kind, level). kind in {"percentile","bca"}.
            sample_kwargs: Extra args for `sample_indices` (weights/strata/blocks).

        Returns:
            (theta_hat, (ci_lo, ci_hi) or None, boot_dist)
                theta_hat: statistic on the original X, shape (k,)
                ci_lo/ci_hi: CI bounds if requested
                boot_dist: bootstrap statistics, shape (n_boot, k)
        """
        sample_kwargs = sample_kwargs or {}
        theta_hat = np.atleast_1d(np.asarray(stat_fn(X)))
        k = theta_hat.shape[0] if theta_hat.ndim > 0 else 1

        boot_stats = np.empty((n_boot, k), dtype=float)
        for b, xb in enumerate(self.iter_resamples(X, n_boot, **sample_kwargs)):
            s = np.asarray(stat_fn(xb))
            boot_stats[b] = np.atleast_1d(s)

        if self.nan_policy == "omit":
            boot_stats = boot_stats[~np.any(np.isnan(boot_stats), axis=1)]

        ci_bounds = None
        if ci is not None:
            kind, level = ci
            if kind == "percentile":
                lo, hi = self._percentile_ci(boot_stats, float(level))
                ci_bounds = (lo, hi)
            elif kind == "bca":
                # Jackknife along resampled axes
                jk = self._jackknife_stats(X, stat_fn, self.axis)
                ci_bounds = self._bca_ci(theta_hat, boot_stats, jk, float(level))
            else:
                raise ValueError("ci kind must be 'percentile' or 'bca'")

        return theta_hat, ci_bounds, boot_stats

    def bootstrap(
        self,
        X: np.ndarray,
        stat_fn: Callable[[np.ndarray], ArrayLike],
        n_boot: int,
        *,
        ci: Tuple[CIKind, float] = ("percentile", 95.0),
        **sample_kwargs,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Convenience wrapper around `apply()` with direct kwargs.

        Args:
            X: Input array (n_rows, n_cols).
            stat_fn: Callable mapping array -> scalar or (k,) vector.
            n_boot: Number of bootstrap replicates.
            ci: Confidence interval spec (kind, level).
            **sample_kwargs: Arguments forwarded to `sample_indices`.

        Returns:
            (theta_hat, (ci_lo, ci_hi), boot_dist)
        """
        return self.apply(X, stat_fn, n_boot, ci=ci, sample_kwargs=sample_kwargs)


# --------------------------- Small math helpers --------------------------- #
# Fast, dependency-free normal CDF/PPF using erfc/erfcinv.
# (Good enough for CI calculations; avoids SciPy dependency.)
def _norm_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))

def _norm_ppf(p: np.ndarray) -> np.ndarray:
    # Acklam approximation adapted for vectorization and numerical stability
    p = np.asarray(p, dtype=float)
    # Clip to (0,1) to avoid infs
    p = np.clip(p, 1e-12, 1 - 1e-12)

    # Coefficients
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]

    pl = p < 0.02425
    pu = p > 1 - 0.02425
    pm = ~(pl | pu)

    x = np.empty_like(p)

    # Lower region
    if np.any(pl):
        q = np.sqrt(-2 * np.log(p[pl]))
        x[pl] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    # Central region
    if np.any(pm):
        q = p[pm] - 0.5
        r = q*q
        x[pm] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                 (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

    # Upper region
    if np.any(pu):
        q = np.sqrt(-2 * np.log(1 - p[pu]))
        x[pu] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                  ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    return x
