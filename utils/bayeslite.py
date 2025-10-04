# bayeslite.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import math
import numpy as np

try:
    import pandas as pd  # optional for pretty summaries
except Exception:  # pragma: no cover
    pd = None


Array = np.ndarray
LogProbFn = Callable[[Array], float]
LogLikFn = Callable[[Array, object], float]
PPFn = Callable[[Array, int, object], Array]


# ------------------------------ Utilities --------------------------------- #
def _as_1d(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    return x.ravel()

def _safe_logsumexp(a: Array) -> float:
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))

def _hdi(samples: Array, cred: float = 0.95) -> Tuple[float, float]:
    """Compute the highest-density interval (HDI) for 1D samples."""
    s = np.sort(_as_1d(samples))
    n = s.size
    if n == 0:
        return (np.nan, np.nan)
    m = max(1, int(np.floor(cred * n)))
    widths = s[m:] - s[: n - m]
    j = np.argmin(widths)
    return float(s[j]), float(s[j + m])

def _autocov(x: Array, lag: int) -> float:
    x = _as_1d(x)
    n = x.size
    mu = np.mean(x)
    return float(np.dot(x[: n - lag] - mu, x[lag:] - mu) / (n - lag))

def _ess(x: Array) -> float:
    """Geyer’s initial positive sequence estimator for effective sample size."""
    x = _as_1d(x)
    n = x.size
    if n < 3:
        return float(n)
    gamma0 = _autocov(x, 0)
    s = 0.0
    k = 1
    while True:
        gamma = _autocov(x, 2 * k - 1) + _autocov(x, 2 * k)
        if gamma <= 0:
            break
        s += 2.0 * gamma
        k += 1
        if 2 * k >= n:
            break
    var = gamma0 + s
    return float(n * gamma0 / var) if var > 0 else float(n)

def _rhat(chains: Array) -> float:
    """Split-chain R-hat for shape (n_chains, n_draws)."""
    c = np.asarray(chains, dtype=float)
    if c.ndim != 2 or c.shape[1] < 4:
        return np.nan
    m, n = c.shape
    chain_means = c.mean(axis=1)
    b = n * chain_means.var(ddof=1)
    w = c.var(axis=1, ddof=1).mean()
    var_hat = (n - 1) / n * w + b / n
    return float(np.sqrt(var_hat / w))

def _mcse(x: Array) -> float:
    """Monte Carlo standard error via batch means."""
    z = _as_1d(x)
    n = z.size
    b = max(5, int(np.sqrt(n)))
    nb = n // b
    if nb < 2:
        return float(np.std(z, ddof=1) / math.sqrt(n))
    means = z[: nb * b].reshape(nb, b).mean(axis=1)
    return float(np.std(means, ddof=1) / math.sqrt(nb))


# ------------------------------ Core Model -------------------------------- #
@dataclass
class BayesModel:
    """Minimal Bayesian model interface.

    Provide log_prior and log_likelihood; posterior predictive is optional.

    Attributes:
        log_prior: Callable mapping θ -> log p(θ).
        log_likelihood: Callable mapping (θ, data) -> log p(data | θ).
        log_posterior: Convenience wrapper summing prior and likelihood.
        posterior_predictive: Optional mapping (θ, n, data) -> draws (n, ...).

    Notes:
        - θ is a 1D array (R^d). You choose the parameterization/dimension.
        - For discrete parameters, work on an unconstrained real transform.
        - Your `log_likelihood` can cache sufficient stats if helpful.
    """

    log_prior: LogProbFn
    log_likelihood: LogLikFn
    posterior_predictive: Optional[PPFn] = None

    def log_posterior(self, theta: Array, data: object) -> float:
        """Compute log p(θ | data) up to an additive constant."""
        return float(self.log_prior(theta) + self.log_likelihood(theta, data))


# ------------------------------ Samplers ---------------------------------- #
@dataclass
class Sampler:
    """General-purpose MCMC samplers: Metropolis–Hastings and Slice.

    Methods:
        mh: Random-walk Gaussian MH with optional adaptive scale.
        slice_univariate: Univariate stepping-out slice sampler (axis-wise).

    All samplers return draws and acceptance/diagnostic info.
    """

    rng: np.random.Generator = np.random.default_rng()

    def mh(
        self,
        model: BayesModel,
        data: object,
        theta0: Array,
        n_draws: int = 4000,
        burn: int = 1000,
        step: float | Array = 0.1,
        adapt: bool = True,
        target_accept: float = 0.25,
        thin: int = 1,
    ) -> Dict[str, object]:
        """Random-walk Gaussian Metropolis–Hastings.

        Args:
            model: Bayesian model.
            data: Data payload passed to the model.
            theta0: Initial parameter vector (d,).
            n_draws: Total draws (including burn-in) per chain.
            burn: Number of initial draws to discard.
            step: Scalar or vector step scale for proposal covariance.
            adapt: If True, adapt step during burn-in toward target_accept.
            target_accept: Target acceptance rate for adaptation (~0.2–0.4).
            thin: Keep one every `thin` draws after burn-in.

        Returns:
            dict with keys:
                'draws' -> (n_kept, d) ndarray
                'accept_rate' -> float
                'logp' -> (n_kept,) ndarray of log posteriors
        """
        theta = _as_1d(theta0)
        d = theta.size
        step_vec = np.full(d, float(step)) if np.isscalar(step) else _as_1d(step)
        cov = np.diag(step_vec ** 2)
        cur_logp = model.log_posterior(theta, data)

        kept = []
        kept_lp = []
        n_accept = 0
        gamma = 0.05  # adaptation learning rate

        for t in range(n_draws):
            prop = self.rng.multivariate_normal(theta, cov)
            prop_logp = model.log_posterior(prop, data)
            log_alpha = prop_logp - cur_logp
            if np.log(self.rng.random()) < log_alpha:
                theta = prop
                cur_logp = prop_logp
                n_accept += 1

            # Adapt only during burn-in
            if adapt and t < burn:
                acc = n_accept / max(1, t + 1)
                # Scale all steps together; robust/simple
                scale = np.exp(gamma * (acc - target_accept))
                cov *= scale

            if t >= burn and ((t - burn) % thin == 0):
                kept.append(theta.copy())
                kept_lp.append(cur_logp)

        draws = np.vstack(kept) if kept else np.empty((0, d))
        return {
            "draws": draws,
            "accept_rate": n_accept / max(1, n_draws),
            "logp": np.array(kept_lp, dtype=float),
        }

    def slice_univariate(
        self,
        model: BayesModel,
        data: object,
        theta0: Array,
        n_draws: int = 4000,
        burn: int = 1000,
        w: float | Array = 1.0,
        m: int = 50,
        thin: int = 1,
    ) -> Dict[str, object]:
        """Axis-wise univariate slice sampling with stepping-out and shrinkage.

        Args:
            model: Bayesian model.
            data: Data payload.
            theta0: Initial parameter vector (d,).
            n_draws: Total draws (including burn-in).
            burn: Burn-in draws to discard.
            w: Initial bracket width (scalar or vector per-dimension).
            m: Max stepping-out steps to avoid infinite loops.
            thin: Keep one every `thin` draws after burn-in.

        Returns:
            dict with keys: 'draws', 'logp'.
        """
        theta = _as_1d(theta0).copy()
        d = theta.size
        wv = np.full(d, float(w)) if np.isscalar(w) else _as_1d(w)

        def logp(v: Array) -> float:
            return model.log_posterior(v, data)

        lp = logp(theta)
        kept = []
        kept_lp = []

        for t in range(n_draws):
            for k in range(d):
                # Vertical level
                y = lp - np.random.exponential(1.0)
                # Create initial bracket
                r = self.rng.random()
                L = theta[k] - r * wv[k]
                R = L + wv[k]
                j = int(self.rng.integers(0, m))
                k_steps = m - 1 - j
                # Step out
                while j > 0 and logp(theta_at(theta, k, L)) > y:
                    L -= wv[k]
                    j -= 1
                while k_steps > 0 and logp(theta_at(theta, k, R)) > y:
                    R += wv[k]
                    k_steps -= 1
                # Shrinkage
                while True:
                    x = self.rng.uniform(L, R)
                    lp_x = logp(theta_at(theta, k, x))
                    if lp_x > y:
                        theta[k] = x
                        lp = lp_x
                        break
                    elif x < theta[k]:
                        L = x
                    else:
                        R = x

            if t >= burn and ((t - burn) % thin == 0):
                kept.append(theta.copy())
                kept_lp.append(lp)

        return {
            "draws": np.vstack(kept) if kept else np.empty((0, d)),
            "logp": np.array(kept_lp, dtype=float),
        }


def theta_at(theta: Array, k: int, val: float) -> Array:
    """Helper for slice sampler: replace theta[k] with val."""
    v = theta.copy()
    v[k] = val
    return v


# --------------------------- Posterior Summaries --------------------------- #
@dataclass
class PosteriorSummary:
    """Summaries, diagnostics, ROPE, Bayes factors, and PPC hooks."""

    draws: Array  # (n_draws, d)
    names: Optional[Iterable[str]] = None

    def table(
        self,
        cred: float = 0.95,
        rope: Optional[Tuple[float, float]] = None,
        split_chains: Optional[int] = None,
    ):
        """Compute a neat summary table for each parameter.

        Args:
            cred: Credible mass for HDI/intervals.
            rope: Region of practical equivalence as (low, high).
            split_chains: If provided, interpret draws as concatenated chains of
                equal length and compute R-hat across them.

        Returns:
            pandas.DataFrame (if pandas is available) else dict of arrays.
        """
        x = np.asarray(self.draws, dtype=float)
        n, d = x.shape
        names = list(self.names) if self.names is not None else [f"θ[{i}]" for i in range(d)]

        means = x.mean(axis=0)
        med = np.median(x, axis=0)
        std = x.std(axis=0, ddof=1)
        q025 = np.quantile(x, 0.5 * (1 - cred), axis=0)
        q975 = np.quantile(x, 1 - 0.5 * (1 - cred), axis=0)
        hdi_lo = np.empty(d)
        hdi_hi = np.empty(d)
        ess = np.empty(d)
        mcse = np.empty(d)

        for j in range(d):
            hdi_lo[j], hdi_hi[j] = _hdi(x[:, j], cred)
            ess[j] = _ess(x[:, j])
            mcse[j] = _mcse(x[:, j])

        rhat = np.full(d, np.nan)
        if split_chains is not None and split_chains > 1:
            chain_len = n // split_chains
            if chain_len >= 4:
                c = x[: split_chains * chain_len].reshape(split_chains, chain_len, d)
                for j in range(d):
                    rhat[j] = _rhat(c[..., j])

        rope_in = rope is not None
        rope_prob = np.full(d, np.nan)
        if rope_in:
            a, b = rope  # type: ignore
            mask = (x >= a) & (x <= b)
            rope_prob = mask.mean(axis=0)

        data = {
            "param": names,
            "mean": means,
            "sd": std,
            "mcse": mcse,
            "ess": ess,
            "rhat": rhat,
            f"{int(100*(1-cred)/2)}%": q025,
            "median": med,
            f"{int(100*(1+cred)/2)}%": q975,
            "hdi_low": hdi_lo,
            "hdi_high": hdi_hi,
        }
        if rope_in:
            data["rope_prob"] = rope_prob

        if pd is not None:
            return pd.DataFrame(data)
        return data

    def savage_dickey(
        self,
        index: int,
        null_value: float,
        prior_pdf: Callable[[float], float],
        kde_bw: Optional[float] = None,
    ) -> float:
        """Savage–Dickey Bayes factor for point null on parameter `index`.

        Args:
            index: Parameter dimension to test.
            null_value: Value at which to evaluate the null.
            prior_pdf: Function evaluating the prior density at `null_value`.
            kde_bw: Optional bandwidth for posterior density KDE (scott by default).

        Returns:
            Bayes factor BF_01 (evidence for H0 over H1).

        Notes:
            - Uses Gaussian KDE on MCMC draws to estimate posterior density.
            - Works only for point nulls on one dimension.
        """
        z = _as_1d(self.draws[:, index])
        n = z.size
        if n < 50:
            raise ValueError("Not enough draws for KDE density estimate.")
        # Scott’s rule
        bw = (kde_bw if kde_bw is not None else np.power(n, -1 / 5) * z.std(ddof=1))
        if bw <= 0:
            raise ValueError("Non-positive KDE bandwidth.")
        # Gaussian KDE at x0
        u = (null_value - z) / bw
        post_pdf = np.mean(np.exp(-0.5 * u * u) / (math.sqrt(2 * math.pi) * bw))
        prior_density = float(prior_pdf(null_value))
        if prior_density <= 0:
            return np.inf
        return float(prior_density / post_pdf)

    def posterior_predictive(
        self,
        pp_fn: PPFn,
        theta_draws: Optional[Array] = None,
        n_obs: int = 1,
        data: object = None,
    ) -> Array:
        """Generate posterior predictive draws for convenience.

        Args:
            pp_fn: Function mapping (θ, n_obs, data) -> draws.
            theta_draws: Optional subset of θ draws; else uses self.draws.
            n_obs: Number of predictive draws per θ (vectorized or looped).
            data: Original data to pass back to pp_fn if needed.

        Returns:
            Array of predictive draws stacked over θ.
        """
        thetas = self.draws if theta_draws is None else np.asarray(theta_draws)
        out = []
        for th in thetas:
            out.append(pp_fn(th, n_obs, data))
        return np.array(out, dtype=float)


# -------------------------- Conjugate Models ------------------------------ #
def beta_binomial_conjugate(
    a: float, b: float, k: int, n: int, size: int = 4000, rng: Optional[np.random.Generator] = None
) -> Array:
    """Posterior draws for θ ~ Beta(a,b), y ~ Bin(n,θ) with y=k."""
    rng = rng or np.random.default_rng()
    return rng.beta(a + k, b + n - k, size=size)

def normal_normal_known_sigma(
    mu0: float, tau0: float, ybar: float, n: int, sigma: float, size: int = 4000, rng: Optional[np.random.Generator] = None
) -> Array:
    """Posterior for μ | σ known; prior μ ~ N(mu0, tau0^2)."""
    rng = rng or np.random.default_rng()
    prec = 1 / tau0**2 + n / sigma**2
    mu_n = (mu0 / tau0**2 + n * ybar / sigma**2) / prec
    sd_n = math.sqrt(1 / prec)
    return rng.normal(mu_n, sd_n, size=size)

def normal_inverse_gamma(
    mu0: float, kappa0: float, alpha0: float, beta0: float, ybar: float, s2: float, n: int,
    size: int = 4000, rng: Optional[np.random.Generator] = None
) -> Tuple[Array, Array]:
    """Conjugate Normal–Inverse-Gamma posterior for (μ, σ^2)."""
    rng = rng or np.random.default_rng()
    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * ybar) / kappa_n
    alpha_n = alpha0 + n / 2
    beta_n = beta0 + 0.5 * (n * s2 + (kappa0 * n * (ybar - mu0) ** 2) / kappa_n)
    sigma2 = 1 / rng.gamma(alpha_n, 1 / beta_n, size=size)
    mu = rng.normal(mu_n, np.sqrt(sigma2 / kappa_n), size=size)
    return mu, sigma2


# --------------------------- High-level helpers --------------------------- #
def quick_bayes(
    model: BayesModel,
    data: object,
    theta0: Array,
    *,
    n_draws: int = 4000,
    burn: int = 1000,
    sampler: str = "slice",
    step: float = 0.2,
    w: float = 1.0,
    thin: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """One-call posterior sampler for any model.

    Args:
        model: Model with log_prior/log_likelihood defined.
        data: Data payload.
        theta0: Initial parameter vector (d,).
        n_draws: Total number of MCMC draws (includes burn-in).
        burn: Burn-in draws to discard.
        sampler: 'slice' (robust) or 'mh' (fast RW-MH).
        step: MH step size (if sampler='mh').
        w: Slice bracket width (if sampler='slice').
        thin: Keep one every `thin` draw after burn-in.
        rng: Optional NumPy Generator.

    Returns:
        dict with keys: 'draws', 'accept_rate' (MH), 'logp'.
    """
    s = Sampler(rng or np.random.default_rng())
    if sampler == "slice":
        return s.slice_univariate(model, data, theta0, n_draws=n_draws, burn=burn, w=w, thin=thin)
    elif sampler == "mh":
        return s.mh(model, data, theta0, n_draws=n_draws, burn=burn, step=step, thin=thin)
    else:
        raise ValueError("sampler must be 'slice' or 'mh'")

def summarize_posterior(
    draws: Array,
    names: Optional[Iterable[str]] = None,
    *,
    cred: float = 0.95,
    rope: Optional[Tuple[float, float]] = None,
    split_chains: Optional[int] = None,
):
    """Produce a tidy posterior summary table (with HDI, ESS, R-hat, MCSE).

    Args:
        draws: Array of shape (n_draws, d).
        names: Parameter names of length d.
        cred: Credible mass for intervals/HDI.
        rope: Optional ROPE tuple (low, high).
        split_chains: If provided (>1), treats draws as concatenated equal chains
            and computes split-R-hat.

    Returns:
        pandas.DataFrame (if pandas available) else dict.
    """
    return PosteriorSummary(draws, names).table(cred=cred, rope=rope, split_chains=split_chains)
