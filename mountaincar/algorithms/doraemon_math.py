from __future__ import annotations

from typing import Any, Mapping

import numpy as np

try:
    from scipy.special import betaln, digamma
except Exception as exc:  # pragma: no cover
    raise RuntimeError("DORAEMON methods require scipy.") from exc


def _bounds_keys(bounds: Mapping[str, tuple[float, float]]) -> list[str]:
    return [str(key) for key in bounds.keys()]


def _bounds_span(bounds: Mapping[str, tuple[float, float]]) -> np.ndarray:
    return np.asarray(
        [max(1e-12, float(high - low)) for low, high in bounds.values()],
        dtype=np.float64,
    )


def _coerce_vector(value: Any, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1 and dim > 1:
        return np.repeat(arr.astype(np.float64, copy=True), dim)
    if arr.size != dim:
        raise ValueError(f"Expected vector of size {dim}, got {arr.size}")
    return arr.astype(np.float64, copy=True)


def project_mu_to_bounds(mu: np.ndarray, bounds: Mapping[str, tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(mu, dtype=np.float64).reshape(-1)
    keys = _bounds_keys(bounds)
    if arr.size != len(keys):
        raise ValueError(f"Expected a {len(keys)}D context, got {arr.size}.")
    return np.asarray(
        [float(np.clip(value, bounds[key][0], bounds[key][1])) for value, key in zip(arr, keys)],
        dtype=np.float64,
    )


def normalize_context(context: np.ndarray, bounds: Mapping[str, tuple[float, float]]) -> np.ndarray:
    context_arr = project_mu_to_bounds(context, bounds)
    keys = _bounds_keys(bounds)
    lows = np.asarray([bounds[key][0] for key in keys], dtype=np.float64)
    span = _bounds_span(bounds)
    return (context_arr - lows) / np.clip(span, 1e-12, None)


def denormalize_context(z: np.ndarray, bounds: Mapping[str, tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64).reshape(-1)
    keys = _bounds_keys(bounds)
    if arr.size != len(keys):
        raise ValueError(f"Expected normalized vector of size {len(keys)}, got {arr.size}.")
    lows = np.asarray([bounds[key][0] for key in keys], dtype=np.float64)
    span = _bounds_span(bounds)
    return lows + arr * span


def normalize_params(force: float, gravity: float, bounds: Mapping[str, tuple[float, float]]) -> np.ndarray:
    return normalize_context(np.asarray([force, gravity], dtype=np.float64), bounds)


def denormalize_params(z: np.ndarray, bounds: Mapping[str, tuple[float, float]]) -> tuple[float, float]:
    arr = denormalize_context(z, bounds)
    if arr.size != 2:
        raise ValueError("denormalize_params is a 2D compatibility wrapper; use denormalize_context for ND.")
    return float(arr[0]), float(arr[1])


def beta_from_mean_conc(mean: np.ndarray, conc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_arr = np.clip(np.asarray(mean, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    conc_arr = np.clip(np.asarray(conc, dtype=np.float64), 2.0001, None)
    alpha = mean_arr * conc_arr
    beta = (1.0 - mean_arr) * conc_arr
    return alpha, beta


def mean_conc_from_beta(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha_arr = np.asarray(alpha, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)
    conc = alpha_arr + beta_arr
    mean = alpha_arr / np.clip(conc, 1e-8, None)
    return mean, conc


def make_beta_dist_at_mu(
    mu: np.ndarray,
    conc: np.ndarray,
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    mu_clipped = project_mu_to_bounds(mu, bounds)
    z = normalize_context(mu_clipped, bounds)
    conc_vec = _coerce_vector(conc, dim=z.size)
    alpha, beta = beta_from_mean_conc(z, conc_vec)
    return {"alpha": alpha, "beta": beta, "bounds": dict(bounds)}


def beta_logpdf(z: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    zc = np.clip(np.asarray(z, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    alpha_arr = np.asarray(alpha, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)
    return (alpha_arr - 1.0) * np.log(zc) + (beta_arr - 1.0) * np.log(1.0 - zc) - betaln(alpha_arr, beta_arr)


def beta_entropy(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    alpha_arr = np.asarray(alpha, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)
    return (
        betaln(alpha_arr, beta_arr)
        - (alpha_arr - 1.0) * digamma(alpha_arr)
        - (beta_arr - 1.0) * digamma(beta_arr)
        + (alpha_arr + beta_arr - 2.0) * digamma(alpha_arr + beta_arr)
    )


def dist_entropy(dist: Mapping[str, Any]) -> float:
    ent = beta_entropy(dist["alpha"], dist["beta"])
    return float(np.sum(ent) + np.sum(np.log(_bounds_span(dist["bounds"]))))


def beta_kl(a0: np.ndarray, b0: np.ndarray, a1: np.ndarray, b1: np.ndarray) -> np.ndarray:
    return (
        betaln(a1, b1)
        - betaln(a0, b0)
        + (a0 - a1) * digamma(a0)
        + (b0 - b1) * digamma(b0)
        + (a1 + b1 - a0 - b0) * digamma(a0 + b0)
    )


def dist_kl(old: Mapping[str, Any], new: Mapping[str, Any]) -> float:
    return float(np.sum(beta_kl(old["alpha"], old["beta"], new["alpha"], new["beta"])))


def sample_beta_params(dist: Mapping[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    z = rng.beta(dist["alpha"], dist["beta"])
    z = np.clip(z, 1e-6, 1.0 - 1e-6)
    context = denormalize_context(z, dist["bounds"])
    return context.astype(np.float64), z.astype(np.float64)


def is_success_estimate(
    z_samples: np.ndarray,
    success: np.ndarray,
    old_dist: Mapping[str, Any],
    new_dist: Mapping[str, Any],
    min_log_w: float = -20.0,
    max_log_w: float = 20.0,
) -> float:
    if len(z_samples) == 0:
        return 0.0

    z = np.clip(np.asarray(z_samples, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    success_arr = np.asarray(success, dtype=np.float64)

    logp_old = np.sum(beta_logpdf(z, old_dist["alpha"], old_dist["beta"]), axis=1)
    logp_new = np.sum(beta_logpdf(z, new_dist["alpha"], new_dist["beta"]), axis=1)

    logw = np.clip(logp_new - logp_old, min_log_w, max_log_w)
    w = np.exp(logw - np.max(logw))
    w = w / np.clip(np.sum(w), 1e-12, None)
    return float(np.sum(w * success_arr))


def doraemon_update_beta(
    old_dist: Mapping[str, Any],
    z_samples: np.ndarray,
    success_samples: np.ndarray,
    alpha_target: float,
    kl_max: float,
    rng: np.random.Generator,
    n_candidates: int,
    mean_std: float,
    logc_std: float,
    c_min: np.ndarray,
    c_max: np.ndarray,
    min_log_w: float,
    max_log_w: float,
    freeze_mean: bool,
) -> tuple[dict[str, Any], dict[str, float | str | dict[str, Any]]]:
    mean_old, conc_old = mean_conc_from_beta(old_dist["alpha"], old_dist["beta"])
    dim = int(np.asarray(mean_old, dtype=np.float64).reshape(-1).size)
    c_min_arr = _coerce_vector(c_min, dim=dim)
    c_max_arr = _coerce_vector(c_max, dim=dim)

    candidates: list[tuple[np.ndarray, np.ndarray]] = [(mean_old.copy(), conc_old.copy())]
    for _ in range(int(n_candidates)):
        if freeze_mean:
            mean = mean_old.copy()
        else:
            mean = np.clip(mean_old + rng.normal(0.0, mean_std, size=dim), 1e-4, 1.0 - 1e-4)

        conc = np.exp(np.log(np.clip(conc_old, 1e-8, None)) + rng.normal(0.0, logc_std, size=dim))
        conc = np.clip(conc, c_min_arr, c_max_arr)
        candidates.append((mean, conc))

    best_feasible: dict[str, float | str | dict[str, Any]] | None = None
    backup: dict[str, float | str | dict[str, Any]] | None = None

    for mean, conc in candidates:
        alpha_new, beta_new = beta_from_mean_conc(mean, conc)
        candidate_dist = {"alpha": alpha_new, "beta": beta_new, "bounds": dict(old_dist["bounds"])}

        kl_value = dist_kl(old_dist, candidate_dist)
        if kl_value > float(kl_max):
            continue

        g_hat = is_success_estimate(
            z_samples=z_samples,
            success=success_samples,
            old_dist=old_dist,
            new_dist=candidate_dist,
            min_log_w=min_log_w,
            max_log_w=max_log_w,
        )
        entropy = dist_entropy(candidate_dist)

        if (backup is None) or (g_hat > float(backup["g_hat"])) or (
            np.isclose(g_hat, float(backup["g_hat"])) and entropy > float(backup["entropy"])
        ):
            backup = {
                "dist": candidate_dist,
                "g_hat": float(g_hat),
                "entropy": float(entropy),
                "kl": float(kl_value),
                "mode": "backup_success",
            }

        if g_hat >= float(alpha_target):
            if (best_feasible is None) or (entropy > float(best_feasible["entropy"])):
                best_feasible = {
                    "dist": candidate_dist,
                    "g_hat": float(g_hat),
                    "entropy": float(entropy),
                    "kl": float(kl_value),
                    "mode": "feasible_entropy",
                }

    if best_feasible is not None:
        return best_feasible["dist"], best_feasible

    if backup is not None:
        return backup["dist"], backup

    fallback = {
        "dist": dict(old_dist),
        "g_hat": 0.0,
        "entropy": float(dist_entropy(old_dist)),
        "kl": 0.0,
        "mode": "no_candidate",
    }
    return dict(old_dist), fallback


__all__ = [
    "beta_entropy",
    "beta_from_mean_conc",
    "beta_kl",
    "beta_logpdf",
    "denormalize_context",
    "denormalize_params",
    "dist_entropy",
    "dist_kl",
    "doraemon_update_beta",
    "is_success_estimate",
    "make_beta_dist_at_mu",
    "mean_conc_from_beta",
    "normalize_context",
    "normalize_params",
    "project_mu_to_bounds",
    "sample_beta_params",
]
