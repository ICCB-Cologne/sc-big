"""
Inference Module
================

Bayesian inference for somatic SNV detection in single cells using Pyro.

License
-------
This file is part of ``sc-big``.

Copyright (C) 2026 Daniel Schütte, daniel.schuette@iccb-cologne.org

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details. You should have received a copy of the GNU General Public
License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import torch
import pyro

import numpy as np
import pyro.distributions as dist
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from scipy.stats import betabinom, binom
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp as _logsumexp
from typing import Final
from pyro.infer import MCMC
from pyro.infer.mcmc import NUTS
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.logging_config import get_logger
from src.inference_prosolo import (
    prob_rho, lodato_af, ln_count_likelihood, SC_COVERAGE_CAP,
)

TAU_DEFAULT: Final[float] = 20.0
TAU_MINIMUM: Final[float] = 1.0

logger = get_logger(__name__)


def multiplicity_prior(m: int, C: int, prior_type: str = "geometric") -> float:
    """
    Compute prior probability for multiplicity.

    Parameters
    ----------
    m : int
        Multiplicity (number of variant copies per cell).
    C : int
        Total copy number at the locus.
    prior_type : str
        Type of prior distribution. Options:
        - "uniform": P(m | C) = 1/C for all m in {1, ..., C}
        - "geometric": P(m | C) ∝ 2^(-m), favoring lower multiplicities

    Returns
    -------
    float
        P(m | C)
    """
    if m < 1 or m > C:
        return 0.0

    if prior_type == "uniform":
        return 1.0 / C

    elif prior_type == "geometric":
        # P(m) ∝ 2^(-m), normalized over {1, ..., C}
        unnormalized = 0.5 ** m
        normalizer = sum(0.5 ** i for i in range(1, C + 1))
        return unnormalized / normalizer

    else:
        raise ValueError(
            f"Unknown prior_type '{prior_type}'. "
            f"Must be 'uniform' or 'geometric'."
        )


def estimate_ccf_prior_from_sc(
    single_cells: list[dict], min_coverage_threshold: int = 3,
) -> tuple[float, float]:
    """
    Estimate informative CCF prior from single-cell aggregate data.

    Uses a simple conjugate Beta prior based on the fraction of cells showing
    variant reads. This formulation provides natural regularization: with
    sparse data the prior approaches uniform Beta(1,1), while with more cells
    it tightens around the observed cellular prevalence.

    Note: Because single-cell genotyping suffers from allelic dropout, this
    empirical prior is conservative and may underestimate CCF at high values.

    Parameters
    ----------
    single_cells : list[dict]
        Single-cell data with 'k_sc' and 'n_sc' fields.
    min_coverage_threshold : int
        Minimum n_sc for a cell to be considered informative (default: 3).

    Returns
    -------
    tuple[float, float]
        (alpha, beta) for Beta prior on CCF. Returns (1,1) for uniform prior
        when no informative cells are available.
    """
    eligible_cells = [
        c for c in single_cells if c["n_sc"] >= min_coverage_threshold
    ]
    n_eligible = len(eligible_cells)

    if n_eligible == 0:
        logger.warning("No eligible cells for CCF prior, using uniform")
        return (1.0, 1.0)

    n_positive = sum(1 for c in eligible_cells if c["k_sc"] >= 1)
    n_negative = n_eligible - n_positive
    alpha = n_positive + 1.0
    beta = n_negative + 1.0

    ccf_estimate = n_positive / n_eligible
    logger.info(
        f"SC-informed CCF prior: {n_positive}/{n_eligible} cells positive "
        f"({ccf_estimate:.1%}), Beta({alpha:.0f}, {beta:.0f})"
    )

    return (alpha, beta)


@dataclass
class InferenceInputs:
    """Input data for Bayesian inference."""
    k_b: int
    n_b: int
    k_sc: int
    n_sc: int
    purity_mean: float = 0.8
    purity_std: float = 0.1
    copy_number_mean: float = 2.0
    copy_number_std: float = 0.5
    kappa: float = 100.0
    epsilon_sc: float = 0.10
    population_tau: float = TAU_DEFAULT
    multiplicity_prior_type: str = "geometric"
    error_model: str = "betabinomial"


def _estimate_tau_mle(
    germline_snps: list[tuple[int, int]]
) -> float | None:
    """
    Estimate tau using maximum likelihood.

    Since reference/alternate allele assignment is arbitrary, we fit a
    symmetric Beta-Binomial with p=0.5 fixed. This handles the mixture of both
    allelic orientations automatically.

    Parameters
    ----------
    germline_snps : list[tuple[int, int]]
        List of (k_g, n_g) tuples for germline heterozygous SNPs.

    Returns
    -------
    float
        MLE estimate of tau, or None if estimation fails.
    """
    if len(germline_snps) < 5:
        return None

    valid_snps = [(k, n) for k, n in germline_snps if n > 0]
    if len(valid_snps) < 5:
        return None

    k_array = np.array([k for k, n in valid_snps])
    n_array = np.array([n for k, n in valid_snps])

    def neg_log_lik(tau):
        if tau < 1.0:
            return 1e10
        alpha = beta = tau / 2.0
        try:
            log_liks = betabinom.logpmf(k_array, n_array, alpha, beta)
            return -np.sum(log_liks)
        except Exception:
            return 1e10

    result = minimize_scalar(
        neg_log_lik, bounds=(1.0, 500.0), method="bounded"
    )

    if result.success and 1.0 <= result.x <= 500.0:
        return result.x  # type: ignore
    return None


def compute_population_tau(
    all_germline_snps: list[list[tuple[int, int]]]
) -> tuple[float, int]:
    """
    Compute population tau from pooled germline SNPs.

    This function pools germline SNP data across all cells and estimates
    tau using MLE. The result is used directly for all cells. We used to do
    per-cell estimation, but low coverage and low number of germline SNPs
    made that extremely noisy.

    Parameters
    ----------
    all_germline_snps : list[list[tuple[int, int]]]
        List of germline SNP data for each cell. Each element is a list of
        (k_g, n_g) tuples representing variant/total reads for germline SNPs.

    Returns
    -------
    tuple[float, int]
        (tau, n_snps) where tau is the MLE estimate (or default 20.0 if
        estimation fails) and n_snps is the total number of SNPs used.
    """
    pooled_germline_snps = []
    for germline_snps in all_germline_snps:
        pooled_germline_snps.extend(germline_snps)

    n_snps = len(pooled_germline_snps)
    mle_estimate = _estimate_tau_mle(pooled_germline_snps)
    if mle_estimate is None:
        logger.warning(
            "Unable to calculate valid MLE estimate for pooled cells; "
            f"returning default tau={TAU_DEFAULT:.2f}"
        )
        return (TAU_DEFAULT, n_snps)

    logger.info(f"Population tau={mle_estimate:.2f} from {n_snps} SNPs")
    return (mle_estimate, n_snps)


def joint_model(
    k_b: int, n_b: int, purity_mean: float, purity_std: float,
    copy_number_mean: float, copy_number_std: float, kappa: float,
) -> None:
    """
    Joint generative model P(k_b, CCF, π, C, m).

    Samples from the prior and conditions on observed k_b.
    """
    ccf = pyro.sample("ccf", dist.Uniform(0.0, 1.0))

    purity = pyro.sample("purity", dist.Normal(purity_mean, purity_std))
    purity = torch.clamp(purity, 0.01, 0.99)

    copy_number_probs = torch.zeros(10)
    for c in range(1, 11):
        copy_number_probs[c - 1] = torch.exp(
            torch.tensor(
                -((c - copy_number_mean) ** 2) / (2 * copy_number_std ** 2)
            )
        )
    copy_number_probs = copy_number_probs / copy_number_probs.sum()
    copy_number = pyro.sample(
        "copy_number", dist.Categorical(copy_number_probs)
    ) + 1

    C = int(copy_number.item())
    mult_probs = torch.ones(C) / C
    multiplicity = pyro.sample(
        "multiplicity", dist.Categorical(mult_probs)
    ) + 1

    expected_vaf = (purity * ccf * multiplicity) / (
        purity * copy_number + 2 * (1 - purity)
    )
    expected_vaf = torch.clamp(expected_vaf, 0.001, 0.999)

    alpha = kappa * expected_vaf
    beta = kappa * (1 - expected_vaf)

    pyro.sample(
        "k_b", dist.BetaBinomial(alpha, beta, total_count=n_b),
        obs=torch.tensor(float(k_b))
    )


def continuous_joint_model(
    k_b: int, n_b: int, purity_mean: float, purity_std: float,
    copy_number_mean: float, copy_number_std: float, kappa: float,
    C: int, m: int, ccf_prior_alpha: float = 1.0, ccf_prior_beta: float = 1.0
) -> None:
    """
    Joint model with fixed discrete parameters (C, m).

    This version is used with MCMC to sample only continuous parameters.
    CCF prior defaults to Beta(1,1) = Uniform(0,1).
    """
    ccf = pyro.sample("ccf", dist.Beta(ccf_prior_alpha, ccf_prior_beta))
    purity = pyro.sample("purity", dist.Normal(purity_mean, purity_std))
    purity = torch.clamp(purity, 0.01, 0.99)

    expected_vaf = (purity * ccf * m) / (purity * C + 2 * (1 - purity))
    expected_vaf = torch.clamp(expected_vaf, 0.001, 0.999)

    alpha = kappa * expected_vaf
    beta = kappa * (1 - expected_vaf)

    pyro.sample(
        "k_b", dist.BetaBinomial(alpha, beta, total_count=n_b),
        obs=torch.tensor(float(k_b))
    )


def _run_single_combination(
    C, m, discrete_prior, n_samples, k_b, n_b, purity_mean, purity_std,
    copy_number_mean, copy_number_std, kappa, ccf_prior_alpha, ccf_prior_beta
):
    """
    Run MCMC inference for a single (C, m) combination with CCF prior.

    Returns
    -------
    dict with keys:
        - samples: list of (ccf, C, m, log_weight) tuples
        - trace: dict with 'ccf' and 'purity' arrays (sample traces)
        - diagnostics: dict with 'r_hat' and 'n_eff' for each parameter
        - C, m: the discrete state values
    """
    model_with_fixed_discrete = partial(
        continuous_joint_model, C=C, m=m,
        ccf_prior_alpha=ccf_prior_alpha, ccf_prior_beta=ccf_prior_beta
    )

    nuts_kernel = NUTS(model_with_fixed_discrete)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=n_samples,
        warmup_steps=int(0.2*n_samples),
        disable_progbar=True
    )

    try:
        logger.info(f"running {n_samples} MCMC steps for C={C}, m={m}")
        mcmc.run(
            k_b, n_b, purity_mean, purity_std,
            copy_number_mean, copy_number_std, kappa
        )
        samples_dict = mcmc.get_samples()
        diagnostics_raw = mcmc.diagnostics()
        diagnostics = {}
        for param in ["ccf", "purity"]:
            if param in diagnostics_raw:
                r_hat = diagnostics_raw[param]["r_hat"]
                n_eff = diagnostics_raw[param]["n_eff"]
                if hasattr(r_hat, "item"):
                    r_hat = r_hat.item()
                if hasattr(n_eff, "item"):
                    n_eff = n_eff.item()
                diagnostics[param] = {"r_hat": r_hat, "n_eff": n_eff}

        trace = {
            "ccf": samples_dict["ccf"].numpy(),
            "purity": samples_dict["purity"].numpy(),
        }

        results = []
        for i in range(len(samples_dict["ccf"])):
            ccf = samples_dict["ccf"][i].item()
            purity = samples_dict["purity"][i].item()
            purity = np.clip(purity, 0.01, 0.99)

            expected_vaf = (purity * ccf * m) / (purity * C + 2 * (1 - purity))
            expected_vaf = np.clip(expected_vaf, 0.001, 0.999)

            alpha = kappa * expected_vaf
            beta_param = kappa * (1 - expected_vaf)
            likelihood = betabinom.pmf(k_b, n_b, alpha, beta_param)

            # Weight = discrete prior × likelihood
            # This properly accounts for how well this (C, m, CCF, π) explains
            # the data.
            log_weight = np.log(discrete_prior) + np.log(likelihood + 1e-300)

            results.append((ccf, C, m, log_weight))

        return {
            "samples": results,
            "trace": trace,
            "diagnostics": diagnostics,
            "C": C,
            "m": m,
            "discrete_prior": discrete_prior,
        }

    except Exception as e:
        return f"MCMC failed for C={C}, m={m}: {e}"


def compute_bulk_posterior(
    k_b: int, n_b: int, purity_mean: float, purity_std: float,
    copy_number_mean: float, copy_number_std: float, kappa: float,
    n_samples: int = 5000, n_workers: int = 1,
    multiplicity_prior_type: str = "geometric",
    ccf_prior_alpha: float = 1.0, ccf_prior_beta: float = 1.0
) -> dict:
    """
    Compute bulk posterior P(CCF, π, C, m | k_b, n_b) using MCMC.

    This function should be called once per dataset and the results cached
    for reuse across all cells.

    Parameters
    ----------
    k_b : int
        Variant reads in bulk sample.
    n_b : int
        Total reads in bulk sample.
    purity_mean : float
        Mean estimate for bulk purity.
    purity_std : float
        Standard deviation for bulk purity.
    copy_number_mean : float
        Mean estimate for copy number.
    copy_number_std : float
        Standard deviation for copy number.
    kappa : float
        Concentration parameter for bulk overdispersion.
    n_samples : int
        Number of MCMC samples to generate (total across all discrete states).
    n_workers : int
        Number of parallel workers for MCMC.
    multiplicity_prior_type : str
        Type of multiplicity prior ("uniform" or "geometric").
    ccf_prior_alpha, ccf_prior_beta : float
        Beta prior parameters for CCF. Defaults to (1,1) = uniform prior.

    Returns
    -------
    dict with keys:
        - samples: list of (CCF, C, m, log_weight) tuples
        - mcmc_traces: list of dicts with trace data per (C, m) chain
    """
    pyro.clear_param_store()

    copy_number_probs = np.zeros(10)
    for c in range(1, 11):
        copy_number_probs[c - 1] = np.exp(
            -((c - copy_number_mean) ** 2) / (2 * copy_number_std ** 2)
        )
    copy_number_probs = copy_number_probs / copy_number_probs.sum()

    tasks = []
    for C in range(1, 11):
        p_C = copy_number_probs[C - 1]
        if p_C < 1e-6:
            continue

        for m in range(1, C + 1):
            p_m_given_C = multiplicity_prior(m, C, multiplicity_prior_type)
            discrete_prior = p_C * p_m_given_C
            tasks.append((C, m, discrete_prior))

    all_samples = []
    mcmc_traces = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                _run_single_combination, C, m, discrete_prior, n_samples,
                k_b, n_b, purity_mean, purity_std, copy_number_mean,
                copy_number_std, kappa, ccf_prior_alpha, ccf_prior_beta
            ) for (C, m, discrete_prior) in tasks
        ]

        for f in as_completed(futures):
            result = f.result()
            if isinstance(result, str):
                logger.warning(result)
            else:
                all_samples.extend(result["samples"])
                mcmc_traces.append({
                    "C": result["C"],
                    "m": result["m"],
                    "trace": result["trace"],
                    "diagnostics": result["diagnostics"],
                    "discrete_prior": result["discrete_prior"],
                })

    if len(all_samples) == 0:
        logger.warning("No valid bulk posterior samples from MCMC")
        return {"samples": [], "mcmc_traces": []}

    logger.info(
        f"Computed bulk posterior: {len(all_samples)} samples "
        f"across {len(mcmc_traces)} discrete states"
    )
    return {"samples": all_samples, "mcmc_traces": mcmc_traces}


def compute_variant_probability(
    inputs: InferenceInputs, n_samples: int = 5000,
    bulk_posterior: list[tuple[float, int, int, float]] | dict | None = None
) -> tuple[float, dict]:
    """
    Compute P(z=1 | k_sc, n_sc, D_b) using hierarchical Bayesian inference.

    Uses MCMC (NUTS kernel with Gibbs sampling for discrete parameters) to
    sample from the posterior P(CCF, π, C, m | k_b, n_b).

    Parameters
    ----------
    inputs : InferenceInputs
        All required input data.
    n_samples : int
        Number of MCMC samples from posterior (after warmup).
        Only used if bulk_posterior is None.
    bulk_posterior : list | dict | None
        Pre-computed bulk posterior from compute_bulk_posterior().
        Can be either the full dict (with "samples" key) or just the samples
        list for backward compatibility.

    Returns
    -------
    tuple
        Posterior probability that variant is present in single cell and dict
        with CCF samples, weights, and concentration.
    """
    pyro.clear_param_store()

    tau = inputs.population_tau
    if bulk_posterior is not None:
        if isinstance(bulk_posterior, dict):
            posterior_samples = bulk_posterior["samples"]
        else:
            posterior_samples = bulk_posterior
    else:
        logger.debug(
            "Computing bulk posterior via MCMC (should have been cached!)"
        )
        result = compute_bulk_posterior(
            inputs.k_b, inputs.n_b, inputs.purity_mean, inputs.purity_std,
            inputs.copy_number_mean, inputs.copy_number_std, inputs.kappa,
            n_samples, multiplicity_prior_type=inputs.multiplicity_prior_type
        )
        posterior_samples = result["samples"]

    if len(posterior_samples) == 0:
        logger.warning("No valid posterior samples")
        return (0.5, {
            "ccf_samples": np.array([]),
            "ccf_weights": np.array([]),
            "concentration": tau,
        })
    log_weights = np.array([lw for _, _, _, lw in posterior_samples])
    log_weights = log_weights - log_weights.max()
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    ccf_values = np.array([ccf for ccf, _, _, _ in posterior_samples])

    use_lodato = inputs.error_model == "lodato"
    if use_lodato:
        n_cap = min(inputs.n_sc, SC_COVERAGE_CAP)
        e = inputs.epsilon_sc
        _ln_rho: dict[tuple[float, int], float] = {}
        for af_val in (0.0, 0.5, 1.0):
            for k_s in range(n_cap + 1):
                _ln_rho[(af_val, k_s)] = prob_rho(af_val, n_cap, k_s)

        _ln_sc_lik: dict[int, float] = {}
        for k_s in range(n_cap + 1):
            rho_s = k_s / n_cap if n_cap > 0 else 0.0
            _ln_sc_lik[k_s] = ln_count_likelihood(
                inputs.k_sc, inputs.n_sc, rho_s, e,
            )

        _ln_lik_z0 = float(_logsumexp([
            _ln_rho[(0.0, k_s)] + _ln_sc_lik[k_s]
            for k_s in range(n_cap + 1)
        ]))

    weighted_prob_z1 = 0.0
    weighted_prob_total = 0.0
    for (ccf, C, m, _), weight in zip(posterior_samples, weights):
        prior_z1 = ccf
        prior_z0 = 1 - ccf

        if use_lodato:
            af = lodato_af(m, C)
            ln_lik_z1 = float(_logsumexp([
                _ln_rho[(af, k_s)] + _ln_sc_lik[k_s]
                for k_s in range(n_cap + 1)
            ]))
            likelihood_z1 = np.exp(ln_lik_z1)
            likelihood_z0 = np.exp(_ln_lik_z0)
        else:
            expected_vaf = m / C
            expected_vaf = max(0.001, min(0.999, expected_vaf))
            alpha_sc = tau * expected_vaf
            beta_sc = tau * (1 - expected_vaf)

            likelihood_z1 = betabinom.pmf(
                inputs.k_sc, inputs.n_sc, alpha_sc, beta_sc
            )

            likelihood_z0 = binom.pmf(
                inputs.k_sc, inputs.n_sc, inputs.epsilon_sc
            )

        prob_z1 = likelihood_z1 * prior_z1
        prob_z0 = likelihood_z0 * prior_z0
        prob_total = prob_z1 + prob_z0

        weighted_prob_z1 += (
            weight * (prob_z1 / prob_total if prob_total > 0 else 0)
        )
        weighted_prob_total += weight

    posterior = (
        weighted_prob_z1 / weighted_prob_total
        if weighted_prob_total > 0 else 0.5
    )

    logger.debug(
        f"P(z=1|D)={posterior:.3f} (MCMC with "
        f"{len(posterior_samples)} samples, tau={tau:.2f})"
    )

    details = {
        "ccf_samples": ccf_values,
        "ccf_weights": weights,
        "concentration": tau,
    }
    return posterior, details


def plot_inference_results(
    results: list[dict], output_file: str = "inference_results.png",
    decision_threshold: float = 0.5,
    hyperparameters: object | None = None
) -> None:
    """
    Plot inference results comparing Bayesian and threshold methods.

    Parameters
    ----------
    results : list[dict]
        List of inference results with keys: cell_id, posterior_prob,
        true_variant_present, k_sc, n_sc.
    output_file : str
        Path to save the plot.
    decision_threshold : float
        Posterior probability threshold for calling a variant as present.
    hyperparameters : object | None
        Hyperparameters for display
    """
    fig = plt.figure(figsize=(18, 5))

    bayesian_preds = [r["posterior_prob"] for r in results]
    true_labels = [r["true_variant_present"] for r in results]

    def compute_f1(preds, labels):
        tp = sum(1 for p, t in zip(preds, labels) if p and t)
        fp = sum(1 for p, t in zip(preds, labels) if p and not t)
        fn = sum(1 for p, t in zip(preds, labels) if not p and t)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return f1

    bayesian_thresholds = np.linspace(0, 1, 101)
    bayesian_f1_scores = []
    for thresh in bayesian_thresholds:
        preds = [p >= thresh for p in bayesian_preds]
        f1 = compute_f1(preds, true_labels)
        bayesian_f1_scores.append(f1)

    optimal_bayesian_idx = np.argmax(bayesian_f1_scores)
    optimal_bayesian_thresh = bayesian_thresholds[optimal_bayesian_idx]
    optimal_bayesian_f1 = bayesian_f1_scores[optimal_bayesian_idx]

    vaf_scores = [
        r["k_sc"] / r["n_sc"] if r["n_sc"] > 0 else 0 for r in results
    ]
    naive_thresholds = np.linspace(0, 1, 101)
    naive_f1_scores = []
    for thresh in naive_thresholds:
        preds = [vaf >= thresh for vaf in vaf_scores]
        f1 = compute_f1(preds, true_labels)
        naive_f1_scores.append(f1)

    optimal_naive_idx = np.argmax(naive_f1_scores)
    optimal_naive_thresh = naive_thresholds[optimal_naive_idx]
    optimal_naive_f1 = naive_f1_scores[optimal_naive_idx]

    gs = fig.add_gridspec(1, 3, wspace=0.3)

    # Plot 1: Posterior vs read count (Bayesian).
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(
        -0.15, 1.05, "a", transform=ax1.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )

    k_sc_values = [r["k_sc"] for r in results]
    all_k_sc = sorted(set(k_sc_values))
    probs_by_ksc_present = {k: [] for k in all_k_sc}
    probs_by_ksc_absent = {k: [] for k in all_k_sc}
    for r, p in zip(results, bayesian_preds):
        if r["true_variant_present"]:
            probs_by_ksc_present[r["k_sc"]].append(p)
        else:
            probs_by_ksc_absent[r["k_sc"]].append(p)

    violin_width = 0.45
    k_sc_with_data = []
    for k in all_k_sc:
        has_data = False

        if len(probs_by_ksc_present[k]) > 1:
            has_data = True
            parts = ax1.violinplot(
                probs_by_ksc_present[k], positions=[k - violin_width / 2],
                widths=violin_width, showmeans=True, showmedians=False
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("green")
                pc.set_alpha(0.7)
            for partname in ["cbars", "cmins", "cmaxes"]:
                if partname in parts:
                    parts[partname].set_color("green")
            if "cmeans" in parts:
                parts["cmeans"].set_color("green")
                parts["cmeans"].set_linewidth(2.5)
        elif len(probs_by_ksc_present[k]) == 1:
            # It's a single point! We can't draw a violin, so we show a marker
            # instead.
            has_data = True
            ax1.scatter(
                [k - violin_width / 2], probs_by_ksc_present[k],
                c="green", s=80, marker="o", alpha=0.7, zorder=5
            )

        if len(probs_by_ksc_absent[k]) > 1:
            has_data = True
            parts = ax1.violinplot(
                probs_by_ksc_absent[k], positions=[k + violin_width / 2],
                widths=violin_width, showmeans=True, showmedians=False
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("red")
                pc.set_alpha(0.7)
            for partname in ["cbars", "cmins", "cmaxes"]:
                if partname in parts:
                    parts[partname].set_color("red")
            if "cmeans" in parts:
                parts["cmeans"].set_color("red")
                parts["cmeans"].set_linewidth(2.5)
        elif len(probs_by_ksc_absent[k]) == 1:
            # Single point - can't draw violin, show marker instead
            has_data = True
            ax1.scatter(
                [k + violin_width / 2], probs_by_ksc_absent[k],
                c="red", s=80, marker="o", alpha=0.7, zorder=5
            )

        if has_data:
            k_sc_with_data.append(k)

    ax1.axhline(decision_threshold, color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(optimal_bayesian_thresh, color="black", linestyle="-",
                alpha=0.7)

    legend_handles = [
        Patch(facecolor="green", alpha=0.7, label="Variant truly present"),
        Patch(facecolor="red", alpha=0.7, label="Variant truly absent"),
        Line2D([0], [0], color="gray", linestyle="--", alpha=0.5,
               label="Fixed (P=0.5)"),
        Line2D([0], [0], color="black", linestyle="-", alpha=0.7,
               label=f"Optimal (P={optimal_bayesian_thresh:.2f})")
    ]
    ax1.legend(handles=legend_handles, loc="best", fontsize=8)

    ax1.set_xlabel(r"Variant reads ($k_{sc}$)")
    ax1.set_ylabel("P(z=1 | D)")
    ax1.set_title("Bayesian: Posterior vs Read Count")
    ax1.set_ylim(-0.05, 1.05)
    if all_k_sc:
        max_k = max(all_k_sc)
        ax1.set_xticks(range(max_k + 1))
        ax1.set_xlim(-0.7, max_k + 0.7)

    # Plot 2: F1 score across thresholds.
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(
        -0.15, 1.05, "b", transform=ax2.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    ax2.plot(bayesian_thresholds, bayesian_f1_scores, "g-", linewidth=2,
             label=f"Bayesian (max F1={optimal_bayesian_f1:.2f} at "
                   f"{optimal_bayesian_thresh:.2f})")
    ax2.plot(naive_thresholds, naive_f1_scores, "r--", linewidth=2,
             label=f"Naive threshold (max F1={optimal_naive_f1:.2f} at "
                   f"{optimal_naive_thresh:.2f})")

    ax2.scatter([optimal_bayesian_thresh], [optimal_bayesian_f1],
                color="green", s=100, zorder=5, marker="o")
    ax2.scatter([optimal_naive_thresh], [optimal_naive_f1],
                color="red", s=100, zorder=5, marker="o")

    ax2.set_xlabel("Threshold (Posterior Probability / VAF)")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 Score vs Threshold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: ROC Curve.
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(
        -0.15, 1.05, "c", transform=ax3.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )

    thresholds = np.linspace(1.01, 0, 102)
    bayesian_tprs = []
    bayesian_fprs = []
    for thresh in thresholds:
        preds = [p >= thresh for p in bayesian_preds]
        tp = sum(1 for p, t in zip(preds, true_labels) if p and t)
        fp = sum(1 for p, t in zip(preds, true_labels) if p and not t)
        tn = sum(1 for p, t in zip(preds, true_labels) if not p and not t)
        fn = sum(1 for p, t in zip(preds, true_labels) if not p and t)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        bayesian_tprs.append(tpr)
        bayesian_fprs.append(fpr)

    bayesian_tprs = np.array(bayesian_tprs)
    bayesian_fprs = np.array(bayesian_fprs)
    bayesian_roc_auc = np.trapezoid(bayesian_tprs, bayesian_fprs)

    vaf_scores = [r["k_sc"] / r["n_sc"] if r["n_sc"] > 0 else 0
                  for r in results]
    vaf_thresholds = np.linspace(1.01, 0, 102)
    naive_tprs = []
    naive_fprs = []
    for thresh in vaf_thresholds:
        preds = [vaf >= thresh for vaf in vaf_scores]
        tp = sum(1 for p, t in zip(preds, true_labels) if p and t)
        fp = sum(1 for p, t in zip(preds, true_labels) if p and not t)
        tn = sum(1 for p, t in zip(preds, true_labels) if not p and not t)
        fn = sum(1 for p, t in zip(preds, true_labels) if not p and t)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        naive_tprs.append(tpr)
        naive_fprs.append(fpr)

    naive_tprs = np.array(naive_tprs)
    naive_fprs = np.array(naive_fprs)
    naive_roc_auc = np.trapezoid(naive_tprs, naive_fprs)

    ax3.plot(bayesian_fprs, bayesian_tprs, "g-",
             linewidth=2, label=f"Bayesian (AUC={bayesian_roc_auc:.3f})")
    ax3.plot(naive_fprs, naive_tprs, "r--",
             linewidth=2, label=f"Naive threshold (AUC={naive_roc_auc:.3f})")
    ax3.plot([0, 1], [0, 1], "k:", linewidth=1, alpha=0.5,
             label="Random (AUC=0.500)")

    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_inference_diagnostics(
    results: list[dict], ccf_posterior: dict,
    bulk_posterior_samples: list[tuple] | None = None,
    population_tau: float | None = None,
    true_tau_mean: float | None = None,
    output_file: str = "inference_diagnostics.png"
) -> None:
    """
    Plot diagnostic plots for inference in a 2x3 grid.

    Parameters
    ----------
    results : list[dict]
        List of inference results with keys: cell_id, posterior_prob,
        true_variant_present, k_sc, n_sc, concentration.
    ccf_posterior : dict
        CCF posterior with keys: ccf_samples, ccf_weights.
    bulk_posterior_samples : list[tuple] | None
        Bulk posterior samples (CCF, C, m, log_weight) for multiplicity plot.
    population_tau : float | None
        Estimated population tau used for all cells.
    true_tau_mean : float | None
        True mean tau from simulation (for comparison).
    output_file : str
        Path to save the plot.
    """
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel a: CCF posterior distribution.
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(
        -0.15, 1.05, "a", transform=ax1.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    ax1.hist(
        ccf_posterior["ccf_samples"], bins=50,
        weights=ccf_posterior["ccf_weights"], alpha=0.7, color="blue",
        density=True
    )
    ccf_mean = np.average(
        ccf_posterior["ccf_samples"], weights=ccf_posterior["ccf_weights"]
    )
    ax1.axvline(
        ccf_mean, color="red", linestyle="--", label=f"Mean={ccf_mean:.3f}"
    )
    ax1.set_xlabel("CCF")
    ax1.set_ylabel("Density")
    ax1.set_title("CCF Posterior Distribution")
    ax1.legend()

    # Panel b: VAF distribution by true label.
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(
        -0.15, 1.05, "b", transform=ax2.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    vafs_present = [
        r["k_sc"] / r["n_sc"] if r["n_sc"] > 0 else 0
        for r in results if r["true_variant_present"]
    ]
    vafs_absent = [
        r["k_sc"] / r["n_sc"] if r["n_sc"] > 0 else 0
        for r in results if not r["true_variant_present"]
    ]
    ax2.hist(
        vafs_present, bins=10, alpha=0.6, color="green",
        label="Variant present"
    )
    ax2.hist(
        vafs_absent, bins=10, alpha=0.6, color="red", label="Variant absent"
    )
    ax2.set_xlabel("Observed VAF")
    ax2.set_ylabel("Count")
    ax2.set_title("VAF Distribution by True Label")
    ax2.legend()

    # Panel c: Calibration plot.
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(
        -0.15, 1.05, "c", transform=ax3.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    observed_freq = []
    for i in range(len(bins) - 1):
        in_bin = [
            r for r in results
            if bins[i] <= r["posterior_prob"] < bins[i + 1]
        ]
        if in_bin:
            freq = sum(r["true_variant_present"] for r in in_bin) / len(in_bin)
            observed_freq.append(freq)
        else:
            observed_freq.append(np.nan)

    ax3.scatter(bin_centers, observed_freq, s=100, alpha=0.7)
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax3.set_xlabel("Predicted probability")
    ax3.set_ylabel("Observed frequency")
    ax3.set_title("Calibration Plot")
    ax3.legend()
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)

    # Panel d: Population tau estimate display.
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.text(
        -0.15, 1.05, "d", transform=ax4.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")

    if population_tau is not None:
        text_lines = [
            r"Population $\tau$ Estimate",
            "",
            rf"$\hat{{\tau}}$ = {population_tau:.1f}",
        ]
        if true_tau_mean is not None:
            text_lines.append(rf"True $\bar{{\tau}}$ = {true_tau_mean:.1f}")
            error_pct = 100 * (population_tau - true_tau_mean) / true_tau_mean
            text_lines.append(f"Error = {error_pct:+.1f}%")

        box_style = "round,pad=0.5"
        box_props = dict(boxstyle=box_style, facecolor="lightblue", alpha=0.7)
        ax4.text(
            0.5, 0.5, "\n".join(text_lines),
            ha="center", va="center", fontsize=14,
            transform=ax4.transAxes, bbox=box_props
        )
    else:
        ax4.text(
            0.5, 0.5, "No tau data", ha="center", va="center",
            transform=ax4.transAxes
        )
    ax4.set_title("Population Tau")

    # Panel e: Multiplicity posterior.
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.text(
        -0.15, 1.05, "e", transform=ax5.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    if bulk_posterior_samples is not None and len(bulk_posterior_samples) > 0:
        log_weights = np.array([lw for _, _, _, lw in bulk_posterior_samples])
        log_weights = log_weights - log_weights.max()
        weights = np.exp(log_weights)
        weights = weights / weights.sum()
        multiplicities = np.array([m for _, _, m, _ in bulk_posterior_samples])

        unique_m = sorted(set(multiplicities))
        m_probs = []
        for m in unique_m:
            mask = multiplicities == m
            m_probs.append(weights[mask].sum())

        ax5.bar(unique_m, m_probs, alpha=0.7, color="purple")
        ax5.set_xlabel("Multiplicity (m)")
        ax5.set_ylabel("Posterior Probability")
        ax5.set_title("Multiplicity Posterior")
        ax5.set_xticks(unique_m)
    else:
        ax5.text(0.5, 0.5, "No multiplicity data", ha="center", va="center")
        ax5.set_title("Multiplicity Posterior")

    # Panel f: Either empty or copy number posterior if available.
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.text(
        -0.15, 1.05, "f", transform=ax6.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    if bulk_posterior_samples is not None and len(bulk_posterior_samples) > 0:
        log_weights = np.array([lw for _, _, _, lw in bulk_posterior_samples])
        log_weights = log_weights - log_weights.max()
        weights = np.exp(log_weights)
        weights = weights / weights.sum()
        copy_numbers = np.array([C for _, C, _, _ in bulk_posterior_samples])

        unique_C = sorted(set(copy_numbers))
        C_probs = []
        for C in unique_C:
            mask = copy_numbers == C
            C_probs.append(weights[mask].sum())

        ax6.bar(unique_C, C_probs, alpha=0.7, color="teal")
        ax6.set_xlabel("Copy Number (C)")
        ax6.set_ylabel("Posterior Probability")
        ax6.set_title("Copy Number Posterior")
        ax6.set_xticks(unique_C)
    else:
        ax6.text(0.5, 0.5, "No copy number data", ha="center", va="center")
        ax6.set_title("Copy Number Posterior")

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_mcmc_traces(
    mcmc_traces: list[dict],
    output_file: str = "mcmc_traces.png"
) -> None:
    """
    Plot MCMC trace plots for each (C, m) chain.

    Parameters
    ----------
    mcmc_traces : list[dict]
        List of trace data from compute_bulk_posterior(), each with keys:
        C, m, trace (dict with 'ccf' and 'purity' arrays), diagnostics.
    output_file : str
        Path to save the plot.
    """
    if not mcmc_traces:
        logger.warning("No MCMC traces to plot")
        return

    mcmc_traces = sorted(mcmc_traces, key=lambda x: (x["C"], x["m"]))
    n_chains = len(mcmc_traces)
    n_cols = min(3, n_chains)
    n_rows = (n_chains + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False
    )
    for idx, trace_data in enumerate(mcmc_traces):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        C = trace_data["C"]
        m = trace_data["m"]
        ccf_trace = trace_data["trace"]["ccf"]
        diagnostics = trace_data.get("diagnostics", {})

        ax.plot(ccf_trace, alpha=0.7, linewidth=0.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("CCF")
        title = f"C={C}, m={m}"
        if "ccf" in diagnostics:
            r_hat = diagnostics["ccf"].get("r_hat", np.nan)
            n_eff = diagnostics["ccf"].get("n_eff", np.nan)
            title += f"\n$\\hat{{R}}$={r_hat:.3f}, ESS={n_eff:.0f}"
        ax.set_title(title, fontsize=10)

        letter = chr(ord('a') + idx)
        ax.text(-0.15, 1.05, letter, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top")

    for idx in range(n_chains, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
