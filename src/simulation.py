"""
Simulation Module
=================

This module implements a simulator that generates ground truth data for testing
the Bayesian model.

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
import json

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import cast
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Hyperparameters:
    """Model hyperparameters for simulation."""
    true_purity: float = 0.8
    true_copy_number: int = 2
    true_multiplicity: int = 1
    true_ccf: float = 0.5
    kappa: float = 100.0
    epsilon_sc: float = 0.10
    n_germline_snps: int = 20
    n_b: int = 100
    sc_coverage: int = 5
    base_concentration_mean: float = 30.0
    base_concentration_cv: float = 0.3


@dataclass
class BulkData:
    """Bulk sequencing observations and ground truth."""
    k_b: int
    n_b: int
    true_purity: float
    true_copy_number: int
    true_multiplicity: int
    true_ccf: float
    true_expected_vaf: float


@dataclass
class SingleCellData:
    """Single cell observations and ground truth."""
    k_sc: int
    n_sc: int
    germline_snps: list[tuple[int, int]]
    true_variant_present: bool
    true_sc_alpha: float
    true_sc_beta: float
    true_concentration: float


def draw_concentration_from_gamma(mean: float, cv: float) -> float:
    """
    Draw concentration parameter tau from Gamma distribution.

    Parameterized by mean and coefficient of variation (CV = std/mean).

    Parameters
    ----------
    mean : float
        Mean concentration parameter across cells
    cv : float
        Coefficient of variation (inter-cell variability)

    Returns
    -------
    float
        Drawn concentration parameter tau
    """
    shape = 1.0 / (cv ** 2)
    scale = mean * (cv ** 2)
    return np.random.gamma(shape, scale)


class BayesianSNVSimulator:
    """Simulator for Bayesian SNV detection model."""

    def __init__(
        self, hyperparams: Hyperparameters | None = None, seed: int = 42
    ) -> None:
        self.hyperparams = hyperparams or Hyperparameters()
        np.random.seed(seed)

    def simulate_dataset(
        self, n_single_cells: int
    ) -> tuple[BulkData, list[SingleCellData]]:
        """Simulate complete dataset with bulk and multiple single cells."""
        bulk_data = self._simulate_bulk_data()

        single_cells = []
        for _ in range(n_single_cells):
            sc_data = self._simulate_single_cell_data(bulk_data)
            single_cells.append(sc_data)

        return bulk_data, single_cells

    def _simulate_bulk_data(self) -> BulkData:
        """
        Simulate bulk sequencing data for variant site.

        The only probabilistic parameter is `k_b` which is generated based on
        a betabinomial distribution.
        """
        true_purity = self.hyperparams.true_purity
        true_copy_number = self.hyperparams.true_copy_number
        true_multiplicity = self.hyperparams.true_multiplicity
        true_ccf = self.hyperparams.true_ccf

        if true_multiplicity > true_copy_number:
            raise ValueError(
                f"Multiplicity ({true_multiplicity}) cannot "
                f"exceed copy number ({true_copy_number})")
        if not (0 <= true_ccf <= 1):
            raise ValueError(f"CCF ({true_ccf}) must be between 0 and 1")

        true_expected_vaf = self._calculate_expected_vaf(
            true_purity, true_ccf, true_multiplicity, true_copy_number
        )
        n_b = self.hyperparams.n_b
        k_b = self._sample_beta_binomial(
            n_b, true_expected_vaf, self.hyperparams.kappa
        )

        return BulkData(
            k_b=k_b,
            n_b=n_b,
            true_purity=true_purity,
            true_copy_number=true_copy_number,
            true_multiplicity=true_multiplicity,
            true_ccf=true_ccf,
            true_expected_vaf=true_expected_vaf
        )

    def _simulate_single_cell_data(
        self, bulk_data: BulkData
    ) -> SingleCellData:
        """
        Simulate single cell data given bulk parameters.

        Uses symmetric Beta-Binomial for germline SNPs (p=0.5) and the same
        concentration parameter tau for somatic variants. This approach
        naturally models amplification bias.
        """
        variant_present = np.random.random() < bulk_data.true_ccf
        tau_cell = self._sample_sc_concentration()
        germline_snps = self._simulate_germline_snps(tau_cell)
        n_sc = np.random.poisson(self.hyperparams.sc_coverage)

        if variant_present:
            expected_sc_vaf = (
                bulk_data.true_multiplicity / bulk_data.true_copy_number
            )
            true_sc_alpha = tau_cell * expected_sc_vaf
            true_sc_beta = tau_cell * (1.0 - expected_sc_vaf)

            if n_sc > 0:
                k_sc = self._sample_beta_binomial_with_params(
                    n_sc, true_sc_alpha, true_sc_beta
                )
            else:
                k_sc = 0
        else:
            true_sc_alpha = tau_cell * 0.5
            true_sc_beta = tau_cell * 0.5

            if n_sc > 0:
                k_sc = np.random.binomial(n_sc, self.hyperparams.epsilon_sc)
            else:
                k_sc = 0

        return SingleCellData(
            k_sc=k_sc,
            n_sc=n_sc,
            germline_snps=germline_snps,
            true_variant_present=variant_present,
            true_sc_alpha=true_sc_alpha,
            true_sc_beta=true_sc_beta,
            true_concentration=tau_cell
        )

    def _calculate_expected_vaf(
        self, purity: float, ccf: float, multiplicity: int, copy_number: int
    ) -> float:
        """Calculate expected VAF from model parameters."""
        numerator = purity * ccf * multiplicity
        denominator = purity * copy_number + 2 * (1 - purity)
        return numerator / denominator

    def _sample_sc_concentration(self) -> float:
        """
        Sample cell-specific concentration parameter (tau).

        The higher the concentration, the lower the amplification bias.
        """
        concentration = draw_concentration_from_gamma(
            mean=self.hyperparams.base_concentration_mean,
            cv=self.hyperparams.base_concentration_cv
        )
        return max(concentration, 1.0)

    def _sample_beta_binomial(
        self, n: int, expected_vaf: float, concentration: float
    ) -> int:
        """Sample from beta-binomial with expected VAF and concentration."""
        alpha = concentration * expected_vaf
        beta = concentration * (1 - expected_vaf)

        if alpha <= 0 or beta <= 0 or expected_vaf <= 0 or expected_vaf >= 1:
            return np.random.binomial(n, max(0.01, min(0.99, expected_vaf)))

        p = np.random.beta(alpha, beta)
        return np.random.binomial(n, p)

    def _sample_beta_binomial_with_params(
        self, n: int, alpha: float, beta: float
    ) -> int:
        """Sample from beta-binomial with explicit alpha/beta parameters."""
        if n == 0:
            return 0

        if alpha <= 0 or beta <= 0:
            p = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            p = max(0.01, min(0.99, p))
            return np.random.binomial(n, p)

        p = np.random.beta(alpha, beta)
        return np.random.binomial(n, p)

    def _simulate_germline_snps(
        self, concentration: float
    ) -> list[tuple[int, int]]:
        """
        Simulate germline heterozygous SNPs using symmetric Beta-Binomial.

        For heterozygous germline SNPs, the true expected VAF is 0.5.
        We use symmetric Beta-Binomial with alpha=beta=tau/2, where tau
        is the concentration parameter.

        Parameters
        ----------
        concentration : float
            Cell-specific concentration parameter (tau)

        Returns
        -------
        list[tuple[int, int]]
            List of (k_reads, n_reads) tuples for germline SNPs
        """
        germline_snps: list[tuple[int, int]] = []
        alpha_germline = beta_germline = concentration / 2.0

        for _ in range(self.hyperparams.n_germline_snps):
            n_reads = np.random.poisson(self.hyperparams.sc_coverage)
            if n_reads > 0:
                alt_reads = self._sample_beta_binomial_with_params(
                    n_reads, alpha_germline, beta_germline
                )
                germline_snps.append((alt_reads, n_reads))

        return germline_snps


def print_simulation_summary(
    bulk_data: BulkData, single_cells: list[SingleCellData]
) -> None:
    """Print summary statistics of simulated dataset."""
    logger.debug("=== Simulation Summary ===")
    logger.debug("Bulk Data:")
    logger.debug(f"  Coverage: {bulk_data.n_b}")
    logger.debug(f"  Variant reads: {bulk_data.k_b}")
    logger.debug(f"  Observed VAF: {bulk_data.k_b/bulk_data.n_b:.3f}")
    logger.debug(f"  True VAF: {bulk_data.true_expected_vaf:.3f}")
    logger.debug(f"  True CCF: {bulk_data.true_ccf:.3f}")
    logger.debug(f"  True purity: {bulk_data.true_purity:.3f}")

    n_variant_present = sum(sc.true_variant_present for sc in single_cells)
    logger.debug(f"Single Cells (n={len(single_cells)}):")
    logger.debug(f"  Variant present: {n_variant_present}/{len(single_cells)}")
    logger.debug(f"  Expected: {bulk_data.true_ccf * len(single_cells):.1f}")

    mean_coverage = np.mean([sc.n_sc for sc in single_cells])
    mean_alt_reads = np.mean([sc.k_sc for sc in single_cells])
    expected_somatic_vaf = (
        bulk_data.true_multiplicity / bulk_data.true_copy_number
    )
    logger.debug(f"  Mean coverage: {mean_coverage:.1f}")
    logger.debug(f"  Mean alt reads: {mean_alt_reads:.1f}")
    logger.debug(
        f"  Expected somatic VAF: {expected_somatic_vaf:.3f} "
        f"(m={bulk_data.true_multiplicity}, C={bulk_data.true_copy_number})"
    )


def plot_simulation_results(
    bulk_data: BulkData,
    single_cells: list[SingleCellData],
    output_file: str,
    hyperparameters: Hyperparameters | None = None
) -> None:
    """
    Plot simulation results with key diagnostics and hyperparameters.

    Parameters
    ----------
    bulk_data : BulkData
        Bulk sequencing data
    single_cells : list[SingleCellData]
        List of single cell data
    output_file : str
        Path to save plot
    hyperparameters : Hyperparameters | None
        Hyperparameters used for simulation (for display in title)
    """
    fig = plt.figure(figsize=(14, 10))

    title_parts = [
        f"Simulation: n={len(single_cells)} cells",
        f"CCF={bulk_data.true_ccf:.2f}",
        f"π={bulk_data.true_purity:.2f}",
        f"C={bulk_data.true_copy_number}",
        f"m={bulk_data.true_multiplicity}"
    ]
    if hyperparameters is not None:
        title_parts.append(f"cov={hyperparameters.sc_coverage}")
    fig.suptitle(" | ".join(title_parts), fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, top=0.93)

    # Plot 1: VAF distributions (variant present vs absent).
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(
        -0.15, 1.05, "a", transform=ax1.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    variant_present = [sc for sc in single_cells if sc.true_variant_present]
    variant_absent = [sc for sc in single_cells if not sc.true_variant_present]
    vaf_present = [sc.k_sc / max(sc.n_sc, 1) for sc in variant_present]
    vaf_absent = [sc.k_sc / max(sc.n_sc, 1) for sc in variant_absent]

    ax1.hist(
        vaf_present, bins=15, alpha=0.7, label="Variant Present", color="green"
    )
    ax1.hist(
        vaf_absent, bins=15, alpha=0.7, label="Variant Absent", color="red"
    )
    ax1.axvline(
        bulk_data.true_multiplicity / bulk_data.true_copy_number, color="blue",
        linestyle="--", label="Expected SC VAF= "
        f"{bulk_data.true_multiplicity}/{bulk_data.true_copy_number}"
    )
    ax1.set_xlabel("Observed VAF")
    ax1.set_ylabel("Count")
    ax1.set_title(
        f"VAF Distribution (n+ ={len(variant_present)}, "
        f"n−={len(variant_absent)})")
    ax1.legend()

    # Plot 2: CCF comparison.
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(
        -0.15, 1.05, "b", transform=ax2.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    observed_ccf = \
        sum(sc.true_variant_present for sc in single_cells) / len(single_cells)
    ax2.bar(
        ["True CCF", "Observed CCF"],
        [bulk_data.true_ccf, observed_ccf],
        color=["steelblue", "coral"],
        alpha=0.7
    )
    ax2.set_ylabel("CCF")
    ax2.set_title("CCF: True vs Observed")
    ax2.set_ylim(0, 1)
    ax2.text(
        0, bulk_data.true_ccf + 0.05, f"{bulk_data.true_ccf:.3f}",
        ha="center", fontsize=10
    )
    ax2.text(
        1, observed_ccf + 0.05, f"{observed_ccf:.3f}",
        ha="center", fontsize=10
    )

    # Plot 3: Germline VAF distribution (shows amplification bias).
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.text(
        -0.15, 1.05, "c", transform=ax3.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    all_germline_vafs = []
    for sc in single_cells:
        for alt_reads, total_reads in sc.germline_snps:
            if total_reads > 0:
                all_germline_vafs.append(alt_reads / total_reads)

    if all_germline_vafs:
        ax3.hist(all_germline_vafs, bins=20, alpha=0.7, color="purple")
        ax3.axvline(0.5, color="black", linestyle="--", label="Expected (0.5)")
        mean_germ = np.mean(all_germline_vafs)
        ax3.axvline(
            mean_germ, color="red", linestyle="--",  # type: ignore
            label=f"Observed ({mean_germ:.3f})"
        )
        ax3.set_xlabel("Germline SNP VAF")
        ax3.set_ylabel("Count (SNPs)")
        ax3.set_title(f"Germline SNPs (n={len(all_germline_vafs)})")
        ax3.legend()
    else:
        ax3.text(
            0.5, 0.5, "No germline SNPs", ha="center", va="center", fontsize=12
        )
        ax3.set_title("Germline SNPs")

    # Plot 4: Concentration distribution (shows cell-to-cell variability).
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.text(
        -0.15, 1.05, "d", transform=ax4.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    effective_concentrations = [
        sc.true_concentration for sc in single_cells
    ]
    ax4.hist(effective_concentrations, bins=15, alpha=0.7, color="orange")
    mean_conc = np.mean(effective_concentrations)
    ax4.axvline(
        mean_conc,  # type: ignore
        color="red", linestyle="--", label=f"Mean={mean_conc:.1f}"
    )
    ax4.set_xlabel("Effective Concentration τ")
    ax4.set_ylabel("Count (cells)")
    ax4.set_title("Cell-Specific Amplification Bias")
    ax4.legend()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def export_simulation(
    bulk_data: BulkData,
    single_cells: list[SingleCellData],
    hyperparameters: Hyperparameters,
    output_file: str | None = None
) -> dict[str, object]:
    """
    Export simulated data in format suitable for inference module.

    Parameters
    ----------
    bulk_data : BulkData
        Bulk sequencing data and ground truth
    single_cells : list[SingleCellData]
        List of single cell data
    hyperparameters : Hyperparameters
        Hyperparameters used for simulation
    output_file : str | None
        Path to save JSON file. If None, data is not saved to disk.

    Returns
    -------
    dict[str, object]
        Dictionary containing all simulation data including hyperparameters

    Saves to disk only if `output_file` is not None.
    """
    data: dict[str, object] = {
        "bulk": {
            "k_b": bulk_data.k_b,
            "n_b": bulk_data.n_b,
        },
        "single_cells": [
            {
                "k_sc": sc.k_sc,
                "n_sc": sc.n_sc,
                "germline_snps": [
                    [alt, total] for alt, total in sc.germline_snps
                ],
            }
            for sc in single_cells
        ],
        "hyperparameters": {
            "true_purity": hyperparameters.true_purity,
            "true_copy_number": hyperparameters.true_copy_number,
            "true_multiplicity": hyperparameters.true_multiplicity,
            "true_ccf": hyperparameters.true_ccf,
            "kappa": hyperparameters.kappa,
            "epsilon_sc": hyperparameters.epsilon_sc,
            "n_germline_snps": hyperparameters.n_germline_snps,
            "n_b": hyperparameters.n_b,
            "sc_coverage": hyperparameters.sc_coverage,
            "base_concentration_mean": hyperparameters.base_concentration_mean,
            "base_concentration_cv": hyperparameters.base_concentration_cv,
        },
        "ground_truth": {
            "bulk": {
                "ccf": bulk_data.true_ccf,
                "purity": bulk_data.true_purity,
                "copy_number": bulk_data.true_copy_number,
                "multiplicity": bulk_data.true_multiplicity,
                "expected_vaf": bulk_data.true_expected_vaf,
            },
            "single_cells": [
                {
                    "variant_present": sc.true_variant_present,
                    "alpha": sc.true_sc_alpha,
                    "beta": sc.true_sc_beta,
                    "effective_concentration": sc.true_concentration,
                }
                for sc in single_cells
            ],
        }
    }

    if output_file is not None:
        try:
            with open(output_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Simulation data export failed: {e}")
            raise ValueError(f"Simulation data export failed: {e}")

    return data


def import_simulation_from_json(
    path: str
) -> tuple[BulkData, list[SingleCellData], Hyperparameters]:
    """
    Import data from a JSON file back into simulation data structures.

    Parameters
    ----------
    path : str
        Path to JSON file

    Returns
    -------
    tuple[BulkData, list[SingleCellData], Hyperparameters]
        Bulk data, list of single cell data, and hyperparameters

    This function can be used by inference modules to load simulated data.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Simulation data import failed: {e}")
        raise

    return import_simulation_from_dict(data)


def import_simulation_from_dict(
    data: dict[str, int | float | list[object] | dict[str, object]]
) -> tuple[BulkData, list[SingleCellData], Hyperparameters]:
    """
    Import data from Python `dict` back into simulation data structures.

    Parameters
    ----------
    data : dict
        Dictionary containing simulation data

    Returns
    -------
    tuple[BulkData, list[SingleCellData], Hyperparameters]
        Bulk data, list of single cell data, and hyperparameters

    This function is mostly used by `import_simulation_from_json`.
    """
    bulk_dict: dict[str, int] = cast(dict[str, int], data["bulk"])
    ground_truth: dict[str, object] = \
        cast(dict[str, object], data["ground_truth"])
    bulk_gt: dict[str, int | float] = \
        cast(dict[str, int | float], ground_truth["bulk"])

    if "hyperparameters" not in data:
        raise ValueError("Hyperparameters not found in simulation data")

    hyper_dict: dict[str, int | float] = \
        cast(dict[str, int | float], data["hyperparameters"])

    hyperparameters = Hyperparameters(
        true_purity=float(hyper_dict["true_purity"]),
        true_copy_number=int(hyper_dict["true_copy_number"]),
        true_multiplicity=int(hyper_dict["true_multiplicity"]),
        true_ccf=float(hyper_dict["true_ccf"]),
        kappa=float(hyper_dict["kappa"]),
        epsilon_sc=float(hyper_dict["epsilon_sc"]),
        n_germline_snps=int(hyper_dict["n_germline_snps"]),
        n_b=int(hyper_dict["n_b"]),
        sc_coverage=int(hyper_dict["sc_coverage"]),
        base_concentration_mean=float(hyper_dict["base_concentration_mean"]),
        base_concentration_cv=float(hyper_dict["base_concentration_cv"]),
    )

    bulk_data = BulkData(
        k_b=int(bulk_dict["k_b"]),
        n_b=int(bulk_dict["n_b"]),
        true_purity=float(bulk_gt["purity"]),
        true_copy_number=int(bulk_gt["copy_number"]),
        true_multiplicity=int(bulk_gt["multiplicity"]),
        true_ccf=float(bulk_gt["ccf"]),
        true_expected_vaf=float(bulk_gt["expected_vaf"]),
    )

    single_cells: list[SingleCellData] = []
    sc_data_list = data["single_cells"]
    sc_gt_list = ground_truth["single_cells"]
    sc_dict: dict[str, int | list[list[int]]]
    sc_gt: dict[str, float | bool]
    for sc_dict, sc_gt in zip(sc_data_list, sc_gt_list):  # type: ignore
        sc = SingleCellData(
            k_sc=int(cast(int, sc_dict["k_sc"])),
            n_sc=int(cast(int, sc_dict["n_sc"])),
            germline_snps=[
                (int(alt), int(total))
                for alt, total in
                cast(list[list[int]], sc_dict["germline_snps"])
            ],
            true_variant_present=bool(sc_gt["variant_present"]),
            true_sc_alpha=float(sc_gt["alpha"]),
            true_sc_beta=float(sc_gt["beta"]),
            true_concentration=float(
                sc_gt.get(
                    "effective_concentration", sc_gt["alpha"] + sc_gt["beta"]
                )
            ),
        )
        single_cells.append(sc)

    return bulk_data, single_cells, hyperparameters
