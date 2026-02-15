"""
Tree-Based Simulation Module
=============================

This module implements a coalescent tree-based simulator that generates
correlated cell genotypes. Each mutation site produces one JSON file.

License
-------
This file is part of ``sc-big``.

Copyright (C) 2026 Daniel Schuette, daniel.schuette@iccb-cologne.org

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
import os

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, leaves_list
from src.logging_config import get_logger
from src.simulation import (
    BulkData,
    SingleCellData,
    Hyperparameters,
    export_simulation,
)
from src.inference_prosolo import lodato_af, sample_lodato

logger = get_logger(__name__)


def mutation_filename(
    mutation_id: int, C: int, m: int,
    purity: float, ccf: float, vaf: float,
) -> str:
    """Build a JSON filename encoding ground truth parameters.

    Example: ``mutation_0003_C3_m2_pi0.80_ccf0.090_vaf0.051.json``
    """
    return (
        f"mutation_{mutation_id:04d}"
        f"_C{C}_m{m}"
        f"_pi{purity:.2f}"
        f"_ccf{ccf:.3f}"
        f"_vaf{vaf:.3f}"
        f".json"
    )


@dataclass
class TreeNode:
    """Node in a coalescent tree."""
    id: int
    parent: int | None
    children: list[int]
    branch_length: float
    is_leaf: bool


@dataclass
class Mutation:
    """A somatic mutation placed on a tree branch."""
    mutation_id: int
    branch_node_id: int
    copy_number: int
    multiplicity: int
    carrier_cell_ids: list[int]
    realized_ccf: float


@dataclass
class TreeHyperparameters:
    """Hyperparameters for coalescent tree-based simulation."""
    n_cells: int = 100
    n_mutations: int = 10
    tau_range_min: float = 20.0
    tau_range_max: float = 80.0
    tau_cv: float = 0.3
    effective_population_size: float = 1.0
    copy_number_amplification_prob: float = 0.1
    copy_number_deletion_prob: float = 0.1
    true_purity: float = 0.8
    kappa: float = 100.0
    epsilon_sc: float = 0.10
    n_germline_snps: int = 500
    n_b: int = 100
    sc_coverage: int = 3
    seed: int = 42
    error_model: str = "betabinomial"
    mutation_ccf_distribution: str = "natural"


def generate_coalescent_tree(
    n_cells: int, Ne: float, rng: np.random.Generator
) -> tuple[list[TreeNode], float]:
    """
    Generate a Kingman coalescent tree.

    Start with n_cells lineages, repeatedly merge two random ones.
    Waiting time t ~ Exp(k*(k-1)/2 / Ne) where k = active lineages.

    Parameters
    ----------
    n_cells : int
        Number of leaf nodes (cells)
    Ne : float
        Effective population size
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    tuple[list[TreeNode], float]
        Flat list of nodes (leaves 0..n-1, internals n..2n-2) and total
        branch length
    """
    nodes: list[TreeNode] = []
    for i in range(n_cells):
        nodes.append(TreeNode(
            id=i, parent=None, children=[], branch_length=0.0, is_leaf=True
        ))

    active_lineages = list(range(n_cells))
    current_time = 0.0
    next_id = n_cells

    while len(active_lineages) > 1:
        k = len(active_lineages)
        rate = k * (k - 1) / 2.0 / Ne
        wait_time = rng.exponential(1.0 / rate)
        current_time += wait_time
        for lin_id in active_lineages:
            nodes[lin_id].branch_length += wait_time

        idx = rng.choice(len(active_lineages), size=2, replace=False)
        child1 = active_lineages[idx[0]]
        child2 = active_lineages[idx[1]]

        parent_node = TreeNode(
            id=next_id, parent=None, children=[child1, child2],
            branch_length=0.0, is_leaf=False
        )
        nodes.append(parent_node)
        nodes[child1].parent = next_id
        nodes[child2].parent = next_id

        active_lineages = [
            lin for lin in active_lineages
            if lin != child1 and lin != child2
        ]
        active_lineages.append(next_id)
        next_id += 1

    # Stem branch between root and MRCA.
    root_id = active_lineages[0]
    nodes[root_id].branch_length = rng.exponential(Ne)

    total_branch_length = sum(n.branch_length for n in nodes)
    return nodes, total_branch_length


def _get_descendant_leaves(node_id: int, nodes: list[TreeNode]) -> list[int]:
    """
    Recursively get all leaf IDs descending from a node.

    Parameters
    ----------
    node_id : int
        ID of the node to start from
    nodes : list[TreeNode]
        All tree nodes

    Returns
    -------
    list[int]
        Sorted list of leaf node IDs
    """
    node = nodes[node_id]
    if node.is_leaf:
        return [node_id]
    leaves: list[int] = []
    for child_id in node.children:
        leaves.extend(_get_descendant_leaves(child_id, nodes))
    return sorted(leaves)


def place_mutations_on_tree(
    nodes: list[TreeNode], total_branch_length: float,
    n_mutations: int, copy_number_deletion_prob: float,
    copy_number_amplification_prob: float, rng: np.random.Generator,
    mutation_ccf_distribution: str = "natural",
) -> list[Mutation]:
    """
    Place mutations on tree branches, weighted by branch length.

    Parameters
    ----------
    nodes : list[TreeNode]
        All tree nodes
    total_branch_length : float
        Sum of all branch lengths
    n_mutations : int
        Number of mutations to place
    copy_number_deletion_prob : float
        Probability of CN=1 (deletion)
    copy_number_amplification_prob : float
        Probability of CN>2 (amplification)
    rng : np.random.Generator
        Random number generator
    mutation_ccf_distribution : str
        "natural" weights branches by length (coalescent prior);
        "uniform" reweights so that each CCF decile gets equal
        total probability

    Returns
    -------
    list[Mutation]
        List of placed mutations
    """
    n_leaves = sum(1 for n in nodes if n.is_leaf)
    eligible_nodes: list[int] = []
    weights: list[float] = []
    node_ccfs: list[float] = []
    node_carriers: list[list[int]] = []
    for node in nodes:
        if node.branch_length > 0:
            eligible_nodes.append(node.id)
            weights.append(node.branch_length)
            leaves = _get_descendant_leaves(node.id, nodes)
            node_ccfs.append(len(leaves) / n_leaves)
            node_carriers.append(leaves)

    weights_arr = np.array(weights)

    if mutation_ccf_distribution == "uniform":
        ccf_arr = np.array(node_ccfs)
        n_bins = 10
        bin_edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
        bin_idx = np.digitize(ccf_arr, bin_edges) - 1
        bin_totals = np.zeros(n_bins)
        for i, bi in enumerate(bin_idx):
            bin_totals[bi] += weights_arr[i]
        n_nonempty = np.sum(bin_totals > 0)
        for i, bi in enumerate(bin_idx):
            if bin_totals[bi] > 0:
                weights_arr[i] = (
                    weights_arr[i] / bin_totals[bi] / n_nonempty
                )
            else:
                weights_arr[i] = 0.0

    weights_arr /= weights_arr.sum()

    MIN_CCF = 0.05
    MAX_PLACEMENT_ATTEMPTS = 100

    mutations: list[Mutation] = []
    for mut_idx in range(n_mutations):
        for attempt in range(MAX_PLACEMENT_ATTEMPTS):
            branch_idx = rng.choice(len(eligible_nodes), p=weights_arr)
            branch_node_id = eligible_nodes[branch_idx]

            # CN assignment: C ∈ {1, 2, 3}.
            u = rng.random()
            if u < copy_number_deletion_prob:
                C = 1
            elif u < (copy_number_deletion_prob
                      + copy_number_amplification_prob):
                C = 3
            else:
                C = 2

            # Geometric multiplicity: P(m|C) ∝ 2^{-m}.
            m_weights = np.array([2.0 ** (-j) for j in range(1, C + 1)])
            m_weights /= m_weights.sum()
            m = int(rng.choice(np.arange(1, C + 1), p=m_weights))

            carrier_cell_ids = node_carriers[branch_idx]
            realized_ccf = node_ccfs[branch_idx]

            if realized_ccf >= MIN_CCF:
                break
        else:
            logger.warning(
                f"Mutation {mut_idx}: could not find branch with "
                f"CCF >= {MIN_CCF} after {MAX_PLACEMENT_ATTEMPTS} attempts; "
                f"using last draw (CCF={realized_ccf:.4f})"
            )

        mutations.append(Mutation(
            mutation_id=mut_idx,
            branch_node_id=branch_node_id,
            copy_number=C,
            multiplicity=m,
            carrier_cell_ids=carrier_cell_ids,
            realized_ccf=realized_ccf,
        ))

    return mutations


def simulate_sequencing_for_mutation(
    mutation: Mutation, n_cells: int, hyperparams: TreeHyperparameters,
    rng: np.random.Generator,
) -> tuple[BulkData, list[SingleCellData]]:
    """
    Simulate sequencing data for a single mutation site.

    Each mutation site gets its own tau_mean draw (amplification bias varies
    by genomic position). Each cell gets a fresh tau draw from
    Gamma(tau_mean, cv). Germline SNPs are drawn independently per site.

    Parameters
    ----------
    mutation : Mutation
        The mutation to simulate sequencing for
    n_cells : int
        Total number of cells
    hyperparams : TreeHyperparameters
        Simulation hyperparameters
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    tuple[BulkData, list[SingleCellData]]
        Bulk data and list of single cell data
    """
    carrier_set = set(mutation.carrier_cell_ids)
    C = mutation.copy_number
    m = mutation.multiplicity
    tau_mean = rng.uniform(
        hyperparams.tau_range_min, hyperparams.tau_range_max
    )
    true_expected_vaf = (
        hyperparams.true_purity * mutation.realized_ccf * m
        / (hyperparams.true_purity * C + 2 * (1 - hyperparams.true_purity))
    )
    true_expected_vaf = max(1e-6, min(1 - 1e-6, true_expected_vaf))
    alpha_bulk = hyperparams.kappa * true_expected_vaf
    beta_bulk = hyperparams.kappa * (1 - true_expected_vaf)
    p_bulk = float(rng.beta(alpha_bulk, beta_bulk))
    k_b = int(rng.binomial(hyperparams.n_b, p_bulk))

    bulk_data = BulkData(
        k_b=k_b,
        n_b=int(hyperparams.n_b),
        true_purity=float(hyperparams.true_purity),
        true_copy_number=int(C),
        true_multiplicity=int(m),
        true_ccf=float(mutation.realized_ccf),
        true_expected_vaf=float(true_expected_vaf),
    )

    use_lodato = hyperparams.error_model == "lodato"

    single_cells: list[SingleCellData] = []
    for cell_id in range(n_cells):
        variant_present = cell_id in carrier_set
        tau_cell = _draw_tau(tau_mean, hyperparams.tau_cv, rng)
        germline_snps = _simulate_germline_snps(
            tau_cell, hyperparams.n_germline_snps,
            hyperparams.sc_coverage, rng,
            use_lodato=use_lodato,
        )

        n_sc = int(rng.poisson(hyperparams.sc_coverage))

        if use_lodato:
            # Lodato amplification model: sample from coverage-dependent
            # beta-binomial with fixed MDA parameters.
            if variant_present:
                af = lodato_af(m, C)
                k_lodato = sample_lodato(af, n_sc, rng)
            else:
                k_lodato = sample_lodato(0.0, n_sc, rng)
            # ProSolo error model: Per-read sequencing error where true-alt
            # reads stay alt with prob 1-e, true-ref reads become alt with prob
            # e/3 (one of 3 wrong bases).
            e = hyperparams.epsilon_sc
            if n_sc > 0 and e > 0:
                alt_kept = rng.binomial(k_lodato, 1.0 - e)
                ref_to_alt = rng.binomial(n_sc - k_lodato, e / 3.0)
                k_sc = int(alt_kept + ref_to_alt)
            else:
                k_sc = k_lodato
            true_sc_alpha = 0.0  # does not apply
            true_sc_beta = 0.0   # does not apply
        else:
            # SC-BIG's (fallback) beta-binomial amplification model.
            if variant_present:
                expected_sc_vaf = m / C
                true_sc_alpha = tau_cell * expected_sc_vaf
                true_sc_beta = tau_cell * (1.0 - expected_sc_vaf)
                if n_sc > 0:
                    p_sc = float(rng.beta(
                        max(true_sc_alpha, 1e-6), max(true_sc_beta, 1e-6)
                    ))
                    k_sc = int(rng.binomial(n_sc, p_sc))
                else:
                    k_sc = 0
            else:
                true_sc_alpha = tau_cell * 0.5
                true_sc_beta = tau_cell * 0.5
                if n_sc > 0:
                    k_sc = int(rng.binomial(n_sc, hyperparams.epsilon_sc))
                else:
                    k_sc = 0

        single_cells.append(SingleCellData(
            k_sc=int(k_sc),
            n_sc=int(n_sc),
            germline_snps=germline_snps,
            true_variant_present=bool(variant_present),
            true_sc_alpha=float(true_sc_alpha),
            true_sc_beta=float(true_sc_beta),
            true_concentration=float(tau_cell),
        ))

    return bulk_data, single_cells


def _draw_tau(
    mean: float, cv: float, rng: np.random.Generator
) -> float:
    """Draw tau from Gamma distribution, floored at 1.0."""
    shape = 1.0 / (cv ** 2)
    scale = mean * (cv ** 2)
    tau = rng.gamma(shape, scale)
    return max(float(tau), 1.0)


def _simulate_germline_snps(
    concentration: float, n_germline_snps: int, sc_coverage: int,
    rng: np.random.Generator, *, use_lodato: bool = False,
) -> list[tuple[int, int]]:
    """
    Simulate germline heterozygous SNPs.

    When use_lodato is False, uses symmetric Beta-Binomial with the given
    concentration (SC-BIG model). When True, uses the Lodato AF=0.5 model.
    Lodato-based inference does not use hSNPs, we simulate them anyways.
    """
    germline_snps: list[tuple[int, int]] = []

    if use_lodato:
        for _ in range(n_germline_snps):
            n_reads = int(rng.poisson(sc_coverage))
            if n_reads > 0:
                alt_reads = sample_lodato(0.5, n_reads, rng)
                germline_snps.append((alt_reads, n_reads))
    else:
        alpha_germline = beta_germline = concentration / 2.0
        for _ in range(n_germline_snps):
            n_reads = int(rng.poisson(sc_coverage))
            if n_reads > 0:
                p = float(
                    rng.beta(
                        max(alpha_germline, 1e-6), max(beta_germline, 1e-6)
                    )
                )
                alt_reads = int(rng.binomial(n_reads, p))
                germline_snps.append((alt_reads, n_reads))

    return germline_snps


class CoalescentSNVSimulator:
    """
    Simulator that generates single-cell SNV data using a coalescent tree to
    achieve realistic clonal composition.
    """

    def __init__(self, hyperparams: TreeHyperparameters) -> None:
        self.hyperparams = hyperparams
        self.rng = np.random.default_rng(hyperparams.seed)

    def simulate(self) -> tuple[
        list[TreeNode], list[Mutation],
        list[tuple[BulkData, list[SingleCellData]]],
    ]:
        """
        Run full simulation pipeline.

        Returns
        -------
        tuple
            (nodes, mutations, datasets) where datasets[i] is
            (BulkData, list[SingleCellData]) for mutation i
        """
        logger.info(
            f"Generating coalescent tree with {self.hyperparams.n_cells} cells"
        )
        nodes, total_branch_length = generate_coalescent_tree(
            self.hyperparams.n_cells,
            self.hyperparams.effective_population_size, self.rng,
        )
        logger.info(
            f"Tree has {len(nodes)} nodes, "
            f"total branch length={total_branch_length:.4f}"
        )

        logger.info(
            f"Placing {self.hyperparams.n_mutations} mutations on tree"
        )
        mutations = place_mutations_on_tree(
            nodes, total_branch_length, self.hyperparams.n_mutations,
            self.hyperparams.copy_number_deletion_prob,
            self.hyperparams.copy_number_amplification_prob,
            self.rng, self.hyperparams.mutation_ccf_distribution,
        )

        for mut in mutations:
            logger.info(
                f"  Mutation {mut.mutation_id}: branch={mut.branch_node_id}, "
                f"C={mut.copy_number}, m={mut.multiplicity}, "
                f"CCF={mut.realized_ccf:.3f} "
                f"({len(mut.carrier_cell_ids)} carriers)"
            )

        logger.info("Simulating sequencing data for each mutation")
        datasets: list[tuple[BulkData, list[SingleCellData]]] = []
        for mut in mutations:
            bulk_data, single_cells = simulate_sequencing_for_mutation(
                mut, self.hyperparams.n_cells, self.hyperparams, self.rng,
            )
            datasets.append((bulk_data, single_cells))
            logger.debug(
                f"  Mutation {mut.mutation_id}: "
                f"bulk k_b={bulk_data.k_b}/{bulk_data.n_b}, "
                f"true_vaf={bulk_data.true_expected_vaf:.3f}"
            )

        return nodes, mutations, datasets


def export_tree_simulation(
    nodes: list[TreeNode], mutations: list[Mutation],
    datasets: list[tuple[BulkData, list[SingleCellData]]],
    hyperparams: TreeHyperparameters, output_dir: str,
) -> None:
    """
    Export tree simulation results.

    Parameters
    ----------
    nodes : list[TreeNode]
        Tree nodes
    mutations : list[Mutation]
        Placed mutations
    datasets : list[tuple[BulkData, list[SingleCellData]]]
        Sequencing data per mutation
    hyperparams : TreeHyperparameters
        Simulation hyperparameters
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    for mut, (bulk_data, single_cells) in zip(mutations, datasets):
        hp = Hyperparameters(
            true_purity=hyperparams.true_purity,
            true_copy_number=mut.copy_number,
            true_multiplicity=mut.multiplicity,
            true_ccf=mut.realized_ccf,
            kappa=hyperparams.kappa,
            epsilon_sc=hyperparams.epsilon_sc,
            n_germline_snps=hyperparams.n_germline_snps,
            n_b=hyperparams.n_b,
            sc_coverage=hyperparams.sc_coverage,
            base_concentration_mean=float(
                (hyperparams.tau_range_min + hyperparams.tau_range_max) / 2
            ),
            base_concentration_cv=hyperparams.tau_cv,
        )
        fname = mutation_filename(
            mut.mutation_id, mut.copy_number, mut.multiplicity,
            hyperparams.true_purity, mut.realized_ccf,
            bulk_data.true_expected_vaf,
        )
        output_file = os.path.join(output_dir, fname)
        export_simulation(bulk_data, single_cells, hp, output_file=output_file)
        logger.info(f"Exported mutation {mut.mutation_id} to {output_file}")

    metadata: dict = {
        "tree": {
            "n_cells": hyperparams.n_cells,
            "n_nodes": len(nodes),
            "effective_population_size": hyperparams.effective_population_size,
            "nodes": [
                {
                    "id": n.id,
                    "parent": n.parent,
                    "children": n.children,
                    "branch_length": n.branch_length,
                    "is_leaf": n.is_leaf,
                }
                for n in nodes
            ],
        },
        "mutations": [
            {
                "mutation_id": mut.mutation_id,
                "branch_node_id": mut.branch_node_id,
                "copy_number": mut.copy_number,
                "multiplicity": mut.multiplicity,
                "carrier_cell_ids": mut.carrier_cell_ids,
                "realized_ccf": mut.realized_ccf,
                "filename": mutation_filename(
                    mut.mutation_id, mut.copy_number, mut.multiplicity,
                    hyperparams.true_purity, mut.realized_ccf,
                    datasets[i][0].true_expected_vaf,
                ),
            }
            for i, mut in enumerate(mutations)
        ],
        "hyperparameters": {
            "n_cells": hyperparams.n_cells,
            "n_mutations": hyperparams.n_mutations,
            "tau_range_min": hyperparams.tau_range_min,
            "tau_range_max": hyperparams.tau_range_max,
            "tau_cv": hyperparams.tau_cv,
            "effective_population_size": hyperparams.effective_population_size,
            "copy_number_amplification_prob":
                hyperparams.copy_number_amplification_prob,
            "copy_number_deletion_prob":
                hyperparams.copy_number_deletion_prob,
            "true_purity": hyperparams.true_purity,
            "kappa": hyperparams.kappa,
            "epsilon_sc": hyperparams.epsilon_sc,
            "n_germline_snps": hyperparams.n_germline_snps,
            "n_b": hyperparams.n_b,
            "sc_coverage": hyperparams.sc_coverage,
            "seed": hyperparams.seed,
            "error_model": hyperparams.error_model,
            "mutation_ccf_distribution":
                hyperparams.mutation_ccf_distribution,
        },
        "ccf_summary": {
            "mean": float(np.mean([m.realized_ccf for m in mutations])),
            "min": float(np.min([m.realized_ccf for m in mutations])),
            "max": float(np.max([m.realized_ccf for m in mutations])),
            "values": [m.realized_ccf for m in mutations],
        },
    }

    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Exported metadata to {metadata_file}")


def plot_tree_simulation(
    nodes: list[TreeNode], mutations: list[Mutation],
    datasets: list[tuple[BulkData, list[SingleCellData]]],
    hyperparams: TreeHyperparameters, output_file: str,
) -> None:
    """
    Plot tree simulation overview.

    Parameters
    ----------
    nodes : list[TreeNode]
        Tree nodes
    mutations : list[Mutation]
        Placed mutations
    datasets : list[tuple[BulkData, list[SingleCellData]]]
        Sequencing data per mutation
    hyperparams : TreeHyperparameters
        Simulation hyperparameters
    output_file : str
        Path to save plot
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Coalescent Tree Simulation: {hyperparams.n_cells} cells, "
        f"{hyperparams.n_mutations} mutations",
        fontsize=14, fontweight="bold"
    )
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, top=0.93)

    # Panel a: Tree dendrogram.
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(
        -0.15, 1.05, "a", transform=ax1.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    _plot_tree_dendrogram(ax1, nodes, mutations)

    # Panel b: CCF histogram.
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(
        -0.15, 1.05, "b", transform=ax2.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    ccfs = [m.realized_ccf for m in mutations]
    ax2.hist(ccfs, bins=max(5, len(mutations) // 2), alpha=0.7,
             color="steelblue", edgecolor="black")
    ax2.axvline(np.mean(ccfs), color="red", linestyle="--",
                label=f"Mean={np.mean(ccfs):.3f}")
    ax2.set_xlabel("Realized CCF")
    ax2.set_ylabel("Count")
    ax2.set_title("CCF Distribution Across Mutations")
    ax2.legend()

    # Panel c: Counts per (C, m) combination.
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.text(
        -0.15, 1.05, "c", transform=ax3.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    cm_counts = Counter((m.copy_number, m.multiplicity) for m in mutations)
    cm_sorted = sorted(cm_counts.keys())
    labels = [f"C={c}, m={m}" for c, m in cm_sorted]
    counts = [cm_counts[k] for k in cm_sorted]
    ax3.bar(np.arange(len(labels)), counts, color="steelblue", alpha=0.7,
            edgecolor="black")
    ax3.set_xticks(np.arange(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha="right")
    ax3.set_ylabel("Number of mutations")
    ax3.set_title("Copy Number / Multiplicity Distribution")

    # Panel d: Clustered cells x mutations heatmap.
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.text(
        -0.15, 1.05, "d", transform=ax4.transAxes, fontsize=16,
        fontweight="bold", va="top"
    )
    n_cells = hyperparams.n_cells
    n_muts = len(mutations)
    heatmap = np.zeros((n_cells, n_muts))
    for j, mut in enumerate(mutations):
        for cell_id in mut.carrier_cell_ids:
            heatmap[cell_id, j] = 1.0
    # @NOTE(ds): Cluster cells by mutation profile, mutations by cell profile.
    if n_cells > 1:
        cell_order = leaves_list(linkage(heatmap, method="ward"))
    else:
        cell_order = np.array([0])
    if n_muts > 1:
        mut_order = leaves_list(linkage(heatmap.T, method="ward"))
    else:
        mut_order = np.array([0])

    heatmap = heatmap[np.ix_(cell_order, mut_order)]
    ax4.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax4.set_xticks([])
    ax4.set_xlabel("Mutations")
    ax4.set_ylabel("Cells")
    ax4.set_title("Mutation status (yellow=absent, red=present)")

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Tree simulation plots saved to {output_file}")


def _plot_tree_dendrogram(
    ax: plt.Axes,
    nodes: list[TreeNode],
    mutations: list[Mutation],
) -> None:
    """Plot a simple tree dendrogram with mutations marked."""
    n_leaves = sum(1 for n in nodes if n.is_leaf)

    x_pos: dict[int, float] = {}
    y_pos: dict[int, float] = {}
    root_id = None
    for n in nodes:
        if n.parent is None and not n.is_leaf:
            root_id = n.id
            break

    if root_id is None:
        ax.text(0.5, 0.5, "Single cell (no tree)", ha="center", va="center")
        ax.set_title("Coalescent Tree")
        return

    leaf_order: list[int] = []
    _get_leaf_order(root_id, nodes, leaf_order)
    for i, leaf_id in enumerate(leaf_order):
        x_pos[leaf_id] = float(i)

    _compute_depths(root_id, nodes, y_pos, 0.0)

    _compute_internal_x(root_id, nodes, x_pos)
    mut_on_branch: dict[int, list[int]] = {}
    for mut in mutations:
        mut_on_branch.setdefault(mut.branch_node_id, []).append(
            mut.mutation_id
        )

    for n in nodes:
        if n.parent is not None:
            ax.plot(
                [x_pos[n.id], x_pos[n.id]],
                [y_pos[n.id], y_pos[n.parent]],
                color="black", linewidth=0.5
            )
            ax.plot(
                [x_pos[n.id], x_pos[n.parent]],
                [y_pos[n.parent], y_pos[n.parent]],
                color="black", linewidth=0.5
            )
            if n.id in mut_on_branch:
                muts_here = mut_on_branch[n.id]
                n_muts_here = len(muts_here)
                y_lo = y_pos[n.parent]
                y_hi = y_pos[n.id]
                for idx, mut_id in enumerate(muts_here):
                    frac = (idx + 1) / (n_muts_here + 1)
                    my = y_lo + frac * (y_hi - y_lo)
                    ax.plot(x_pos[n.id], my, "r*", markersize=8)
                    ax.annotate(
                        f"m{mut_id}", (x_pos[n.id], my),
                        fontsize=6, color="red",
                        xytext=(3, 0), textcoords="offset points"
                    )

    stem = nodes[root_id].branch_length
    if stem > 0:
        y_stem_top = y_pos[root_id] - stem
        ax.plot(
            [x_pos[root_id], x_pos[root_id]],
            [y_pos[root_id], y_stem_top],
            color="black", linewidth=0.5
        )
        if root_id in mut_on_branch:
            muts_here = mut_on_branch[root_id]
            n_muts_here = len(muts_here)
            for idx, mut_id in enumerate(muts_here):
                frac = (idx + 1) / (n_muts_here + 1)
                my = y_pos[root_id] - frac * stem
                ax.plot(x_pos[root_id], my, "r*", markersize=8)
                ax.annotate(
                    f"m{mut_id}", (x_pos[root_id], my),
                    fontsize=6, color="red",
                    xytext=(3, 0), textcoords="offset points"
                )

    ax.set_title(f"Coalescent Tree ({n_leaves} cells)")
    ax.set_xlabel("Cell")
    ax.set_ylabel("Coalescent Time")
    if n_leaves > 50:
        ax.set_xticks([])
    ax.invert_yaxis()


def _get_leaf_order(
    node_id: int, nodes: list[TreeNode], leaf_order: list[int]
) -> None:
    """Get leaves in left-to-right order."""
    node = nodes[node_id]
    if node.is_leaf:
        leaf_order.append(node_id)
        return
    for child_id in node.children:
        _get_leaf_order(child_id, nodes, leaf_order)


def _compute_depths(
    node_id: int, nodes: list[TreeNode], y_pos: dict[int, float], depth: float
) -> None:
    """Compute depth of each node (distance from root)."""
    y_pos[node_id] = depth
    node = nodes[node_id]
    for child_id in node.children:
        _compute_depths(
            child_id, nodes, y_pos, depth + nodes[child_id].branch_length
        )


def _compute_internal_x(
    node_id: int, nodes: list[TreeNode], x_pos: dict[int, float]
) -> float:
    """Compute x position of internal nodes as mean of children."""
    node = nodes[node_id]
    if node.is_leaf:
        return x_pos[node_id]
    child_xs = [
        _compute_internal_x(cid, nodes, x_pos) for cid in node.children
    ]
    x_pos[node_id] = np.mean(child_xs)
    return x_pos[node_id]
