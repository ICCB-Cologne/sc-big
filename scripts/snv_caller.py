"""
Command-line interface for SC-BIG, a Bayesian scDNA SNV caller.

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
import argparse
import json

import numpy as np

from multiprocessing import Pool, cpu_count
from src.logging_config import setup_logging, get_logger
from src.simulation import (
    BayesianSNVSimulator,
    Hyperparameters,
    export_simulation,
    print_simulation_summary,
    plot_simulation_results,
    import_simulation_from_json,
)
from src.inference import (
    InferenceInputs,
    compute_variant_probability,
    compute_bulk_posterior,
    compute_population_tau,
    estimate_ccf_prior_from_sc,
    plot_inference_results,
    plot_inference_diagnostics,
    plot_mcmc_traces,
)
from src.inference_prosolo import (
    prosolo_call_native_batch,
)
from src.tree_simulation import (
    CoalescentSNVSimulator,
    TreeHyperparameters,
    export_tree_simulation,
    plot_tree_simulation,
)
from src.comparison import plot_comparison

logger = get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def simulate_command(args: argparse.Namespace) -> None:
    """Execute simulation subcommand."""
    hyperparams: Hyperparameters = Hyperparameters(
        true_purity=args.purity,
        true_copy_number=args.copy_number,
        true_multiplicity=args.multiplicity,
        true_ccf=args.ccf,
        kappa=args.kappa,
        epsilon_sc=args.epsilon_sc,
        n_germline_snps=args.n_germline_snps,
        n_b=args.bulk_coverage,
        sc_coverage=args.sc_coverage,
        base_concentration_mean=args.base_concentration_mean,
        base_concentration_cv=args.base_concentration_cv,
    )

    logger.info(
        f"Simulating {args.n_cells} single cells with seed {args.seed}"
    )
    logger.debug(f"Hyperparameters: {hyperparams}")

    simulator = BayesianSNVSimulator(hyperparams=hyperparams, seed=args.seed)
    bulk_data, single_cells = simulator.simulate_dataset(
        n_single_cells=args.n_cells
    )

    if args.verbose:
        print_simulation_summary(bulk_data, single_cells)

    if args.output:
        logger.info(f"Exporting simulation data, saving as {args.output}")
        export_simulation(
            bulk_data, single_cells, hyperparams, output_file=args.output
        )

    if args.plot:
        logger.info(f"Generating plots, saving as {args.plot}")
        plot_simulation_results(
            bulk_data, single_cells, args.plot, hyperparameters=hyperparams
        )

    logger.info("Simulation completed successfully")


def simulate_tree_command(args: argparse.Namespace) -> None:
    """Execute coalescent tree-based simulation subcommand."""
    hyperparams = TreeHyperparameters(
        n_cells=args.n_cells,
        n_mutations=args.n_mutations,
        tau_range_min=args.tau_range_low,
        tau_range_max=args.tau_range_high,
        tau_cv=args.tau_cv,
        effective_population_size=args.effective_population_size,
        copy_number_amplification_prob=args.copy_number_amplification_prob,
        copy_number_deletion_prob=args.copy_number_deletion_prob,
        true_purity=args.purity,
        kappa=args.kappa,
        epsilon_sc=args.epsilon_sc,
        n_germline_snps=args.n_germline_snps,
        n_b=args.bulk_coverage,
        sc_coverage=args.sc_coverage,
        seed=args.seed,
        error_model=args.error_model,
        mutation_ccf_distribution=args.ccf_distribution,
    )

    logger.info(
        f"Running coalescent tree simulation: "
        f"{args.n_cells} cells, {args.n_mutations} mutations, seed={args.seed}"
    )

    simulator = CoalescentSNVSimulator(hyperparams=hyperparams)
    nodes, mutations, datasets = simulator.simulate()

    if args.verbose:
        logger.info("=== Tree Simulation Summary ===")
        logger.info(f"  Cells: {hyperparams.n_cells}")
        logger.info(f"  Mutations: {len(mutations)}")
        ccfs = [m.realized_ccf for m in mutations]
        logger.info(
            f"  CCF range: [{min(ccfs):.3f}, {max(ccfs):.3f}], "
            f"mean={np.mean(ccfs):.3f}"
        )
        for mut in mutations:
            logger.info(
                f"  Mutation {mut.mutation_id}: C={mut.copy_number}, "
                f"m={mut.multiplicity}, CCF={mut.realized_ccf:.3f} "
                f"({len(mut.carrier_cell_ids)} carriers)"
            )

    if args.output_dir:
        logger.info(f"Exporting tree simulation to {args.output_dir}")
        export_tree_simulation(
            nodes, mutations, datasets, hyperparams, args.output_dir
        )

    if args.plot:
        logger.info(f"Generating tree simulation plots to {args.plot}")
        plot_tree_simulation(
            nodes, mutations, datasets, hyperparams, args.plot
        )

    logger.info("Tree simulation completed successfully")


def process_single_cell(task_args):
    """Process a single cell (for parallel execution)."""
    i, sc, bulk_data, args, pop_tau, bulk_posterior = task_args
    error_model = getattr(args, "error_model", "betabinomial")
    inputs = InferenceInputs(
        k_b=bulk_data.k_b,
        n_b=bulk_data.n_b,
        k_sc=sc.k_sc,
        n_sc=sc.n_sc,
        purity_mean=args.purity_mean,
        purity_std=args.purity_std,
        copy_number_mean=args.copy_number_mean,
        copy_number_std=args.copy_number_std,
        kappa=args.kappa,
        epsilon_sc=args.epsilon_sc,
        population_tau=pop_tau,
        multiplicity_prior_type=args.multiplicity_prior,
        error_model=error_model,
    )

    prob, details = compute_variant_probability(
        inputs, n_samples=args.n_samples, bulk_posterior=bulk_posterior
    )

    return {
        "cell_id": i,
        "posterior_prob": prob,
        "true_variant_present": sc.true_variant_present,
        "k_sc": sc.k_sc,
        "n_sc": sc.n_sc,
        "concentration": details["concentration"],
        "ccf_samples": details["ccf_samples"],
        "ccf_weights": details["ccf_weights"],
    }


def infer_command(args: argparse.Namespace) -> None:
    """Execute inference subcommand."""
    logger.info(f"Loading simulation data from {args.data}")
    bulk_data, single_cells, hyperparameters = import_simulation_from_json(
        args.data
    )

    n_workers = args.n_workers if args.n_workers > 0 else cpu_count()
    logger.info(
        f"Running inference on {len(single_cells)} single cells "
        f"(n_samples={args.n_samples}, workers={n_workers})"
    )

    error_model = getattr(args, "error_model", "betabinomial")
    if error_model == "lodato":
        logger.info("Lodato error model: skipping tau estimation (unused)")
        pop_tau = 0.0
    else:
        logger.info("Computing population tau from pooled germline SNPs")
        all_germline = [sc.germline_snps for sc in single_cells]
        pop_tau, n_snps = compute_population_tau(all_germline)

    logger.info("Estimating CCF prior from single-cell aggregate data")
    ccf_alpha, ccf_beta = estimate_ccf_prior_from_sc(
        single_cells=[
            {"k_sc": sc.k_sc, "n_sc": sc.n_sc} for sc in single_cells
        ]
    )

    logger.info(
        "Computing bulk posterior once (will be reused for all cells)"
    )
    bulk_posterior_result = compute_bulk_posterior(
        k_b=bulk_data.k_b,
        n_b=bulk_data.n_b,
        purity_mean=args.purity_mean,
        purity_std=args.purity_std,
        copy_number_mean=args.copy_number_mean,
        copy_number_std=args.copy_number_std,
        kappa=args.kappa,
        n_samples=args.n_samples,
        n_workers=n_workers,
        multiplicity_prior_type=args.multiplicity_prior,
        ccf_prior_alpha=ccf_alpha,
        ccf_prior_beta=ccf_beta
    )
    bulk_posterior_samples = bulk_posterior_result["samples"]
    mcmc_traces = bulk_posterior_result["mcmc_traces"]

    tasks = [
        (i, sc, bulk_data, args, pop_tau, bulk_posterior_samples)
        for i, sc in enumerate(single_cells)
    ]
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_single_cell, tasks)
    else:
        results = [process_single_cell(task) for task in tasks]

    ccf_posterior = None
    if results[0]["ccf_samples"] is not None:
        ccf_posterior = {
            "ccf_samples": results[0]["ccf_samples"],
            "ccf_weights": results[0]["ccf_weights"],
        }

    for r in results:
        logger.info(
            f"Cell {r['cell_id']}: P(z=1|D)={r['posterior_prob']:.3f}, "
            f"true_z={int(r['true_variant_present'])}, "
            f"k_sc={r['k_sc']}/{r['n_sc']}"
        )

    pprob_threshold = args.posterior_prob_threshold
    correct = sum(
        1 for r in results
        if (r["posterior_prob"] > pprob_threshold) == r["true_variant_present"]
    )
    accuracy = correct / len(results)
    logger.info(
        f"Inference completed: accuracy={accuracy:.2%} "
        f"({correct}/{len(results)}) for threshold {pprob_threshold:.2%}"
    )

    if hasattr(args, "output") and args.output:
        logger.info(f"Saving inference results to {args.output}")
        output_data = {
            "results": results,
            "ccf_posterior": ccf_posterior,
            "accuracy": accuracy,
            "threshold": pprob_threshold,
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
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, cls=NumpyEncoder)

    if args.plot:
        logger.info(f"Generating inference plots, saving to {args.plot}")
        plot_inference_results(
            results, output_file=args.plot, decision_threshold=pprob_threshold
        )

        if ccf_posterior and args.diagnostics:
            diag_file = args.plot.replace(".png", "_diagnostics.png")
            logger.info(f"Generating diagnostic plots to {diag_file}")
            true_taus = [sc.true_concentration for sc in single_cells]
            true_tau_mean = np.mean(true_taus)
            plot_inference_diagnostics(
                results, ccf_posterior,
                bulk_posterior_samples=bulk_posterior_samples,
                population_tau=pop_tau,
                true_tau_mean=true_tau_mean,
                output_file=diag_file
            )

            trace_file = args.plot.replace(".png", "_traces.png")
            logger.info(f"Generating MCMC trace plots to {trace_file}")
            plot_mcmc_traces(mcmc_traces, output_file=trace_file)


def infer_prosolo_command(args: argparse.Namespace) -> None:
    """Execute ProSolo inference subcommand."""
    logger.info(f"Loading simulation data from {args.data}")
    bulk_data, single_cells, hyperparameters = import_simulation_from_json(
        args.data
    )

    n_workers = args.n_workers if args.n_workers > 0 else cpu_count()
    logger.info(
        f"Running ProSolo inference on {len(single_cells)} cells "
        f"(base_error_rate={args.base_error_rate}, workers={n_workers})"
    )

    cells = [(sc.k_sc, sc.n_sc) for sc in single_cells]
    native_results = prosolo_call_native_batch(
        cells=cells,
        k_b=bulk_data.k_b,
        n_b=bulk_data.n_b,
        base_error_rate=args.base_error_rate,
        n_workers=n_workers,
    )

    results = []
    for i, (sc, nr) in enumerate(zip(single_cells, native_results)):
        results.append({
            "cell_id": i,
            "posterior_prob": nr["prob_alt"],
            "posteriors": nr["posteriors"],
            "true_variant_present": sc.true_variant_present,
            "k_sc": sc.k_sc,
            "n_sc": sc.n_sc,
        })
        logger.debug(
            f"Cell {i}: P(alt)={nr['prob_alt']:.3f}, "
            f"true_z={int(sc.true_variant_present)}, "
            f"k_sc={sc.k_sc}/{sc.n_sc}"
        )

    pprob_threshold = args.posterior_prob_threshold
    correct = sum(
        1 for r in results
        if (r["posterior_prob"] > pprob_threshold) == r["true_variant_present"]
    )
    accuracy = correct / len(results)
    logger.info(
        f"ProSolo completed: accuracy={accuracy:.2%} "
        f"({correct}/{len(results)}) for threshold {pprob_threshold:.2%}"
    )

    if hasattr(args, "output") and args.output:
        logger.info(f"Saving ProSolo results to {args.output}")
        output_data = {
            "results": results,
            "accuracy": accuracy,
            "threshold": pprob_threshold,
            "method": "prosolo_native",
            "base_error_rate": args.base_error_rate,
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
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, cls=NumpyEncoder)


def compare_command(args: argparse.Namespace) -> None:
    """Execute comparison subcommand."""
    logger.info(
        f"Loading {len(args.scbig)} SC-BIG and "
        f"{len(args.prosolo)} ProSolo result files"
    )

    scbig_results = []
    for path in args.scbig:
        with open(path) as f:
            data = json.load(f)
        ccf = data["hyperparameters"]["true_ccf"]
        for r in data["results"]:
            r["ccf"] = ccf
        scbig_results.extend(data["results"])

    prosolo_results = []
    for path in args.prosolo:
        with open(path) as f:
            data = json.load(f)
        ccf = data["hyperparameters"]["true_ccf"]
        for r in data["results"]:
            r["ccf"] = ccf
        prosolo_results.extend(data["results"])

    plot_comparison(scbig_results, prosolo_results, args.output)
    logger.info(f"Comparison plot saved to {args.output}")


def main() -> None:
    """Main entry point for SC-BIG CLI."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="SC-BIG, a Bayesian scDNA SNV caller"
    )

    subparsers: argparse._SubParsersAction[argparse.ArgumentParser] = \
        parser.add_subparsers(
            dest="subcommand", help="Available commands", required=True,
        )

    simulate_parser = subparsers.add_parser(
        "simulate", help="Simulate bulk and single-cell sequencing data"
    )

    simulate_parser.add_argument(
        "-n", "--n-cells", type=int, required=True,
        help="Number of single cells to simulate",
    )
    simulate_parser.add_argument(
        "-o", "--output", type=str,
        help="Name of output JSON file for simulation data "
        "(output is not saved if omitted)",
    )
    simulate_parser.add_argument(
        "-p", "--plot", type=str,
        help="Name of output file for simulation plots "
        "(output is not saved if omitted)",
    )
    simulate_parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print detailed simulation summary",
    )
    simulate_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed to use for simulation",
    )
    simulate_parser.add_argument(
        "--purity", type=float, default=0.8, help="Tumor purity (0-1)",
    )
    simulate_parser.add_argument(
        "--copy-number", type=int, default=2,
        help="Copy number at variant site (1-10)",
    )
    simulate_parser.add_argument(
        "--multiplicity", type=int, default=1,
        help="Variant multiplicity (must be <= copy-number)",
    )
    simulate_parser.add_argument(
        "--ccf", type=float, default=0.5, help="Cancer cell fraction (0-1)",
    )
    simulate_parser.add_argument(
        "--bulk-coverage", type=int, default=100,
        help="Bulk sequencing coverage",
    )
    simulate_parser.add_argument(
        "--sc-coverage", type=int, default=5,
        help="Single-cell sequencing coverage",
    )
    simulate_parser.add_argument(
        "--n-germline-snps", type=int, default=20,
        help="Number of germline SNPs to simulate",
    )
    simulate_parser.add_argument(
        "--kappa", type=float, default=100.0,
        help="Bulk overdispersion concentration parameter",
    )
    simulate_parser.add_argument(
        "--epsilon-sc", type=float, default=0.10,
        help="Single-cell sequencing error rate",
    )
    simulate_parser.add_argument(
        "--base-concentration-mean", type=float, default=30.0,
        help="Mean concentration parameter (tau) across cells. "
        "Higher values lead to less amplification / dropout",
    )
    simulate_parser.add_argument(
        "--base-concentration-cv", type=float, default=0.3,
        help="Coefficient of variation for concentration parameter across "
        "cells. Controls inter-cell variability in amplification bias",
    )

    infer_parser = subparsers.add_parser(
        "infer", help="Infer probability of SNV being present based on "
        "single-cell and bulk data"
    )

    infer_parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="Path to data input file (requires format of `simulate` command)"
    )
    infer_parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Enable verbose logging"
    )
    infer_parser.add_argument(
        "-o", "--output", type=str,
        help="Output file for inference results JSON (optional)"
    )
    infer_parser.add_argument(
        "-p", "--plot", type=str,
        help="Output file for inference plots (output is not saved if omitted)"
    )
    infer_parser.add_argument(
        "--diagnostics", action="store_true", default=False,
        help="Generate additional diagnostic plots (requires -p/--plot)"
    )
    infer_parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel workers (0 = use all CPUs)"
    )
    infer_parser.add_argument(
        "--n-samples", type=int, default=5000,
        help="Number of Monte Carlo samples for posterior estimation"
    )
    infer_parser.add_argument(
        "--purity-mean", type=float, default=0.8,
        help="Mean purity estimate"
    )
    infer_parser.add_argument(
        "--purity-std", type=float, default=0.1,
        help="Purity standard deviation"
    )
    infer_parser.add_argument(
        "--copy-number-mean", type=float, default=2.0,
        help="Mean copy number estimate"
    )
    infer_parser.add_argument(
        "--copy-number-std", type=float, default=0.5,
        help="Copy number standard deviation"
    )
    infer_parser.add_argument(
        "--kappa", type=float, default=100.0,
        help="Bulk overdispersion concentration parameter"
    )
    infer_parser.add_argument(
        "--epsilon-sc", type=float, default=0.10,
        help="Single-cell sequencing error rate"
    )
    infer_parser.add_argument(
        "--posterior-prob-threshold", type=float, default=0.5,
        help="Posterior probability threshold for calling a variant"
    )
    infer_parser.add_argument(
        "--multiplicity-prior", type=str, default="geometric",
        choices=["uniform", "geometric"],
        help="Type of multiplicity prior: 'uniform' (P(m|C)=1/C) or "
        "'geometric' (P(m|C) ∝ 2^(-m), favors m=1)"
    )
    infer_parser.add_argument(
        "--error-model", type=str, default="betabinomial",
        choices=["betabinomial", "lodato"],
        help="Amplification error model for single-cell likelihood: "
        "'betabinomial' (SC-BIG model) or 'lodato' (MDA model)",
    )

    prosolo_parser = subparsers.add_parser(
        "infer-prosolo",
        help="Run ProSolo binary (Laehnemann et al. 2021) on synthetic BAMs "
        "from simulated read counts"
    )

    prosolo_parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="Path to data input file (requires format of `simulate` command)"
    )
    prosolo_parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Enable verbose logging"
    )
    prosolo_parser.add_argument(
        "-o", "--output", type=str,
        help="Output file for ProSolo results JSON (optional)"
    )
    prosolo_parser.add_argument(
        "--base-error-rate", type=float, default=0.01,
        help="Sequencing base-call error rate (mapped to Phred base quality)"
    )
    prosolo_parser.add_argument(
        "--posterior-prob-threshold", type=float, default=0.5,
        help="Posterior probability threshold for calling a variant"
    )
    prosolo_parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel workers (0 = use all CPUs)"
    )

    tree_parser = subparsers.add_parser(
        "simulate-tree",
        help="Simulate data for a cell population using a coalescent tree"
    )

    tree_parser.add_argument(
        "-n", "--n-cells", type=int, required=True,
        help="Number of cells (leaves in the coalescent tree)",
    )
    tree_parser.add_argument(
        "--n-mutations", type=int, default=10,
        help="Number of somatic mutations to place on the tree",
    )
    tree_parser.add_argument(
        "--effective-population-size", type=float, default=1.0,
        help="Effective population size (Ne) for coalescent process",
    )
    tree_parser.add_argument(
        "--copy-number-amplification-prob", type=float, default=0.1,
        help="Probability of copy number amplification (C>2) per mutation",
    )
    tree_parser.add_argument(
        "--copy-number-deletion-prob", type=float, default=0.1,
        help="Probability of copy number deletion (C=1) per mutation",
    )
    tree_parser.add_argument(
        "--purity", type=float, default=0.8, help="Tumor purity (0-1)",
    )
    tree_parser.add_argument(
        "--kappa", type=float, default=100.0,
        help="Bulk overdispersion concentration parameter",
    )
    tree_parser.add_argument(
        "--epsilon-sc", type=float, default=0.10,
        help="Single-cell sequencing error rate",
    )
    tree_parser.add_argument(
        "--n-germline-snps", type=int, default=500,
        help="Number of germline SNPs to simulate per cell per locus",
    )
    tree_parser.add_argument(
        "--bulk-coverage", type=int, default=100,
        help="Bulk sequencing coverage",
    )
    tree_parser.add_argument(
        "--sc-coverage", type=int, default=3,
        help="Single-cell sequencing coverage",
    )
    tree_parser.add_argument(
        "--tau-range-low", type=float, default=20.0,
        help="Lower bound of per-locus mean tau",
    )
    tree_parser.add_argument(
        "--tau-range-high", type=float, default=80.0,
        help="Upper bound of per-locus mean tau"
    )
    tree_parser.add_argument(
        "--tau-cv", type=float, default=0.3,
        help="Coefficient of variation for per-cell tau around the locus mean"
    )
    tree_parser.add_argument(
        "--error-model", type=str, default="betabinomial",
        choices=["betabinomial", "lodato"],
        help="Amplification error model for read generation: "
        "'betabinomial' (SC-BIG model) or 'lodato' (MDA model)"
    )
    tree_parser.add_argument(
        "--ccf-distribution", type=str, default="natural",
        choices=["natural", "uniform"],
        help="Mutation CCF distribution: 'natural' (branch-length "
        "weighted) or 'uniform' (equal probability per CCF decile)"
    )
    tree_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    tree_parser.add_argument(
        "-o", "--output-dir", type=str,
        help="Output directory for per-mutation JSONs and metadata",
    )
    tree_parser.add_argument(
        "-p", "--plot", type=str,
        help="Output file for tree simulation plots",
    )
    tree_parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print detailed simulation summary",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare SC-BIG and ProSolo results"
    )
    compare_parser.add_argument(
        "--scbig", nargs="+", required=True,
        help="SC-BIG inference result JSON files (one per mutation)",
    )
    compare_parser.add_argument(
        "--prosolo", nargs="+", required=True,
        help="ProSolo inference result JSON files (one per mutation)",
    )
    compare_parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output PNG file for comparison figure",
    )
    compare_parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Enable verbose logging",
    )

    args: argparse.Namespace = parser.parse_args()
    setup_logging(verbose=args.verbose if hasattr(args, "verbose") else False)
    selected_subcommand: str = args.subcommand
    if selected_subcommand == "simulate":
        simulate_command(args)
    elif selected_subcommand == "infer":
        infer_command(args)
    elif selected_subcommand == "infer-prosolo":
        infer_prosolo_command(args)
    elif selected_subcommand == "simulate-tree":
        simulate_tree_command(args)
    elif selected_subcommand == "compare":
        compare_command(args)
    else:
        logger.error(f"Unknown subcommand: {selected_subcommand}")
        raise ValueError(f"Unknown subcommand: {selected_subcommand}")


if __name__ == "__main__":
    main()
