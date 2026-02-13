"""
Bayesian single-cell DNA SNV caller
===================================

This package implements a Bayesian model for re-detecting somatic single
nucleotide variants (SNVs) in single-cell DNA sequencing data using
complementary bulk sequencing information.

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
from src.logging_config import setup_logging, get_logger
from src.simulation import (
    BayesianSNVSimulator,
    BulkData,
    SingleCellData,
    Hyperparameters,
    print_simulation_summary,
    plot_simulation_results,
    export_simulation,
    import_simulation_from_dict,
    import_simulation_from_json,
)
from src.inference import (
    compute_bulk_posterior,
    compute_variant_probability,
    InferenceInputs,
)
from src.tree_simulation import (
    CoalescentSNVSimulator,
    TreeHyperparameters,
    TreeNode,
    Mutation,
)
from src.inference_prosolo import prosolo_call_native, lodato_af, sample_lodato
from src.comparison import plot_comparison

__version__ = "0.0.1"
__author__ = "Daniel Schütte"

__all__ = [
    "setup_logging",
    "get_logger",
    "BayesianSNVSimulator",
    "BulkData",
    "SingleCellData",
    "Hyperparameters",
    "print_simulation_summary",
    "plot_simulation_results",
    "export_simulation",
    "import_simulation_from_dict",
    "import_simulation_from_json",
    "compute_bulk_posterior",
    "compute_variant_probability",
    "InferenceInputs",
    "CoalescentSNVSimulator",
    "TreeHyperparameters",
    "TreeNode",
    "Mutation",
    "prosolo_call_native",
    "lodato_af",
    "sample_lodato",
    "plot_comparison",
]
