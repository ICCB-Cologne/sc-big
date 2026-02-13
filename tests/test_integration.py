"""
Simulation Integration tests
============================

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
import pytest
import json
import tempfile
import os

import numpy as np

from src.simulation import (
    BayesianSNVSimulator,
    Hyperparameters,
    export_simulation,
    import_simulation_from_dict,
    import_simulation_from_json,
)

from src.inference import compute_population_tau


class TestExportImportIntegration:
    """Test export and import functionality."""

    @pytest.fixture
    def simulation_data(self):
        """Create simulation data for testing."""
        hyperparams = Hyperparameters()
        simulator = BayesianSNVSimulator(hyperparams=hyperparams, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=10)
        return bulk_data, single_cells, hyperparams

    def test_export_structure(self, simulation_data):
        """Test that exported data has correct structure."""
        bulk_data, single_cells, hyperparams = simulation_data
        exported = export_simulation(
            bulk_data, single_cells, hyperparams, output_file=None
        )

        assert "bulk" in exported
        assert "single_cells" in exported
        assert "ground_truth" in exported
        assert "hyperparameters" in exported

        assert "k_b" in exported["bulk"]
        assert "n_b" in exported["bulk"]

        assert "bulk" in exported["ground_truth"]
        assert "single_cells" in exported["ground_truth"]

        assert len(exported["single_cells"]) == 10
        assert len(exported["ground_truth"]["single_cells"]) == 10

    def test_export_import_roundtrip(self, simulation_data):
        """Test that export followed by import preserves data."""
        bulk_data, single_cells, hyperparams = simulation_data
        exported = export_simulation(
            bulk_data, single_cells, hyperparams, output_file=None
        )
        imported_bulk, imported_sc, imported_hyper = \
            import_simulation_from_dict(
                exported
            )

        assert imported_bulk.k_b == bulk_data.k_b
        assert imported_bulk.n_b == bulk_data.n_b
        assert imported_bulk.true_ccf == bulk_data.true_ccf
        assert imported_bulk.true_purity == bulk_data.true_purity
        assert imported_bulk.true_copy_number == bulk_data.true_copy_number
        assert imported_bulk.true_multiplicity == bulk_data.true_multiplicity
        assert imported_bulk.true_expected_vaf == bulk_data.true_expected_vaf

        assert len(imported_sc) == len(single_cells)
        for orig, imp in zip(single_cells, imported_sc):
            assert imp.k_sc == orig.k_sc
            assert imp.n_sc == orig.n_sc
            assert imp.true_variant_present == orig.true_variant_present
            assert imp.true_sc_alpha == orig.true_sc_alpha
            assert imp.true_sc_beta == orig.true_sc_beta
            assert len(imp.germline_snps) == len(orig.germline_snps)
            for (imp_alt, imp_tot), (orig_alt, orig_tot) in zip(
                imp.germline_snps, orig.germline_snps
            ):
                assert imp_alt == orig_alt
                assert imp_tot == orig_tot

        assert imported_hyper.true_purity == hyperparams.true_purity
        assert imported_hyper.true_copy_number == hyperparams.true_copy_number
        assert (
            imported_hyper.true_multiplicity == hyperparams.true_multiplicity
        )
        assert imported_hyper.true_ccf == hyperparams.true_ccf
        assert imported_hyper.kappa == hyperparams.kappa
        assert imported_hyper.epsilon_sc == hyperparams.epsilon_sc
        assert imported_hyper.n_germline_snps == hyperparams.n_germline_snps
        assert imported_hyper.n_b == hyperparams.n_b
        assert imported_hyper.sc_coverage == hyperparams.sc_coverage
        assert (
            imported_hyper.base_concentration_mean ==
            hyperparams.base_concentration_mean
        )
        assert (
            imported_hyper.base_concentration_cv ==
            hyperparams.base_concentration_cv
        )

    def test_export_json_serializable(self, simulation_data):
        """Test that exported data is JSON serializable."""
        bulk_data, single_cells, hyperparams = simulation_data
        exported = export_simulation(
            bulk_data, single_cells, hyperparams, output_file=None
        )

        json_str = json.dumps(exported)
        assert isinstance(json_str, str)

        loaded = json.loads(json_str)
        assert loaded == exported

    def test_export_import_with_json_file(self, simulation_data):
        """Test export/import through JSON file."""
        bulk_data, single_cells, hyperparams = simulation_data

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_path = f.name

        try:
            export_simulation(
                bulk_data, single_cells, hyperparams, output_file=temp_path
            )
            imported_bulk, imported_sc, imported_hyper = \
                import_simulation_from_json(temp_path)

            assert imported_bulk.k_b == bulk_data.k_b
            assert len(imported_sc) == len(single_cells)
            assert imported_hyper.true_ccf == hyperparams.true_ccf
        finally:
            os.unlink(temp_path)

    def test_import_validates_data_types(self, simulation_data):
        """Test that import correctly converts data types."""
        bulk_data, single_cells, hyperparams = simulation_data
        exported = export_simulation(bulk_data, single_cells, hyperparams)
        exported["bulk"]["k_b"] = str(exported["bulk"]["k_b"])
        exported["bulk"]["n_b"] = str(exported["bulk"]["n_b"])

        imported_bulk, imported_sc, imported_hyper = \
            import_simulation_from_dict(
                exported
            )

        assert isinstance(imported_bulk.k_b, int)
        assert isinstance(imported_bulk.n_b, int)

    def test_export_preserves_edge_cases(self):
        """Test export/import with edge case parameters."""
        hp = Hyperparameters(
            true_ccf=0.0,
            true_purity=1.0,
            true_copy_number=1,
            true_multiplicity=1,
            n_germline_snps=0,
        )
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=1)

        exported = export_simulation(bulk_data, single_cells, hp)
        imported_bulk, imported_sc, imported_hyper = \
            import_simulation_from_dict(
                exported
            )

        assert imported_bulk.true_ccf == 0.0
        assert imported_bulk.true_purity == 1.0
        assert len(imported_sc) == 1
        assert len(imported_sc[0].germline_snps) == 0
        assert imported_hyper.true_ccf == 0.0
        assert imported_hyper.n_germline_snps == 0

    def test_hyperparameters_required(self):
        """Test that hyperparameters are required in exported data."""
        simulator = BayesianSNVSimulator(seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=5)
        hp = Hyperparameters()

        exported = export_simulation(bulk_data, single_cells, hp)
        assert "hyperparameters" in exported
        del exported["hyperparameters"]
        with pytest.raises(ValueError, match="Hyperparameters not found"):
            import_simulation_from_dict(exported)

    def test_custom_hyperparameters_preserved(self):
        """Test that custom hyperparameter values are correctly preserved."""
        hp = Hyperparameters(
            true_purity=0.65,
            true_copy_number=4,
            true_multiplicity=3,
            true_ccf=0.35,
            kappa=250.0,
            epsilon_sc=0.05,
            n_germline_snps=30,
            n_b=200,
            sc_coverage=10,
            base_concentration_mean=50.0,
            base_concentration_cv=0.15,
        )
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=5)

        exported = export_simulation(bulk_data, single_cells, hp)
        imported_bulk, imported_sc, imported_hyper = \
            import_simulation_from_dict(
                exported
            )

        assert imported_hyper.true_purity == 0.65
        assert imported_hyper.true_copy_number == 4
        assert imported_hyper.true_multiplicity == 3
        assert imported_hyper.true_ccf == 0.35
        assert imported_hyper.kappa == 250.0
        assert imported_hyper.epsilon_sc == 0.05
        assert imported_hyper.n_germline_snps == 30
        assert imported_hyper.n_b == 200
        assert imported_hyper.sc_coverage == 10
        assert imported_hyper.base_concentration_mean == 50.0
        assert imported_hyper.base_concentration_cv == 0.15


class TestEndToEndWorkflow:
    """Test complete simulation workflows."""

    def test_simulate_export_workflow(self):
        """Test complete workflow from simulation to export."""
        hp = Hyperparameters(
            true_purity=0.6,
            true_copy_number=3,
            true_multiplicity=1,
            true_ccf=0.4,
        )
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=123)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=20)

        exported = export_simulation(bulk_data, single_cells, hp)
        assert "bulk" in exported
        assert "single_cells" in exported
        assert "hyperparameters" in exported
        for sc in exported["single_cells"]:
            assert "k_sc" in sc
            assert "n_sc" in sc
            assert "germline_snps" in sc

        assert "ground_truth" in exported

    def test_multiple_simulations_independent(self):
        """
        Test that multiple simulations with different seeds are independent.
        """
        hp = Hyperparameters(true_ccf=0.5)

        sim1 = BayesianSNVSimulator(hyperparams=hp, seed=1)
        bulk1, sc1 = sim1.simulate_dataset(n_single_cells=10)

        sim2 = BayesianSNVSimulator(hyperparams=hp, seed=2)
        bulk2, sc2 = sim2.simulate_dataset(n_single_cells=10)

        assert bulk1.k_b != bulk2.k_b or sc1[0].k_sc != sc2[0].k_sc

    def test_large_scale_simulation(self):
        """Test simulation with large number of cells."""
        hp = Hyperparameters()
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = \
            simulator.simulate_dataset(n_single_cells=500)

        assert len(single_cells) == 500
        exported = export_simulation(bulk_data, single_cells, hp)
        assert len(exported["single_cells"]) == 500
        assert "hyperparameters" in exported

        json_str = json.dumps(exported)
        assert len(json_str) > 0


class TestPopulationTauInferenceAccuracy:
    """
    Integration tests for population tau (concentration parameter) inference.

    These tests verify that the symmetric Beta-Binomial MLE approach allows
    accurate recovery of the population-level concentration parameter.
    """

    def test_population_tau_low_concentration(self):
        """
        Test population tau inference with low concentration (high bias).

        With base_concentration_mean=10, cells have high amplification bias.
        """
        hp = Hyperparameters(
            base_concentration_mean=10.0,
            base_concentration_cv=0.1,
            n_germline_snps=2000,
            sc_coverage=10,
        )
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=5)

        all_germline = [sc.germline_snps for sc in single_cells]
        pop_tau, n_snps = compute_population_tau(all_germline)
        true_taus = [sc.true_sc_alpha + sc.true_sc_beta for sc in single_cells]
        true_median = np.median(true_taus)

        rel_error = abs(pop_tau - true_median) / true_median
        assert rel_error < 0.4, \
            f"Population tau {pop_tau:.2f} differs from true median " \
            f"{true_median:.2f} by {100*rel_error:.1f}%"

    def test_population_tau_medium_concentration(self):
        """Test population tau inference with medium concentration."""
        hp = Hyperparameters(
            base_concentration_mean=30.0,
            base_concentration_cv=0.1,
            n_germline_snps=2000,
            sc_coverage=10,
        )
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=5)

        all_germline = [sc.germline_snps for sc in single_cells]
        pop_tau, n_snps = compute_population_tau(all_germline)
        true_taus = [sc.true_sc_alpha + sc.true_sc_beta for sc in single_cells]
        true_median = np.median(true_taus)

        rel_error = abs(pop_tau - true_median) / true_median
        assert rel_error < 0.3, \
            f"Population tau {pop_tau:.2f} differs from true median " \
            f"{true_median:.2f} by {100*rel_error:.1f}%"

    def test_population_tau_high_concentration(self):
        """
        Test tau inference with high concentration (low bias).
        """
        hp = Hyperparameters(
            base_concentration_mean=100.0,
            base_concentration_cv=0.1,
            n_germline_snps=1000,
            sc_coverage=10,
        )
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=20)

        all_germline = [sc.germline_snps for sc in single_cells]
        pop_tau, n_snps = compute_population_tau(all_germline)
        true_taus = [sc.true_sc_alpha + sc.true_sc_beta for sc in single_cells]
        true_median = np.median(true_taus)

        rel_error = abs(pop_tau - true_median) / true_median
        assert rel_error < 0.2, \
            f"Population tau {pop_tau:.2f} differs from true median " \
            f"{true_median:.2f} by {100*rel_error:.1f}%"

    def test_population_tau_very_high_concentration(self):
        """
        Test tau inference with very high concentration.
        """
        hp = Hyperparameters(
            base_concentration_mean=200.0,
            base_concentration_cv=0.1,
            n_germline_snps=2000,
            sc_coverage=10,
        )
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=30)

        all_germline = [sc.germline_snps for sc in single_cells]
        pop_tau, n_snps = compute_population_tau(all_germline)
        true_taus = [sc.true_sc_alpha + sc.true_sc_beta for sc in single_cells]
        true_median = np.median(true_taus)

        rel_error = abs(pop_tau - true_median) / true_median
        assert rel_error < 0.15, \
            f"Population tau {pop_tau:.2f} differs from true median " \
            f"{true_median:.2f} by {100*rel_error:.1f}%"

    def test_population_tau_with_varying_cv(self):
        """
        Test tau inference with different coefficients of variation.
        """
        hp_low_cv = Hyperparameters(
            base_concentration_mean=50.0,
            base_concentration_cv=0.1,
            n_germline_snps=2000,
            sc_coverage=10,
        )
        simulator_low = BayesianSNVSimulator(hyperparams=hp_low_cv, seed=42)
        bulk_data, sc_low = simulator_low.simulate_dataset(n_single_cells=5)

        all_germline_low = [sc.germline_snps for sc in sc_low]
        pop_tau_low, _ = compute_population_tau(all_germline_low)
        true_taus_low = [sc.true_sc_alpha + sc.true_sc_beta for sc in sc_low]

        hp_high_cv = Hyperparameters(
            base_concentration_mean=50.0,
            base_concentration_cv=0.5,
            n_germline_snps=2000,
            sc_coverage=10,
        )
        simulator_high = BayesianSNVSimulator(hyperparams=hp_high_cv, seed=42)
        bulk_data, sc_high = simulator_high.simulate_dataset(n_single_cells=5)

        all_germline_high = [sc.germline_snps for sc in sc_high]
        pop_tau_high, _ = compute_population_tau(all_germline_high)
        true_taus_high = [sc.true_sc_alpha + sc.true_sc_beta for sc in sc_high]

        error_low = abs(pop_tau_low - np.median(true_taus_low)) / \
            np.median(true_taus_low)
        error_high = abs(pop_tau_high - np.median(true_taus_high)) / \
            np.median(true_taus_high)

        assert error_low < 0.2, \
            f"High error {100*error_low:.1f}% with low CV"
        assert error_high < 0.25, \
            f"High error {100*error_high:.1f}% with high CV"

        assert np.std(true_taus_high) > np.std(true_taus_low), \
            "High CV should produce more variable tau values"
