"""
Simulation tests
================

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

import numpy as np

from src.simulation import (
    BayesianSNVSimulator,
    Hyperparameters,
    BulkData,
    SingleCellData,
)


class TestHyperparameters:
    """Test Hyperparameters dataclass."""

    def test_default_values(self):
        """Test that default hyperparameters are valid."""
        hp = Hyperparameters()
        assert 0 <= hp.true_purity <= 1
        assert hp.true_copy_number >= 1
        assert hp.true_multiplicity >= 1
        assert 0 <= hp.true_ccf <= 1
        assert hp.kappa > 0
        assert 0 <= hp.epsilon_sc <= 1
        assert hp.n_germline_snps > 0
        assert hp.n_b > 0
        assert hp.sc_coverage > 0

    def test_custom_values(self):
        """Test custom hyperparameter creation."""
        hp = Hyperparameters(
            true_purity=0.7,
            true_copy_number=3,
            true_multiplicity=2,
            true_ccf=0.3,
        )
        assert hp.true_purity == 0.7
        assert hp.true_copy_number == 3
        assert hp.true_multiplicity == 2
        assert hp.true_ccf == 0.3


class TestBayesianSNVSimulator:
    """Test BayesianSNVSimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create a simulator with default parameters."""
        return BayesianSNVSimulator(seed=42)

    def test_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.hyperparams is not None
        assert isinstance(simulator.hyperparams, Hyperparameters)

    def test_simulate_bulk_data(self, simulator):
        """Test bulk data simulation."""
        bulk_data = simulator._simulate_bulk_data()
        assert isinstance(bulk_data, BulkData)
        assert bulk_data.k_b >= 0
        assert bulk_data.n_b > 0
        assert bulk_data.k_b <= bulk_data.n_b

    def test_simulate_bulk_data_invalid_multiplicity(self):
        """Test that invalid multiplicity raises ValueError."""
        hp = Hyperparameters(true_copy_number=2, true_multiplicity=3)
        simulator = BayesianSNVSimulator(hyperparams=hp)
        with pytest.raises(ValueError, match="Multiplicity.*cannot exceed"):
            simulator._simulate_bulk_data()

    def test_simulate_bulk_data_invalid_ccf(self):
        """Test that invalid CCF raises ValueError."""
        hp = Hyperparameters(true_ccf=1.5)
        simulator = BayesianSNVSimulator(hyperparams=hp)
        with pytest.raises(ValueError, match="CCF.*must be between 0 and 1"):
            simulator._simulate_bulk_data()

    def test_simulate_single_cell_data(self, simulator):
        """Test single cell data simulation."""
        bulk_data = simulator._simulate_bulk_data()
        sc_data = simulator._simulate_single_cell_data(bulk_data)

        assert isinstance(sc_data, SingleCellData)
        assert sc_data.k_sc >= 0
        assert sc_data.n_sc > 0
        assert sc_data.k_sc <= sc_data.n_sc
        assert isinstance(sc_data.true_variant_present, bool)
        assert isinstance(sc_data.germline_snps, list)

    def test_simulate_dataset(self, simulator):
        """Test complete dataset simulation."""
        n_cells = 10
        bulk_data, single_cells = simulator.simulate_dataset(n_cells)

        assert isinstance(bulk_data, BulkData)
        assert len(single_cells) == n_cells
        assert all(isinstance(sc, SingleCellData) for sc in single_cells)

    def test_calculate_expected_vaf(self, simulator):
        """Test expected VAF calculation."""
        vaf = simulator._calculate_expected_vaf(
            purity=0.8, ccf=0.5, multiplicity=1, copy_number=2
        )
        assert 0 <= vaf <= 1

        vaf_pure = simulator._calculate_expected_vaf(
            purity=1.0, ccf=1.0, multiplicity=1, copy_number=2
        )
        assert abs(vaf_pure - 0.5) < 0.001

    def test_sample_sc_concentration(self, simulator):
        """Test single cell concentration sampling."""
        concentrations = [
            simulator._sample_sc_concentration() for _ in range(100)
        ]
        assert all(c >= 1.0 for c in concentrations)
        assert np.mean(concentrations) > 1.0

    def test_sample_beta_binomial(self, simulator):
        """Test beta-binomial sampling."""
        samples = [
            simulator._sample_beta_binomial(
                n=100, expected_vaf=0.5, concentration=10
            )
            for _ in range(100)
        ]
        assert all(0 <= s <= 100 for s in samples)
        # @NOTE(ds): Mean should be around 50 for VAF=0.5, n=100.
        assert 30 <= np.mean(samples) <= 70

    def test_sample_beta_binomial_edge_cases(self, simulator):
        """Test beta-binomial sampling with edge case parameters."""
        sample = simulator._sample_beta_binomial(
            n=100, expected_vaf=0.0, concentration=10
        )
        assert 0 <= sample <= 10

        sample = simulator._sample_beta_binomial(
            n=100, expected_vaf=1.0, concentration=10
        )
        assert 90 <= sample <= 100

    def test_effective_concentration_penalty(self, simulator):
        """Test that allelic imbalance reduces effective concentration."""
        penalty_balanced = 4.0 * 0.5 * (1.0 - 0.5)
        assert penalty_balanced == 1.0

        penalty_moderate = 4.0 * 0.3 * (1.0 - 0.3)
        assert penalty_moderate == 0.84

        penalty_strong = 4.0 * 0.1 * (1.0 - 0.1)
        assert abs(penalty_strong - 0.36) < 0.001

        penalty_extreme = 4.0 * 0.01 * (1.0 - 0.01)
        assert abs(penalty_extreme - 0.0396) < 0.001

    def test_germline_symmetric_vaf(self):
        """
        Test that germline SNPs use symmetric Beta-Binomial (mean VAF ≈ 0.5).

        In the simulation, germline heterozygous SNPs are sampled from a
        symmetric Beta-Binomial with p=0.5.
        """
        hp = Hyperparameters(
            base_concentration_mean=50.0,
            base_concentration_cv=0.1,
            n_germline_snps=500,
            sc_coverage=10,
        )
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=50)

        all_vafs = []
        for sc in single_cells:
            for k, n in sc.germline_snps:
                if n > 0:
                    all_vafs.append(k / n)

        mean_vaf = np.mean(all_vafs)
        assert 0.45 <= mean_vaf <= 0.55, \
            f"Mean VAF {mean_vaf:.3f} not symmetric"

        vafs_below_half = sum(1 for v in all_vafs if v < 0.5)
        vafs_above_half = sum(1 for v in all_vafs if v > 0.5)
        ratio = vafs_below_half / vafs_above_half
        assert 0.8 <= ratio <= 1.2, "VAF distribution not balanced around 0.5"

    def test_simulate_germline_snps(self, simulator):
        """Test germline SNP simulation."""
        concentration = 20.0
        germline_snps = simulator._simulate_germline_snps(
            concentration=concentration,
        )

        assert isinstance(germline_snps, list)
        assert len(germline_snps) <= simulator.hyperparams.n_germline_snps

        for alt_reads, total_reads in germline_snps:
            assert alt_reads >= 0
            assert total_reads > 0  # bc. SNPs with 0 coverage are filtered out
            assert alt_reads <= total_reads

    def test_reproducibility(self):
        """Test that simulations are reproducible with same seed."""
        sim1 = BayesianSNVSimulator(seed=123)
        bulk1, sc1 = sim1.simulate_dataset(n_single_cells=5)

        sim2 = BayesianSNVSimulator(seed=123)
        bulk2, sc2 = sim2.simulate_dataset(n_single_cells=5)

        assert bulk1.k_b == bulk2.k_b
        assert len(sc1) == len(sc2)
        assert sc1[0].k_sc == sc2[0].k_sc


class TestSimulationValidation:
    """Test simulation validation logic."""

    @pytest.fixture
    def valid_simulation(self):
        """Create a valid simulation."""
        simulator = BayesianSNVSimulator(seed=42)
        return simulator.simulate_dataset(n_single_cells=100)

    def test_bulk_read_counts_valid(self, valid_simulation):
        """Test that bulk read counts are valid."""
        bulk_data, _ = valid_simulation
        assert bulk_data.k_b <= bulk_data.n_b
        assert bulk_data.k_b >= 0
        assert bulk_data.n_b > 0

    def test_bulk_parameters_in_range(self, valid_simulation):
        """Test that bulk parameters are in valid ranges."""
        bulk_data, _ = valid_simulation
        assert 0 <= bulk_data.true_ccf <= 1
        assert 0 <= bulk_data.true_purity <= 1
        assert 1 <= bulk_data.true_copy_number <= 10
        assert 1 <= bulk_data.true_multiplicity <= bulk_data.true_copy_number

    def test_single_cell_read_counts_valid(self, valid_simulation):
        """Test that single cell read counts are valid."""
        _, single_cells = valid_simulation

        for sc in single_cells:
            assert sc.k_sc <= sc.n_sc
            assert sc.k_sc >= 0

            for alt_reads, total_reads in sc.germline_snps:
                assert alt_reads <= total_reads
                assert alt_reads >= 0
                assert total_reads > 0

    def test_ccf_consistency(self, valid_simulation):
        """Test that CCF is consistent with variant presence."""
        bulk_data, single_cells = valid_simulation

        n_variant_present = sum(sc.true_variant_present for sc in single_cells)
        expected_variant = bulk_data.true_ccf * len(single_cells)
        std_dev = np.sqrt(
            # @NOTE(ds): Binomial variance is n * p * (1-p).
            len(single_cells) * bulk_data.true_ccf * (1 - bulk_data.true_ccf)
        )
        assert abs(n_variant_present - expected_variant) <= 3.0 * std_dev

    def test_germline_vaf_distribution(self, valid_simulation):
        """Test that germline VAF distribution is centered around 0.5."""
        _, single_cells = valid_simulation

        all_germline_vafs = []
        for sc in single_cells:
            for alt_reads, total_reads in sc.germline_snps:
                if total_reads > 0:
                    all_germline_vafs.append(alt_reads / total_reads)

        if all_germline_vafs:
            mean_germline_vaf = np.mean(all_germline_vafs)
            assert abs(mean_germline_vaf - 0.5) < 0.15

    def test_somatic_vaf_expectation(self):
        """
        Test that somatic VAF matches expectation in variant-present cells.
        """
        hp = Hyperparameters(
            true_purity=0.8,
            true_copy_number=2,
            true_multiplicity=1,
            true_ccf=0.8,
            sc_coverage=20,
        )
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = \
            simulator.simulate_dataset(n_single_cells=100)

        variant_present_cells = [
            sc for sc in single_cells if sc.true_variant_present
        ]

        if variant_present_cells:
            expected_somatic_vaf = (
                bulk_data.true_multiplicity / bulk_data.true_copy_number
            )
            somatic_vafs = [
                sc.k_sc / max(sc.n_sc, 1) for sc in variant_present_cells
            ]
            mean_somatic_vaf = np.mean(somatic_vafs)
            assert abs(mean_somatic_vaf - expected_somatic_vaf) < 0.25


class TestParameterizedHyperparameters:
    """Test simulation with various hyperparameter combinations."""

    @pytest.mark.parametrize("purity", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_various_purities(self, purity):
        """Test simulation with different purity values."""
        hp = Hyperparameters(true_purity=purity)
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=20)

        assert bulk_data.true_purity == purity
        assert 0 <= bulk_data.true_expected_vaf <= 1

    @pytest.mark.parametrize("ccf", [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    def test_various_ccf_values(self, ccf):
        """Test simulation with different CCF values."""
        hp = Hyperparameters(true_ccf=ccf)
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = \
            simulator.simulate_dataset(n_single_cells=500)

        assert bulk_data.true_ccf == ccf

        n_variant_present = sum(sc.true_variant_present for sc in single_cells)
        expected = ccf * len(single_cells)

        if ccf == 0.0:
            assert n_variant_present == 0
        elif ccf == 1.0:
            assert n_variant_present == len(single_cells)
        else:
            std_dev = np.sqrt(len(single_cells) * ccf * (1 - ccf))
            assert abs(n_variant_present - expected) <= 4.0 * std_dev

    @pytest.mark.parametrize(
        "copy_number,multiplicity",
        [
            (1, 1),
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 2),
            (5, 3),
            (10, 5),
        ],
    )
    def test_various_copy_number_multiplicity(self, copy_number, multiplicity):
        """Test simulation with different CN/multiplicity combinations."""
        hp = Hyperparameters(
            true_copy_number=copy_number, true_multiplicity=multiplicity
        )
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=20)

        assert bulk_data.true_copy_number == copy_number
        assert bulk_data.true_multiplicity == multiplicity

        expected_sc_vaf = multiplicity / copy_number
        assert 0 < expected_sc_vaf <= 1

    @pytest.mark.parametrize(
        "bc_mean,bc_cv",
        [
            (120, 0.01),
            (50, 0.01),
            (80, 0.1),
            (80, 0.2),
            (40, 0.15),
            (30, 0.2),
            (20, 0.15),
            (10, 0.1),
            (2, 0.1),
            (2, 0.5),
        ],
    )
    def test_various_dropout_scenarios(self, bc_mean, bc_cv):
        """Test simulation with different dropout/bias scenarios."""
        hp = Hyperparameters(
            base_concentration_mean=bc_mean,
            base_concentration_cv=bc_cv,
            sc_coverage=20,
            n_germline_snps=100,
        )
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = \
            simulator.simulate_dataset(n_single_cells=500)

        true_concentrations = []
        mean_vafs = []
        for sc in single_cells:
            true_concentrations.append(sc.true_concentration)
            for (k, n) in sc.germline_snps:
                if n > 0:
                    mean_vafs.append(k / n)
        assert k > 0
        assert n > 0

        mean_vaf = np.mean(mean_vafs)
        assert abs(mean_vaf - 0.5) < 0.1

        mean_concentration = np.mean(true_concentrations)
        assert abs(mean_concentration - bc_mean) < 1.0

    @pytest.mark.parametrize("coverage", [1, 3, 5, 10, 20])
    def test_various_coverage_levels(self, coverage):
        """Test simulation with different coverage levels."""
        hp = Hyperparameters(sc_coverage=coverage, n_b=coverage * 20)
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=20)

        cov_std = np.sqrt(coverage)
        for sc in single_cells:
            assert (coverage - 3*cov_std) <= sc.n_sc <= (coverage + 3*cov_std)


class TestExtremeParameterCombinations:
    """Test simulation with extreme parameter combinations."""

    def test_extreme_low_purity_low_ccf(self):
        """Test with very low purity and CCF."""
        hp = Hyperparameters(true_purity=0.1, true_ccf=0.01)
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = \
            simulator.simulate_dataset(n_single_cells=100)

        assert bulk_data.true_expected_vaf < 0.01

    def test_extreme_high_copy_number(self):
        """Test with maximum copy number."""
        hp = Hyperparameters(true_copy_number=10, true_multiplicity=5)
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=50)

        assert bulk_data.true_copy_number == 10
        assert bulk_data.true_multiplicity == 5

    def test_extreme_pure_clonal(self):
        """Test with pure tumor, clonal variant."""
        hp = Hyperparameters(true_purity=1.0, true_ccf=1.0)
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=50)

        assert all(sc.true_variant_present for sc in single_cells)

    def test_extreme_no_germline_snps(self):
        """Test with no germline SNPs."""
        hp = Hyperparameters(n_germline_snps=0)
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=20)

        for sc in single_cells:
            assert len(sc.germline_snps) == 0

    def test_extreme_single_cell(self):
        """Test with just one single cell."""
        simulator = BayesianSNVSimulator(seed=42)
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=1)

        assert len(single_cells) == 1
        assert isinstance(single_cells[0].true_variant_present, bool)
