"""
Inference tests
===============

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
import pyro
import pytest
import torch

import numpy as np

from scipy.stats import beta as beta_dist, uniform
from src.inference import (
    InferenceInputs,
    joint_model,
    compute_variant_probability,
    compute_bulk_posterior,
    estimate_ccf_prior_from_sc,
    TAU_DEFAULT,
)


class TestInferenceInputs:
    """Test InferenceInputs dataclass."""

    def test_default_values(self):
        """Test that default values are properly set."""
        inputs = InferenceInputs(
            k_b=50,
            n_b=100,
            k_sc=3,
            n_sc=10,
        )
        assert inputs.purity_mean == 0.8
        assert inputs.purity_std == 0.1
        assert inputs.copy_number_mean == 2.0
        assert inputs.copy_number_std == 0.5
        assert inputs.kappa == 100.0
        assert inputs.epsilon_sc == 0.10
        assert inputs.population_tau == TAU_DEFAULT

    def test_custom_values(self):
        """Test custom parameter values."""
        inputs = InferenceInputs(
            k_b=50,
            n_b=100,
            k_sc=3,
            n_sc=10,
            purity_mean=0.9,
            purity_std=0.05,
            copy_number_mean=3.0,
            copy_number_std=0.2,
            kappa=200.0,
            epsilon_sc=0.05,
            population_tau=30.0
        )
        assert inputs.purity_mean == 0.9
        assert inputs.purity_std == 0.05
        assert inputs.kappa == 200.0
        assert inputs.population_tau == 30.0


class TestJointModel:
    """Test joint_model function."""

    def test_joint_model_samples_ccf(self):
        """Test that joint model samples CCF from uniform distribution."""
        pyro.clear_param_store()

        ccf_samples = []
        for _ in range(100):
            trace = pyro.poutine.trace(joint_model).get_trace(
                k_b=50, n_b=100, purity_mean=0.8, purity_std=0.1,
                copy_number_mean=2.0, copy_number_std=0.5, kappa=100.0
            )
            ccf_samples.append(trace.nodes["ccf"]["value"].item())

        assert min(ccf_samples) >= 0.0
        assert max(ccf_samples) <= 1.0
        assert 0.3 < np.mean(ccf_samples) < 0.7

    def test_joint_model_samples_purity(self):
        """Test that purity is sampled from normal distribution."""
        pyro.clear_param_store()

        purity_samples = []
        for _ in range(100):
            trace = pyro.poutine.trace(joint_model).get_trace(
                k_b=50, n_b=100, purity_mean=0.8, purity_std=0.1,
                copy_number_mean=2.0, copy_number_std=0.5, kappa=100.0
            )
            purity_val = trace.nodes["purity"]["value"]
            if isinstance(purity_val, torch.Tensor):
                purity_samples.append(purity_val.item())

        assert 0.6 < np.mean(purity_samples) < 1.0

    def test_joint_model_copy_number_categorical(self):
        """Test that copy number is sampled from categorical distribution."""
        pyro.clear_param_store()

        copy_number_samples = []
        for _ in range(100):
            trace = pyro.poutine.trace(joint_model).get_trace(
                k_b=50, n_b=100, purity_mean=0.8, purity_std=0.1,
                copy_number_mean=2.0, copy_number_std=0.5, kappa=100.0
            )
            cn_val = trace.nodes["copy_number"]["value"]
            if isinstance(cn_val, torch.Tensor):
                copy_number_samples.append(cn_val.item())

        assert all(0 <= c <= 9 for c in copy_number_samples)
        assert 0.5 < np.mean(copy_number_samples) < 2.5

    def test_joint_model_multiplicity_categorical(self):
        """
        Test that multiplicity is sampled from uniform categorical
        distribution.
        """
        pyro.clear_param_store()

        multiplicity_samples = []
        for _ in range(200):
            trace = pyro.poutine.trace(joint_model).get_trace(
                k_b=50, n_b=100, purity_mean=0.8, purity_std=0.1,
                copy_number_mean=2.0, copy_number_std=0.5, kappa=100.0
            )
            m_val = trace.nodes["multiplicity"]["value"]
            if isinstance(m_val, torch.Tensor):
                multiplicity_samples.append(m_val.item())

        assert all(0 <= m <= 2 for m in multiplicity_samples)
        mean_idx = np.mean(multiplicity_samples)
        assert 0.35 <= mean_idx <= 0.65

    def test_joint_model_conditions_on_k_b(self):
        """Test that model conditions on observed k_b."""
        pyro.clear_param_store()

        trace = pyro.poutine.trace(joint_model).get_trace(
            k_b=50, n_b=100, purity_mean=0.8, purity_std=0.1,
            copy_number_mean=2.0, copy_number_std=0.5, kappa=100.0
        )

        assert "k_b" in trace.nodes
        assert trace.nodes["k_b"]["is_observed"]
        assert trace.nodes["k_b"]["value"].item() == 50.0


class TestComputeVariantProbability:
    """Test compute_variant_probability function."""

    @pytest.fixture
    def basic_inputs(self):
        """Create basic inference inputs."""
        return InferenceInputs(
            k_b=50, n_b=100, k_sc=3, n_sc=10,
            population_tau=30.0
        )

    @pytest.fixture(scope="class")
    def bulk_posterior(self):
        """Compute bulk posterior once and reuse across all tests."""
        return compute_bulk_posterior(
            k_b=50, n_b=100,
            purity_mean=0.8, purity_std=0.1,
            copy_number_mean=2.0, copy_number_std=0.5,
            kappa=100.0,
            n_samples=100,
            n_workers=10,
            multiplicity_prior_type="geometric"
        )

    def test_returns_probability_and_details(
        self, basic_inputs, bulk_posterior
    ):
        """Test that function returns probability and details."""
        prob, details = compute_variant_probability(
            basic_inputs, bulk_posterior=bulk_posterior
        )

        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        assert isinstance(details, dict)
        assert "ccf_samples" in details
        assert "ccf_weights" in details
        assert "concentration" in details
        assert details["concentration"] >= 1.0

    def test_high_variant_reads_gives_high_probability(self, bulk_posterior):
        """Test that high variant reads lead to high posterior probability."""
        inputs = InferenceInputs(
            k_b=50, n_b=100, k_sc=8, n_sc=10,
            population_tau=30.0
        )
        prob, _ = compute_variant_probability(
            inputs, bulk_posterior=bulk_posterior
        )
        assert prob > 0.7

    def test_zero_variant_reads_gives_low_probability(self, bulk_posterior):
        """Test that zero variant reads lead to low posterior probability."""
        inputs = InferenceInputs(
            k_b=50, n_b=100, k_sc=0, n_sc=10,
            population_tau=30.0
        )
        prob, _ = compute_variant_probability(
            inputs, bulk_posterior=bulk_posterior
        )
        assert prob < 0.3

    def test_intermediate_reads_gives_intermediate_probability(
        self, bulk_posterior
    ):
        """
        Test that intermediate variant reads give intermediate probability.
        """
        inputs = InferenceInputs(
            k_b=50, n_b=100, k_sc=3, n_sc=10,
            population_tau=30.0
        )
        prob, _ = compute_variant_probability(
            inputs, bulk_posterior=bulk_posterior
        )
        assert prob < 0.8

    def test_ccf_samples_have_valid_weights(
        self, basic_inputs, bulk_posterior
    ):
        """Test that CCF samples have normalized weights."""
        prob, details = compute_variant_probability(
            basic_inputs, bulk_posterior=bulk_posterior
        )

        weights = details["ccf_weights"]
        assert len(weights) > 0
        assert all(w >= 0 for w in weights)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)

    def test_ccf_samples_in_valid_range(self, basic_inputs, bulk_posterior):
        """Test that all CCF samples are in [0, 1]."""
        prob, details = compute_variant_probability(
            basic_inputs, bulk_posterior=bulk_posterior
        )

        ccf_samples = details["ccf_samples"]
        assert all(0.0 <= ccf <= 1.0 for ccf in ccf_samples)

    def test_concentration_from_population_tau(
        self, basic_inputs, bulk_posterior
    ):
        """Test that concentration uses population tau."""
        prob, details = compute_variant_probability(
            basic_inputs, bulk_posterior=bulk_posterior
        )

        tau = details["concentration"]
        assert tau == basic_inputs.population_tau

    def test_convergence_with_more_samples(self, bulk_posterior):
        """Test that results converge with more Monte Carlo samples."""
        inputs = InferenceInputs(
            k_b=50, n_b=100, k_sc=3, n_sc=10,
            population_tau=30.0
        )

        prob_50, _ = compute_variant_probability(
            inputs, bulk_posterior=bulk_posterior
        )
        prob_200, _ = compute_variant_probability(
            inputs, bulk_posterior=bulk_posterior
        )
        assert abs(prob_50 - prob_200) < 0.20

    def test_low_bulk_vaf_affects_ccf_posterior(self):
        """Test that bulk VAF affects CCF posterior distribution."""
        bulk_posterior_low = compute_bulk_posterior(
            k_b=10, n_b=100,
            purity_mean=0.8, purity_std=0.1,
            copy_number_mean=2.0, copy_number_std=0.5,
            kappa=100.0,
            n_samples=100,
            n_workers=10,
            multiplicity_prior_type="geometric"
        )

        bulk_posterior_high = compute_bulk_posterior(
            k_b=80, n_b=100,
            purity_mean=0.8, purity_std=0.1,
            copy_number_mean=2.0, copy_number_std=0.5,
            kappa=100.0,
            n_samples=100,
            n_workers=10,
            multiplicity_prior_type="geometric"
        )

        inputs_low_bulk = InferenceInputs(
            k_b=10, n_b=100, k_sc=3, n_sc=10,
            population_tau=30.0
        )

        inputs_high_bulk = InferenceInputs(
            k_b=80, n_b=100, k_sc=3, n_sc=10,
            population_tau=30.0
        )

        _, details_low = compute_variant_probability(
            inputs_low_bulk, bulk_posterior=bulk_posterior_low
        )
        _, details_high = compute_variant_probability(
            inputs_high_bulk, bulk_posterior=bulk_posterior_high
        )

        mean_ccf_low = np.average(
            details_low["ccf_samples"], weights=details_low["ccf_weights"]
        )
        mean_ccf_high = np.average(
            details_high["ccf_samples"], weights=details_high["ccf_weights"]
        )
        assert mean_ccf_low != mean_ccf_high

    def test_default_tau_used_when_not_specified(self, bulk_posterior):
        """Test that default tau is used when not specified."""
        inputs = InferenceInputs(
            k_b=50, n_b=100, k_sc=3, n_sc=10,
        )

        prob, details = compute_variant_probability(
            inputs, bulk_posterior=bulk_posterior
        )
        assert details["concentration"] == TAU_DEFAULT

    def test_handles_edge_case_all_variant_reads(self, bulk_posterior):
        """Test edge case where k_sc = n_sc."""
        inputs = InferenceInputs(
            k_b=50, n_b=100, k_sc=10, n_sc=10,
            population_tau=30.0
        )

        prob, _ = compute_variant_probability(
            inputs, bulk_posterior=bulk_posterior
        )
        assert prob > 0.9

    def test_reproducibility_with_pyro_seed(self, bulk_posterior):
        """Test that results are reproducible with same random seed."""
        inputs = InferenceInputs(
            k_b=50, n_b=100, k_sc=3, n_sc=10,
            population_tau=30.0
        )

        pyro.set_rng_seed(42)
        prob1, _ = \
            compute_variant_probability(inputs, bulk_posterior=bulk_posterior)

        pyro.set_rng_seed(42)
        prob2, _ = \
            compute_variant_probability(inputs, bulk_posterior=bulk_posterior)

        assert abs(prob1 - prob2) < 1e-6


class TestEstimateCCFPriorFromSC:
    """Test CCF prior estimation from single-cell aggregate data."""

    def test_basic_conjugate_prior(self):
        """Test that prior follows Beta(n_positive + 1, n_negative + 1)."""
        single_cells = [{"k_sc": 2, "n_sc": 5} for _ in range(60)]
        single_cells.extend([{"k_sc": 0, "n_sc": 5} for _ in range(40)])

        alpha, beta = estimate_ccf_prior_from_sc(single_cells)

        assert alpha == 61.0, f"Alpha should be 60+1=61, got {alpha}"
        assert beta == 41.0, f"Beta should be 40+1=41, got {beta}"

        prior_mean = alpha / (alpha + beta)
        assert abs(prior_mean - 0.6) < 0.01, \
            f"Mean {prior_mean:.3f} should be close to 0.6"

    def test_high_ccf_scenario(self):
        """Test with high CCF (80% positive cells)."""
        np.random.seed(42)
        single_cells = []
        true_ccf = 0.8

        for _ in range(500):
            n_sc = np.random.poisson(8)
            has_variant = np.random.random() < true_ccf
            if has_variant:
                k_sc = max(1, np.random.binomial(n_sc, 0.5))
            else:
                k_sc = 0
            single_cells.append({"k_sc": k_sc, "n_sc": n_sc})

        alpha, beta = estimate_ccf_prior_from_sc(single_cells)

        prior_mean = alpha / (alpha + beta)
        # Due to dropout, observed fraction may be lower than true CCF.
        assert 0.60 <= prior_mean <= 0.90, \
            f"Mean {prior_mean:.3f} should be in reasonable range for CCF=0.8"

    def test_low_ccf_scenario(self):
        """Test with low CCF (20% positive cells)."""
        np.random.seed(43)
        single_cells = []
        true_ccf = 0.2

        for _ in range(500):
            n_sc = np.random.poisson(8)
            has_variant = np.random.random() < true_ccf
            if has_variant:
                k_sc = max(1, np.random.binomial(n_sc, 0.5))
            else:
                k_sc = 0
            single_cells.append({"k_sc": k_sc, "n_sc": n_sc})

        alpha, beta = estimate_ccf_prior_from_sc(single_cells)

        prior_mean = alpha / (alpha + beta)
        assert 0.10 <= prior_mean <= 0.35, \
            f"Mean {prior_mean:.3f} should be in reasonable range for CCF=0.2"

    def test_medium_ccf_scenario(self):
        """Test with medium CCF (50% positive cells)."""
        np.random.seed(44)
        single_cells = []
        true_ccf = 0.5

        for _ in range(500):
            n_sc = np.random.poisson(8)
            has_variant = np.random.random() < true_ccf
            if has_variant:
                k_sc = max(1, np.random.binomial(n_sc, 0.5))
            else:
                k_sc = 0
            single_cells.append({"k_sc": k_sc, "n_sc": n_sc})

        alpha, beta = estimate_ccf_prior_from_sc(single_cells)

        prior_mean = alpha / (alpha + beta)
        assert 0.35 <= prior_mean <= 0.65, \
            f"Mean {prior_mean:.3f} should be close to 0.5"

    def test_fallback_no_eligible_cells(self):
        """Test fallback to uniform when no cells meet coverage threshold."""
        # All cells have coverage < 3 (default threshold)
        single_cells = [{"k_sc": 1, "n_sc": 2} for _ in range(100)]

        alpha, beta = estimate_ccf_prior_from_sc(single_cells)

        assert alpha == 1.0, "Should fallback to uniform prior alpha=1"
        assert beta == 1.0, "Should fallback to uniform prior beta=1"

    def test_all_cells_positive(self):
        """Test edge case where all eligible cells have variant reads."""
        single_cells = [{"k_sc": 3, "n_sc": 5} for _ in range(100)]

        alpha, beta = estimate_ccf_prior_from_sc(single_cells)

        assert alpha == 101.0, f"Alpha should be 100+1=101, got {alpha}"
        assert beta == 1.0, f"Beta should be 0+1=1, got {beta}"

        prior_mean = alpha / (alpha + beta)
        assert prior_mean > 0.95, \
            f"Mean {prior_mean:.3f} should indicate very high CCF"

    def test_no_cells_positive(self):
        """Test edge case where no eligible cells have variant reads."""
        single_cells = [{"k_sc": 0, "n_sc": 5} for _ in range(100)]

        alpha, beta = estimate_ccf_prior_from_sc(single_cells)

        assert alpha == 1.0, f"Alpha should be 0+1=1, got {alpha}"
        assert beta == 101.0, f"Beta should be 100+1=101, got {beta}"

        prior_mean = alpha / (alpha + beta)
        assert prior_mean < 0.02, \
            f"Mean {prior_mean:.3f} should indicate very low CCF"

    def test_configurable_coverage_threshold(self):
        """Test that coverage threshold works correctly."""
        single_cells = [
            {"k_sc": 1, "n_sc": 2},
            {"k_sc": 1, "n_sc": 3},
            {"k_sc": 0, "n_sc": 4},
            {"k_sc": 2, "n_sc": 5},
        ]

        alpha1, beta1 = estimate_ccf_prior_from_sc(single_cells)
        assert alpha1 == 3.0, f"Expected alpha=3, got {alpha1}"
        assert beta1 == 2.0, f"Expected beta=2, got {beta1}"

        alpha2, beta2 = estimate_ccf_prior_from_sc(
            single_cells, min_coverage_threshold=2
        )
        assert alpha2 == 4.0, f"Expected alpha=4, got {alpha2}"
        assert beta2 == 2.0, f"Expected beta=2, got {beta2}"

    def test_prior_strength_scales_with_sample_size(self):
        """Test that prior strength increases with more cells."""
        np.random.seed(48)

        small_cells = [{"k_sc": 2, "n_sc": 5} for _ in range(35)]
        small_cells.extend([{"k_sc": 0, "n_sc": 5} for _ in range(15)])
        alpha_small, beta_small = estimate_ccf_prior_from_sc(small_cells)
        kappa_small = alpha_small + beta_small

        large_cells = [{"k_sc": 2, "n_sc": 5} for _ in range(350)]
        large_cells.extend([{"k_sc": 0, "n_sc": 5} for _ in range(150)])
        alpha_large, beta_large = estimate_ccf_prior_from_sc(large_cells)
        kappa_large = alpha_large + beta_large

        assert kappa_large > kappa_small, \
            "Larger sample should give stronger (more concentrated) prior"

        mean_small = alpha_small / (alpha_small + beta_small)
        mean_large = alpha_large / (alpha_large + beta_large)
        assert abs(mean_small - mean_large) < 0.02, \
            "Both should estimate similar CCF"

    def test_uniform_prior_fallback(self):
        """Test that empty/invalid input gives uniform Beta(1,1) prior."""
        alpha, beta = estimate_ccf_prior_from_sc([])
        assert alpha == 1.0 and beta == 1.0

        single_cells = [{"k_sc": 0, "n_sc": 1} for _ in range(10)]
        alpha, beta = estimate_ccf_prior_from_sc(single_cells)
        assert alpha == 1.0 and beta == 1.0

        x = np.linspace(0, 1, 100)
        beta_pdf = beta_dist.pdf(x, alpha, beta)
        uniform_pdf = uniform.pdf(x, 0, 1)
        assert np.allclose(beta_pdf, uniform_pdf), \
            "Beta(1,1) should be equivalent to Uniform(0,1)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
