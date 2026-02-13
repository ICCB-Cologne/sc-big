"""
Simulation performance tests
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
import time

from src.simulation import BayesianSNVSimulator, Hyperparameters


class TestSimulationPerformance:
    """Test simulation performance characteristics."""

    @pytest.mark.parametrize("n_cells", [10, 100, 1000])
    def test_simulation_speed(self, n_cells):
        """Test simulation completes in reasonable time."""
        simulator = BayesianSNVSimulator(seed=42)

        start = time.time()
        bulk_data, single_cells = simulator.simulate_dataset(n_cells)
        duration = time.time() - start

        assert len(single_cells) == n_cells

        if n_cells == 10:
            assert duration < 0.1
        elif n_cells == 100:
            assert duration < 0.5
        elif n_cells == 1000:
            assert duration < 5.0

    @pytest.mark.parametrize("n_germline_snps", [0, 10, 50, 100, 1000])
    def test_germline_snp_performance(self, n_germline_snps):
        """Test performance with varying numbers of germline SNPs."""
        hp = Hyperparameters(n_germline_snps=n_germline_snps)
        simulator = BayesianSNVSimulator(hyperparams=hp, seed=42)

        start = time.time()
        bulk_data, single_cells = simulator.simulate_dataset(n_single_cells=50)
        duration = time.time() - start

        assert duration < 2.0

        for sc in single_cells:
            assert len(sc.germline_snps) <= n_germline_snps

    def test_bulk_simulation_performance(self):
        """Test bulk data simulation performance."""
        simulator = BayesianSNVSimulator(seed=42)

        start = time.time()
        for _ in range(1000):
            simulator._simulate_bulk_data()
        duration = time.time() - start

        assert duration < 1.0

    def test_single_cell_simulation_performance(self):
        """Test single cell simulation performance."""
        simulator = BayesianSNVSimulator(seed=42)
        bulk_data = simulator._simulate_bulk_data()

        start = time.time()
        for _ in range(1000):
            simulator._simulate_single_cell_data(bulk_data)
        duration = time.time() - start

        assert duration < 2.0

    @pytest.mark.parametrize("n_cells", [10, 100, 1000, 10000])
    def test_dataset_simulation_speed(self, n_cells):
        """Test complete dataset simulation speed."""
        simulator = BayesianSNVSimulator(seed=42)

        start = time.time()
        bulk_data, single_cells = simulator.simulate_dataset(n_cells)
        duration = time.time() - start

        assert len(single_cells) == n_cells

        if n_cells == 10:
            assert duration < 0.1
        elif n_cells == 100:
            assert duration < 0.5
        elif n_cells == 1000:
            assert duration < 5.0
