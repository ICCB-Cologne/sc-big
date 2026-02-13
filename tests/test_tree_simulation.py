"""
Tree simulation tests
=====================

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
import os
import tempfile

import numpy as np
import pytest

from src.tree_simulation import (
    TreeHyperparameters,
    CoalescentSNVSimulator,
    generate_coalescent_tree,
    place_mutations_on_tree,
    export_tree_simulation,
    mutation_filename,
    _get_descendant_leaves,
)
from src.simulation import import_simulation_from_json


class TestKingmanCoalescent:
    """Test coalescent tree generation."""

    def test_correct_node_counts(self):
        """Tree with n leaves should have n-1 internal nodes."""
        n = 20
        rng = np.random.default_rng(42)
        nodes, total_bl = generate_coalescent_tree(n, Ne=1.0, rng=rng)

        n_leaves = sum(1 for nd in nodes if nd.is_leaf)
        n_internal = sum(1 for nd in nodes if not nd.is_leaf)

        assert n_leaves == n
        assert n_internal == n - 1
        assert len(nodes) == 2 * n - 1

    def test_root_has_no_parent(self):
        """Root node should have parent=None."""
        rng = np.random.default_rng(42)
        nodes, _ = generate_coalescent_tree(10, Ne=1.0, rng=rng)

        roots = [nd for nd in nodes if nd.parent is None]
        assert len(roots) == 1
        assert not roots[0].is_leaf

    def test_all_leaves_have_parents(self):
        """Every leaf should have a parent."""
        rng = np.random.default_rng(42)
        nodes, _ = generate_coalescent_tree(15, Ne=1.0, rng=rng)

        for nd in nodes:
            if nd.is_leaf:
                assert nd.parent is not None

    def test_all_branch_lengths_positive(self):
        """All non-root branch lengths should be positive."""
        rng = np.random.default_rng(42)
        nodes, _ = generate_coalescent_tree(10, Ne=1.0, rng=rng)

        for nd in nodes:
            if nd.parent is not None:
                assert nd.branch_length > 0, \
                    f"Node {nd.id} has non-positive branch length"

    def test_tree_is_binary(self):
        """Every internal node should have exactly 2 children."""
        rng = np.random.default_rng(42)
        nodes, _ = generate_coalescent_tree(10, Ne=1.0, rng=rng)

        for nd in nodes:
            if not nd.is_leaf:
                assert len(nd.children) == 2, \
                    f"Internal node {nd.id} has {len(nd.children)} children"

    def test_seed_reproducibility(self):
        """Same seed should produce identical trees."""
        rng1 = np.random.default_rng(123)
        nodes1, bl1 = generate_coalescent_tree(10, Ne=1.0, rng=rng1)

        rng2 = np.random.default_rng(123)
        nodes2, bl2 = generate_coalescent_tree(10, Ne=1.0, rng=rng2)

        assert len(nodes1) == len(nodes2)
        assert abs(bl1 - bl2) < 1e-10
        for n1, n2 in zip(nodes1, nodes2):
            assert n1.id == n2.id
            assert n1.parent == n2.parent
            assert n1.children == n2.children
            assert abs(n1.branch_length - n2.branch_length) < 1e-10

    def test_total_branch_length_positive(self):
        """Total branch length should be positive."""
        rng = np.random.default_rng(42)
        _, total_bl = generate_coalescent_tree(10, Ne=1.0, rng=rng)
        assert total_bl > 0

    def test_two_cells(self):
        """Minimal tree: 2 leaves, 1 internal node."""
        rng = np.random.default_rng(42)
        nodes, _ = generate_coalescent_tree(2, Ne=1.0, rng=rng)

        assert len(nodes) == 3
        assert sum(1 for nd in nodes if nd.is_leaf) == 2
        assert sum(1 for nd in nodes if not nd.is_leaf) == 1

    def test_effective_population_size_scaling(self):
        """Larger Ne should produce longer branch lengths on average."""
        rng1 = np.random.default_rng(42)
        _, bl_small = generate_coalescent_tree(50, Ne=0.1, rng=rng1)

        rng2 = np.random.default_rng(42)
        _, bl_large = generate_coalescent_tree(50, Ne=10.0, rng=rng2)

        assert bl_large > bl_small


class TestDescendantLeaves:
    """Test the _get_descendant_leaves helper."""

    def test_leaf_returns_self(self):
        """A leaf node's descendants should be just itself."""
        rng = np.random.default_rng(42)
        nodes, _ = generate_coalescent_tree(5, Ne=1.0, rng=rng)

        for nd in nodes:
            if nd.is_leaf:
                assert _get_descendant_leaves(nd.id, nodes) == [nd.id]

    def test_root_returns_all_leaves(self):
        """Root's descendants should be all leaves."""
        n = 10
        rng = np.random.default_rng(42)
        nodes, _ = generate_coalescent_tree(n, Ne=1.0, rng=rng)

        root_id = None
        for nd in nodes:
            if nd.parent is None:
                root_id = nd.id
                break

        all_leaves = sorted(
            _get_descendant_leaves(root_id, nodes)
        )
        expected = list(range(n))
        assert all_leaves == expected


class TestMutationPlacement:
    """Test mutation placement on tree."""

    @pytest.fixture
    def tree_and_rng(self):
        """Generate a tree for testing."""
        rng = np.random.default_rng(42)
        nodes, total_bl = generate_coalescent_tree(20, Ne=1.0, rng=rng)
        return nodes, total_bl, rng

    def test_correct_mutation_count(self, tree_and_rng):
        """Should produce exactly n_mutations mutations."""
        nodes, total_bl, rng = tree_and_rng
        n_muts = 5
        mutations = place_mutations_on_tree(
            nodes, total_bl, n_muts, 0.1, 0.1, rng
        )
        assert len(mutations) == n_muts

    def test_valid_branches(self, tree_and_rng):
        """Mutations should be on valid nodes with positive branch length."""
        nodes, total_bl, rng = tree_and_rng
        mutations = place_mutations_on_tree(
            nodes, total_bl, 10, 0.1, 0.1, rng
        )

        for mut in mutations:
            assert 0 <= mut.branch_node_id < len(nodes)
            assert nodes[mut.branch_node_id].branch_length > 0

    def test_carrier_cells_match_descendants(self, tree_and_rng):
        """Carrier cells should be exactly the leaf descendants."""
        nodes, total_bl, rng = tree_and_rng
        mutations = place_mutations_on_tree(
            nodes, total_bl, 10, 0.1, 0.1, rng
        )

        for mut in mutations:
            expected = sorted(
                _get_descendant_leaves(mut.branch_node_id, nodes)
            )
            assert sorted(mut.carrier_cell_ids) == expected

    def test_realized_ccf_correct(self, tree_and_rng):
        """Realized CCF should equal |carriers| / n_leaves."""
        nodes, total_bl, rng = tree_and_rng
        n_leaves = sum(1 for nd in nodes if nd.is_leaf)
        mutations = place_mutations_on_tree(
            nodes, total_bl, 10, 0.1, 0.1, rng
        )

        for mut in mutations:
            expected_ccf = len(mut.carrier_cell_ids) / n_leaves
            assert abs(mut.realized_ccf - expected_ccf) < 1e-10

    def test_copy_number_range(self, tree_and_rng):
        """Copy number should be in {1, 2, 3}."""
        nodes, total_bl, rng = tree_and_rng
        mutations = place_mutations_on_tree(
            nodes, total_bl, 50, 0.1, 0.1, rng
        )

        for mut in mutations:
            assert 1 <= mut.copy_number <= 3
            assert 1 <= mut.multiplicity <= mut.copy_number

    def test_no_amplification_no_deletion(self):
        """With prob=0 for both, all CN should be 2."""
        rng = np.random.default_rng(42)
        nodes, total_bl = generate_coalescent_tree(20, Ne=1.0, rng=rng)
        mutations = place_mutations_on_tree(
            nodes, total_bl, 20, 0.0, 0.0, rng
        )

        for mut in mutations:
            assert mut.copy_number == 2

    def test_all_deletion(self):
        """With deletion_prob=1.0, all CN should be 1."""
        rng = np.random.default_rng(42)
        nodes, total_bl = generate_coalescent_tree(20, Ne=1.0, rng=rng)
        mutations = place_mutations_on_tree(
            nodes, total_bl, 20, 1.0, 0.0, rng
        )

        for mut in mutations:
            assert mut.copy_number == 1
            assert mut.multiplicity == 1

    def test_all_amplification_gives_cn3(self):
        """With amplification_prob=1.0, all CN should be 3."""
        rng = np.random.default_rng(42)
        nodes, total_bl = generate_coalescent_tree(20, Ne=1.0, rng=rng)
        mutations = place_mutations_on_tree(
            nodes, total_bl, 20, 0.0, 1.0, rng
        )

        for mut in mutations:
            assert mut.copy_number == 3

    def test_minimum_ccf_enforced(self):
        """All mutations should have realized CCF >= 0.05."""
        rng = np.random.default_rng(42)
        nodes, total_bl = generate_coalescent_tree(100, Ne=1.0, rng=rng)
        mutations = place_mutations_on_tree(
            nodes, total_bl, 50, 0.1, 0.1, rng
        )

        for mut in mutations:
            assert mut.realized_ccf >= 0.05, \
                f"Mutation {mut.mutation_id} has CCF={mut.realized_ccf}"

    def test_geometric_multiplicity_bias(self):
        """Geometric prior should favor m=1 over higher multiplicities."""
        rng = np.random.default_rng(42)
        nodes, total_bl = generate_coalescent_tree(100, Ne=1.0, rng=rng)
        # Force all CN=3 to test multiplicity distribution.
        mutations = place_mutations_on_tree(
            nodes, total_bl, 200, 0.0, 1.0, rng
        )

        m_counts = {1: 0, 2: 0, 3: 0}
        for mut in mutations:
            m_counts[mut.multiplicity] += 1

        # With geometric prior P(m|C=3) ∝ 2^{-m}:
        # P(m=1) ≈ 4/7, P(m=2) ≈ 2/7, P(m=3) ≈ 1/7.
        assert m_counts[1] > m_counts[2], \
            f"Expected m=1 ({m_counts[1]}) > m=2 ({m_counts[2]})"
        assert m_counts[2] > m_counts[3], \
            f"Expected m=2 ({m_counts[2]}) > m=3 ({m_counts[3]})"


class TestCoalescentSNVSimulator:
    """Test the full CoalescentSNVSimulator."""

    def test_smoke_test(self):
        """Full simulation should run without errors."""
        hp = TreeHyperparameters(
            n_cells=20, n_mutations=3, n_germline_snps=10, seed=42
        )
        sim = CoalescentSNVSimulator(hp)
        nodes, mutations, datasets = sim.simulate()

        assert len(nodes) == 2 * 20 - 1
        assert len(mutations) == 3
        assert len(datasets) == 3

    def test_dataset_format(self):
        """Each dataset should contain BulkData and list[SingleCellData]."""
        hp = TreeHyperparameters(
            n_cells=10, n_mutations=2, n_germline_snps=5, seed=42
        )
        sim = CoalescentSNVSimulator(hp)
        nodes, mutations, datasets = sim.simulate()

        for bulk_data, single_cells in datasets:
            assert bulk_data.k_b >= 0
            assert bulk_data.k_b <= bulk_data.n_b
            assert len(single_cells) == 10

            for sc in single_cells:
                assert sc.k_sc >= 0
                assert sc.k_sc <= sc.n_sc
                assert isinstance(sc.true_variant_present, bool)
                assert isinstance(sc.germline_snps, list)

    def test_carrier_cells_have_variant_present(self):
        """Cells in carrier set should have true_variant_present=True."""
        hp = TreeHyperparameters(
            n_cells=20, n_mutations=3, n_germline_snps=5, seed=42
        )
        sim = CoalescentSNVSimulator(hp)
        nodes, mutations, datasets = sim.simulate()

        for mut, (bulk_data, single_cells) in zip(mutations, datasets):
            carrier_set = set(mut.carrier_cell_ids)
            for i, sc in enumerate(single_cells):
                if i in carrier_set:
                    assert sc.true_variant_present is True
                else:
                    assert sc.true_variant_present is False

    def test_bulk_vaf_plausible(self):
        """Bulk k_b/n_b should be roughly consistent with realized CCF."""
        hp = TreeHyperparameters(
            n_cells=100, n_mutations=5, n_germline_snps=10,
            n_b=1000, kappa=500.0, true_purity=1.0, seed=42,
        )
        sim = CoalescentSNVSimulator(hp)
        nodes, mutations, datasets = sim.simulate()

        for mut, (bulk_data, single_cells) in zip(mutations, datasets):
            observed_vaf = bulk_data.k_b / bulk_data.n_b
            expected = mut.realized_ccf * mut.multiplicity / mut.copy_number
            assert abs(observed_vaf - expected) < 0.3, \
                f"Mutation {mut.mutation_id}: observed={observed_vaf:.3f}, " \
                f"expected={expected:.3f}"

    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        hp = TreeHyperparameters(
            n_cells=15, n_mutations=3, n_germline_snps=5, seed=99
        )
        sim1 = CoalescentSNVSimulator(hp)
        nodes1, muts1, ds1 = sim1.simulate()

        sim2 = CoalescentSNVSimulator(hp)
        nodes2, muts2, ds2 = sim2.simulate()

        assert len(muts1) == len(muts2)
        for m1, m2 in zip(muts1, muts2):
            assert m1.carrier_cell_ids == m2.carrier_cell_ids
            assert m1.copy_number == m2.copy_number

        for (b1, sc1), (b2, sc2) in zip(ds1, ds2):
            assert b1.k_b == b2.k_b
            assert sc1[0].k_sc == sc2[0].k_sc

    def test_per_site_germline_snps(self):
        """Each mutation site should have independent germline SNPs."""
        hp = TreeHyperparameters(
            n_cells=10, n_mutations=3, n_germline_snps=20, seed=42
        )
        sim = CoalescentSNVSimulator(hp)
        _, _, datasets = sim.simulate()

        snps_mut0 = datasets[0][1][0].germline_snps
        snps_mut1 = datasets[1][1][0].germline_snps
        if len(snps_mut0) > 0 and len(snps_mut1) > 0:
            differs = any(
                s0 != s1 for s0, s1 in zip(snps_mut0, snps_mut1)
            )
            assert differs, \
                "Germline SNPs should differ across mutation sites"


class TestExportImportTree:
    """Test export and import of tree simulation data."""

    def test_export_creates_files(self):
        """Export should create per-mutation JSONs and metadata."""
        hp = TreeHyperparameters(
            n_cells=10, n_mutations=3, n_germline_snps=5, seed=42
        )
        sim = CoalescentSNVSimulator(hp)
        nodes, mutations, datasets = sim.simulate()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_tree_simulation(
                nodes, mutations, datasets, hp, tmpdir
            )

            metadata_path = os.path.join(tmpdir, "metadata.json")
            assert os.path.exists(metadata_path)

            with open(metadata_path) as f:
                metadata = json.load(f)
            assert "tree" in metadata
            assert "mutations" in metadata
            assert "hyperparameters" in metadata
            assert "ccf_summary" in metadata
            assert len(metadata["mutations"]) == 3

            for mut, (bulk_data, _) in zip(mutations, datasets):
                fname = mutation_filename(
                    mut.mutation_id, mut.copy_number, mut.multiplicity,
                    hp.true_purity, mut.realized_ccf,
                    bulk_data.true_expected_vaf,
                )
                mut_path = os.path.join(tmpdir, fname)
                assert os.path.exists(mut_path)

    def test_per_mutation_json_loadable(self):
        """Per-mutation JSONs should be loadable."""
        hp = TreeHyperparameters(
            n_cells=10, n_mutations=2, n_germline_snps=5, seed=42
        )
        sim = CoalescentSNVSimulator(hp)
        nodes, mutations, datasets = sim.simulate()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_tree_simulation(
                nodes, mutations, datasets, hp, tmpdir
            )

            for mut, (orig_bulk, _) in zip(mutations, datasets):
                fname = mutation_filename(
                    mut.mutation_id, mut.copy_number, mut.multiplicity,
                    hp.true_purity, mut.realized_ccf,
                    orig_bulk.true_expected_vaf,
                )
                mut_path = os.path.join(tmpdir, fname)
                bulk_data, single_cells, hyperparams = \
                    import_simulation_from_json(mut_path)

                assert bulk_data.k_b >= 0
                assert len(single_cells) == 10
                assert hyperparams.true_copy_number == mut.copy_number
                assert hyperparams.true_multiplicity == mut.multiplicity
                assert abs(hyperparams.true_ccf - mut.realized_ccf) < 1e-10

    def test_metadata_structure(self):
        """Metadata JSON should have expected structure."""
        hp = TreeHyperparameters(
            n_cells=10, n_mutations=2, n_germline_snps=5, seed=42
        )
        sim = CoalescentSNVSimulator(hp)
        nodes, mutations, datasets = sim.simulate()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_tree_simulation(
                nodes, mutations, datasets, hp, tmpdir
            )

            with open(os.path.join(tmpdir, "metadata.json")) as f:
                metadata = json.load(f)

            tree = metadata["tree"]
            assert tree["n_cells"] == 10
            assert tree["n_nodes"] == len(nodes)
            assert len(tree["nodes"]) == len(nodes)

            for mut_meta in metadata["mutations"]:
                assert "mutation_id" in mut_meta
                assert "branch_node_id" in mut_meta
                assert "copy_number" in mut_meta
                assert "multiplicity" in mut_meta
                assert "carrier_cell_ids" in mut_meta
                assert "realized_ccf" in mut_meta

            ccf = metadata["ccf_summary"]
            assert "mean" in ccf
            assert "min" in ccf
            assert "max" in ccf
            assert ccf["min"] <= ccf["mean"] <= ccf["max"]

    def test_round_trip_data_integrity(self):
        """Data should survive export/import without corruption."""
        hp = TreeHyperparameters(
            n_cells=15, n_mutations=2, n_germline_snps=10, seed=42
        )
        sim = CoalescentSNVSimulator(hp)
        nodes, mutations, datasets = sim.simulate()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_tree_simulation(
                nodes, mutations, datasets, hp, tmpdir
            )

            for mut, (orig_bulk, orig_sc) in zip(mutations, datasets):
                fname = mutation_filename(
                    mut.mutation_id, mut.copy_number, mut.multiplicity,
                    hp.true_purity, mut.realized_ccf,
                    orig_bulk.true_expected_vaf,
                )
                mut_path = os.path.join(tmpdir, fname)
                loaded_bulk, loaded_sc, _ = import_simulation_from_json(
                    mut_path
                )

                assert loaded_bulk.k_b == orig_bulk.k_b
                assert loaded_bulk.n_b == orig_bulk.n_b
                assert len(loaded_sc) == len(orig_sc)

                for orig, loaded in zip(orig_sc, loaded_sc):
                    assert loaded.k_sc == orig.k_sc
                    assert loaded.n_sc == orig.n_sc
                    assert loaded.true_variant_present == \
                        orig.true_variant_present
