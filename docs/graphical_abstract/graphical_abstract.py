#!/usr/bin/env python3
"""
Generate individual PNG panels for the SC-BIG graphical abstract.

Usage:
    python3 scripts/graphical_abstract.py -o <out_dir>
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.stats import beta, betabinom

PALETTE = {
    "ref": "#a8d5e2",        # soft blue
    "alt": "#e88d67",        # warm orange
    "nocov": "#e0e0e0",      # light grey
    "prior": "#7eb5a6",      # sage green
    "posterior": "#c3729d",  # muted pink
    "scbig": "#5b8fa8",      # teal-blue
    "other": "#d4a373",      # sandy brown
    "grid_pos": "#d94f4f",   # red for variant-present cells
    "grid_neg": "#4f86d9",   # blue for variant-absent cells
    "bg": "#ffffff",         # white
    # "bg": "#fafaf8",       # warm off-white
}

PROB_CMAP = LinearSegmentedColormap.from_list(
    "prob", [PALETTE["grid_neg"], "#f5f5f0", PALETTE["grid_pos"]])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": PALETTE["bg"],
    "figure.facecolor": PALETTE["bg"],
    "savefig.facecolor": PALETTE["bg"],
})

np.random.seed(42)
N_CELLS = 10
N_VARIANTS = 6
VARIANT_POS = [2, 5, 8]


def _legend_below(ax, fig, ncol=2, fontsize=9):
    """Place legend horizontally below the plot without overlapping."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=fontsize,
               loc="lower center", ncol=ncol)


def panel_bulk_pileup(out):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    n_positions = 11
    positions = np.arange(n_positions)
    ref_counts = np.array([230, 245, 210, 250, 235, 170, 225, 240, 205, 220,
                           238])
    alt_counts = np.zeros(n_positions, dtype=int)
    ref_counts[VARIANT_POS[0]] = 240 - 80
    alt_counts[VARIANT_POS[0]] = 80
    ref_counts[VARIANT_POS[1]] = 260 - 140
    alt_counts[VARIANT_POS[1]] = 140
    ref_counts[VARIANT_POS[2]] = 220 - 15
    alt_counts[VARIANT_POS[2]] = 15

    ax.bar(positions, ref_counts, color=PALETTE["ref"], edgecolor="white",
           linewidth=0.8, label="Reference", width=0.8)
    ax.bar(positions, alt_counts, bottom=ref_counts, color=PALETTE["alt"],
           edgecolor="white", linewidth=0.8, label="Variant", width=0.8)
    ax.set_xlabel("Genomic position", fontsize=12)
    ax.set_ylabel("# Reads", fontsize=12)
    ax.set_ylim(0, 250)
    ax.set_xticks([])
    fig.subplots_adjust(bottom=0.20)
    _legend_below(ax, fig, ncol=2)
    fig.savefig(os.path.join(out, "bulk_pileup.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def panel_sc_pileup(out):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    n_positions = 11
    positions = np.arange(n_positions)
    ref_counts = np.array([2, 1, 3, 0, 3, 1, 1, 2, 0, 1, 2])
    alt_counts = np.zeros(n_positions, dtype=int)
    ref_counts[VARIANT_POS[0]] = 4 - 1
    alt_counts[VARIANT_POS[0]] = 1
    ref_counts[VARIANT_POS[1]] = 2 - 1
    alt_counts[VARIANT_POS[1]] = 1
    ref_counts[VARIANT_POS[2]] = 0
    alt_counts[VARIANT_POS[2]] = 0

    ax.bar(positions, ref_counts, color=PALETTE["ref"], edgecolor="white",
           linewidth=0.8, label="Reference", width=0.8)
    ax.bar(positions, alt_counts, bottom=ref_counts, color=PALETTE["alt"],
           edgecolor="white", linewidth=0.8, label="Variant", width=0.8)
    ax.text(VARIANT_POS[2], 0.3, "?", ha="center", va="bottom",
            fontsize=20, fontweight="bold", color=PALETTE["alt"])
    ax.set_xlabel("Genomic position", fontsize=12)
    ax.set_ylabel("# Reads", fontsize=12)
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 2, 4])
    ax.set_xticks([])
    fig.subplots_adjust(bottom=0.22)
    _legend_below(ax, fig, ncol=2)
    fig.savefig(os.path.join(out, "sc_pileup.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def _make_sc_vaf_matrix():
    """Generate a sparse VAF matrix: most entries NaN (no coverage)."""
    vaf = np.full((N_CELLS, N_VARIANTS), np.nan)
    for j in range(N_VARIANTS):
        covered = np.random.choice(
            N_CELLS, size=int(N_CELLS * 0.3), replace=False
        )
        for i in covered:
            if np.random.rand() < 0.35:
                vaf[i, j] = np.random.uniform(0.3, 1.0)
            else:
                vaf[i, j] = np.random.uniform(0.0, 0.1)
    return vaf


def panel_sc_matrix(out):
    vaf = _make_sc_vaf_matrix()
    fig, ax = plt.subplots(figsize=(3.0, 2.4))

    grey_bg = np.zeros((N_CELLS, N_VARIANTS))
    grey_cmap = ListedColormap([PALETTE["nocov"]])
    ax.imshow(grey_bg, cmap=grey_cmap, aspect="auto", vmin=0, vmax=1)
    im = ax.imshow(vaf, cmap=PROB_CMAP, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Variants", fontsize=12)
    ax.set_ylabel("Single Cells", fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, N_VARIANTS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_CELLS, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", size=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.25, pad=0.05, shrink=0.8)
    cbar.set_label("VAF", fontsize=11)
    cbar.set_ticks([0, 0.5, 1])

    grey_patch = mpatches.Patch(
        facecolor=PALETTE["nocov"], edgecolor="#bbb", label="No cov."
    )
    cbar.ax.legend(handles=[grey_patch], frameon=False, fontsize=11,
                   loc="upper center", bbox_to_anchor=(0.5, -0.12),
                   handlelength=1.6)
    fig.savefig(os.path.join(out, "sc_sparse_matrix.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def panel_germline_vafs(out):
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    tau = 5.0
    n_snps = 4000
    coverages = np.random.poisson(lam=3, size=n_snps)
    coverages = np.clip(coverages, 1, None)
    alt_reads = np.array([
        betabinom.rvs(n, tau / 2, tau / 2) for n in coverages
    ])
    vafs = alt_reads / coverages
    n_bins = 30
    bin_edges = np.linspace(0, 1, n_bins + 1)
    sns.histplot(vafs, bins=bin_edges, color=PALETTE["prior"],
                 edgecolor="white", linewidth=0.6, ax=ax, stat="density",
                 alpha=0.85)
    expected = np.zeros(n_bins)
    for n in coverages:
        ks = np.arange(0, n + 1)
        pmf = betabinom.pmf(ks, n, tau / 2, tau / 2)
        vaf_k = ks / n
        bin_idx = np.clip(np.digitize(vaf_k, bin_edges) - 1, 0, n_bins - 1)
        for ki, bi in enumerate(bin_idx):
            expected[bi] += pmf[ki]
    bin_width = bin_edges[1] - bin_edges[0]
    expected_density = expected / (n_snps * bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.step(bin_centers, expected_density, where="mid", color="#3d6b5e",
            linewidth=2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("hSNP VAF", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.annotate(r"$\hat{\tau}$ estimation", xy=(0.5, 0.92),
                xycoords="axes fraction", ha="center", fontsize=11,
                fontstyle="italic", color="#3d6b5e")
    ax.annotate(r"BetaBinom$(n,\,\frac{\tau}{2},\,\frac{\tau}{2})$",
                xy=(0.5, 0.82), xycoords="axes fraction", ha="center",
                fontsize=10, color="#3d6b5e")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "germline_vafs.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def panel_ccf_prior(out):
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    a, b = 8, 12
    xs = np.linspace(0, 1, 200)
    ax.fill_between(xs, beta.pdf(xs, a, b), color=PALETTE["prior"],
                    alpha=0.5)
    ax.plot(xs, beta.pdf(xs, a, b), color="#3d6b5e", linewidth=2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Empirical CCF Prior", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ccf_prior.png"), dpi=300)
    plt.close(fig)


def panel_ccf_posterior(out):
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    a_prior, b_prior = 8, 12
    a_post, b_post = 22, 10
    xs = np.linspace(0, 1, 200)
    ax.fill_between(xs, beta.pdf(xs, a_prior, b_prior),
                    color=PALETTE["prior"], alpha=0.3, label="Prior")
    ax.plot(xs, beta.pdf(xs, a_prior, b_prior), color="#3d6b5e",
            linewidth=1.5, linestyle="--")
    ax.fill_between(xs, beta.pdf(xs, a_post, b_post),
                    color=PALETTE["posterior"], alpha=0.5, label="Posterior")
    ax.plot(xs, beta.pdf(xs, a_post, b_post), color="#8b3a62",
            linewidth=2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("CCF Posterior", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, 1)
    fig.savefig(os.path.join(out, "ccf_posterior.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def panel_cell_posteriors(out):
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    posteriors = np.zeros((N_CELLS, N_VARIANTS))
    for j in range(N_VARIANTS):
        for i in range(N_CELLS):
            if np.random.rand() < 0.35:
                posteriors[i, j] = np.random.uniform(0.65, 0.98)
            else:
                posteriors[i, j] = np.random.uniform(0.02, 0.20)

    im = ax.imshow(posteriors, cmap=PROB_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Variants", fontsize=12)
    ax.set_ylabel("Single Cells", fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, N_VARIANTS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_CELLS, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", size=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.25, pad=0.05, shrink=0.80)
    cbar.set_label("P(var. present)", fontsize=11)
    cbar.set_ticks([0, 0.5, 1])
    fig.savefig(os.path.join(out, "cell_posteriors.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def panel_roc(out):
    fig, ax = plt.subplots(figsize=(3.0, 3.6))
    fpr = np.linspace(0, 1, 200)
    tpr_scbig = 1 - (1 - fpr) ** 8.0
    tpr_other = 1 - (1 - fpr) ** 3.0
    ax.plot(fpr, tpr_scbig, color=PALETTE["scbig"], linewidth=2.5,
            label="SC-BIG")
    ax.plot(fpr, tpr_other, color=PALETTE["other"], linewidth=2.5,
            label="Other")
    ax.plot([0, 1], [0, 1], color="#bbb", linewidth=1, linestyle="--")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("FPR", fontsize=12)
    ax.set_ylabel("TPR", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    fig.subplots_adjust(bottom=0.12)
    _legend_below(ax, fig, ncol=2)
    fig.savefig(os.path.join(out, "roc_curve.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate graphical abstract panels as PNGs."
    )
    parser.add_argument(
        "-o", "--output-dir", default="panels",
        help="Directory for output PNGs (default: panels/)"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    panel_bulk_pileup(args.output_dir)
    panel_sc_pileup(args.output_dir)
    panel_sc_matrix(args.output_dir)
    panel_germline_vafs(args.output_dir)
    panel_ccf_prior(args.output_dir)
    panel_ccf_posterior(args.output_dir)
    panel_cell_posteriors(args.output_dir)
    panel_roc(args.output_dir)

    print(f"All panels saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
