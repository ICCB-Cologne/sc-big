"""
Comparison figure for SC-BIG vs ProSolo.

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
import numpy as np
import matplotlib.pyplot as plt

from src.logging_config import get_logger

logger = get_logger(__name__)

_CCF_BINS = [
    (0.0, 0.3, "Low CCF (0, 0.3]"),
    (0.3, 0.7, "Mid CCF (0.3, 0.7]"),
    (0.7, 1.0, "High CCF (0.7, 1.0]"),
]

_PANEL_LETTERS = [
    ("a", "b", "c"),
    ("d", "e", "f"),
    ("g", "h", "i"),
    ("j", "k", "l"),
]

_MIN_CELLS = 10


def _roc_curve(probs, labels):
    """Compute (FPR, TPR) pairs sweeping the threshold from 1 to 0."""
    thresholds = np.linspace(1.01, -0.01, 200)
    fprs, tprs = [], []
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    for t in thresholds:
        tp = sum(1 for p, y in zip(probs, labels) if p >= t and y)
        fp = sum(1 for p, y in zip(probs, labels) if p >= t and not y)
        tprs.append(tp / n_pos if n_pos > 0 else 0.0)
        fprs.append(fp / n_neg if n_neg > 0 else 0.0)

    return np.array(fprs), np.array(tprs)


def _f1_curve(probs, labels):
    """Compute F1 score at each threshold from 0 to 1."""
    thresholds = np.linspace(0, 1, 101)
    f1s = []
    for t in thresholds:
        tp = sum(1 for p, y in zip(probs, labels) if p >= t and y)
        fp = sum(1 for p, y in zip(probs, labels) if p >= t and not y)
        fn = sum(1 for p, y in zip(probs, labels) if p < t and y)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return thresholds, np.array(f1s)


def _plot_comparison_row(axes, scbig_results, prosolo_results, panel_letters):
    """
    Plot one row of the comparison figure (ROC, F1, calibration).

    Parameters
    ----------
    axes : array of 3 Axes
    scbig_results : list[dict]
    prosolo_results : list[dict]
    panel_letters : tuple of 3 str
        E.g. ("a", "b", "c").
    """
    scbig_probs = [r["posterior_prob"] for r in scbig_results]
    scbig_labels = [r["true_variant_present"] for r in scbig_results]
    pro_probs = [r["posterior_prob"] for r in prosolo_results]
    pro_labels = [r["true_variant_present"] for r in prosolo_results]
    naive_probs = [r["k_sc"] / r["n_sc"] if r["n_sc"] > 0 else 0.0
                   for r in scbig_results]
    naive_labels = scbig_labels

    # 1. ROC curve.
    ax = axes[0]
    ax.text(
        -0.15, 1.05, panel_letters[0], transform=ax.transAxes, fontsize=16,
        fontweight="bold", va="top",
    )

    s_fpr, s_tpr = _roc_curve(scbig_probs, scbig_labels)
    p_fpr, p_tpr = _roc_curve(pro_probs, pro_labels)
    n_fpr, n_tpr = _roc_curve(naive_probs, naive_labels)
    s_auc = np.trapezoid(s_tpr, s_fpr)
    p_auc = np.trapezoid(p_tpr, p_fpr)
    n_auc = np.trapezoid(n_tpr, n_fpr)

    ax.plot(s_fpr, s_tpr, "g-", linewidth=2,
            label=f"SC-BIG (AUC={s_auc:.3f})")
    ax.plot(p_fpr, p_tpr, "b--", linewidth=2,
            label=f"ProSolo (AUC={p_auc:.3f})")
    ax.plot(n_fpr, n_tpr, "r-.", linewidth=2,
            label=f"Naive VAF (AUC={n_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, alpha=0.5,
            label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # 2. F1 score vs threshold.
    ax = axes[1]
    ax.text(
        -0.15, 1.05, panel_letters[1], transform=ax.transAxes, fontsize=16,
        fontweight="bold", va="top",
    )

    s_thresh, s_f1 = _f1_curve(scbig_probs, scbig_labels)
    p_thresh, p_f1 = _f1_curve(pro_probs, pro_labels)
    n_thresh, n_f1 = _f1_curve(naive_probs, naive_labels)
    s_best_idx = np.argmax(s_f1)
    p_best_idx = np.argmax(p_f1)
    n_best_idx = np.argmax(n_f1)

    ax.plot(s_thresh, s_f1, "g-", linewidth=2,
            label=(f"SC-BIG (max F1={s_f1[s_best_idx]:.3f} "
                   f"at t={s_thresh[s_best_idx]:.2f})"))
    ax.plot(p_thresh, p_f1, "b--", linewidth=2,
            label=(f"ProSolo (max F1={p_f1[p_best_idx]:.3f} "
                   f"at t={p_thresh[p_best_idx]:.2f})"))
    ax.plot(n_thresh, n_f1, "r-.", linewidth=2,
            label=(f"Naive VAF (max F1={n_f1[n_best_idx]:.3f} "
                   f"at t={n_thresh[n_best_idx]:.2f})"))
    ax.scatter([s_thresh[s_best_idx]], [s_f1[s_best_idx]],
               color="green", s=100, zorder=5)
    ax.scatter([p_thresh[p_best_idx]], [p_f1[p_best_idx]],
               color="blue", s=100, zorder=5)
    ax.scatter([n_thresh[n_best_idx]], [n_f1[n_best_idx]],
               color="red", s=100, zorder=5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs Threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Calibration plot.
    ax = axes[2]
    ax.text(
        -0.15, 1.05, panel_letters[2], transform=ax.transAxes, fontsize=16,
        fontweight="bold", va="top",
    )

    bins = np.linspace(0, 1, 11)

    for name, probs, labels, color, marker in [
        ("SC-BIG", scbig_probs, scbig_labels, "green", "o"),
        ("ProSolo", pro_probs, pro_labels, "blue", "s"),
        ("Naive VAF", naive_probs, naive_labels, "red", "D"),
    ]:
        bin_means, bin_obs = [], []
        for i in range(len(bins) - 1):
            in_bin = [
                (p, y) for p, y in zip(probs, labels)
                if bins[i] <= p < bins[i + 1]
            ]
            if in_bin:
                bin_means.append(np.mean([p for p, _ in in_bin]))
                bin_obs.append(np.mean([y for _, y in in_bin]))

        if len(bin_means) >= 2:
            coeffs = np.polyfit(bin_means, bin_obs, 1)
            a, b = coeffs
            ss_res = sum(
                (y - (a * x + b)) ** 2
                for x, y in zip(bin_means, bin_obs)
            )
            ss_tot = sum(
                (y - np.mean(bin_obs)) ** 2 for y in bin_obs
            )
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            fit_x = np.linspace(0, 1, 100)
            ax.plot(fit_x, a * fit_x + b, color=color, linewidth=2.5,
                    alpha=0.5)
            label = f"{name} (y={a:.2f}x+{b:.2f}, R\u00b2={r2:.3f})"
        else:
            label = name

        ax.scatter(bin_means, bin_obs, color=color, s=56,
                   marker=marker, label=label, zorder=5, alpha=0.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Plot")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def _filter_by_ccf(results, lo, hi, first_bin):
    """Return results with CCF in (lo, hi]; first bin is [lo, hi]."""
    if first_bin:
        return [r for r in results if lo <= r.get("ccf", -1) <= hi]
    return [r for r in results if lo < r.get("ccf", -1) <= hi]


def plot_comparison(scbig_results, prosolo_results, output_file,
                    stratified_output_file=None):
    """
    Produce comparison figures: overall and CCF-stratified.

    Parameters
    ----------
    scbig_results : list[dict]
        Pooled SC-BIG results, each with ``posterior_prob``,
        ``true_variant_present``, and ``ccf``.
    prosolo_results : list[dict]
        Pooled ProSolo results, same keys.
    output_file : str
        Path for overall comparison PNG.
    stratified_output_file : str or None
        Path for CCF-stratified comparison PNG.  If None, derived
        from output_file by appending ``_stratified`` before the
        extension.
    """
    # Figure 1: overall comparison (single row).
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), squeeze=False)
    _plot_comparison_row(
        axes[0], scbig_results, prosolo_results, _PANEL_LETTERS[0],
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Overall comparison plot saved to {output_file}")

    # Figure 2: CCF-stratified comparison.
    strat_rows = []
    for idx, (lo, hi, label) in enumerate(_CCF_BINS):
        first_bin = (idx == 0)
        s_bin = _filter_by_ccf(scbig_results, lo, hi, first_bin)
        p_bin = _filter_by_ccf(prosolo_results, lo, hi, first_bin)
        if len(s_bin) >= _MIN_CELLS and len(p_bin) >= _MIN_CELLS:
            strat_rows.append((label, s_bin, p_bin))
        else:
            logger.info(
                f"Skipping {label}: SC-BIG n={len(s_bin)}, "
                f"ProSolo n={len(p_bin)} (min {_MIN_CELLS})"
            )

    if not strat_rows:
        logger.warning("No CCF bins with enough data for "
                       "stratified plot")
        return

    if stratified_output_file is None:
        base, ext = output_file.rsplit(".", 1)
        stratified_output_file = f"{base}_stratified.{ext}"

    n_rows = len(strat_rows)
    fig, all_axes = plt.subplots(
        n_rows, 3, figsize=(18, 5 * n_rows), squeeze=False,
    )
    for row_idx, (label, s_res, p_res) in enumerate(strat_rows):
        _plot_comparison_row(
            all_axes[row_idx], s_res, p_res, _PANEL_LETTERS[row_idx],
        )

    plt.tight_layout(h_pad=4.0)

    for row_idx, (label, _, _) in enumerate(strat_rows):
        mid_ax = all_axes[row_idx][1]
        mid_ax.text(
            0.5, 1.25, label, transform=mid_ax.transAxes,
            ha="center", va="bottom", fontsize=13, fontweight="bold",
        )
    plt.savefig(stratified_output_file, dpi=300,
                bbox_inches="tight")
    plt.close()
    logger.info(f"Stratified comparison plot saved to "
                f"{stratified_output_file}")
