"""
ProSolo Lodato Model and Native Binary Wrapper
===============================================

This module provides:

1. **Lodato amplification bias model** — the coverage-dependent beta-binomial
   MDA model from Lodato et al. (2015, Science), used for simulation and for
   SC-BIG's optional ``lodato`` error model.  Parameters are taken from the
   ``libprosic`` v0.7.3 Rust source.

2. **Native ProSolo binary wrapper** — generates synthetic BAM files from
   read-count data and invokes the ``prosolo`` (Laehnemann et al. 2021) binary
   to obtain variant-call posteriors.

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
import os
import pysam
import subprocess
import shutil
import tempfile

import numpy as np

from multiprocessing import Pool
from scipy.special import betaln, gammaln, logsumexp
from typing import Final
from src.logging_config import get_logger

logger = get_logger(__name__)

_ALPHA_SLOPE: Final[float] = -0.000027183
_ALPHA_INTERCEPT: Final[float] = 0.068567471
_BETA_SLOPE: Final[float] = 0.007454388
_BETA_INTERCEPT: Final[float] = 2.367486659
_W_SLOPE: Final[float] = 0.000548761
_W_INTERCEPT: Final[float] = 0.540396786
_ALPHA1_SLOPE: Final[float] = 0.057378844
_ALPHA1_INTERCEPT: Final[float] = 0.669733191
_ALPHA2_SLOPE: Final[float] = 0.003233912
_ALPHA2_INTERCEPT: Final[float] = 0.399261625

SC_COVERAGE_CAP: Final[int] = 100             # from ProSolo
BASE_ERROR_RATE_DEFAULT: Final[float] = 0.01  # is PHRED = 20

# ProSolo event names whose posterior probabilities sum to P(alt present). See
# ProSolo paper, Methods eq. 9.
_ALT_PRESENT_EVENTS: Final[tuple[str, ...]] = (
    "PROB_ADO_TO_REF", "PROB_ERR_REF", "PROB_HET",
    "PROB_ADO_TO_ALT", "PROB_HOM_ALT",
)

_CONTIG_LEN: Final[int] = 10000
_VAR_POS_0BASED: Final[int] = 4999
_READ_LEN: Final[int] = 100
_READ_START: Final[int] = _VAR_POS_0BASED - 49


def _ln_beta_binom(k: int, n: int, a: float, b: float) -> float:
    """
    Log PMF of BetaBinomial(k; n, a, b).

    BetaBin(k; n, a, b) = C(n,k) · B(k+a, n−k+b) / B(a, b)
    """
    ln_binom = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    ln_beta_ratio = betaln(k + a, n - k + b) - betaln(a, b)
    return float(ln_binom + ln_beta_ratio)


def prob_rho(af: float, n: int, k: int) -> float:
    """
    Lodato amplification bias: ln[P(k/n | af)].

    Parameters
    ----------
    af : {0.0, 0.5, 1.0}
        True underlying allele frequency (diploid).
    n : int
        Read count (coverage), must already be capped at SC_COVERAGE_CAP.
    k : int
        Alt read count.

    Returns
    -------
    float
        Log probability.
    """
    if n == 0:
        return 0.0 if k == 0 else -np.inf

    if af == 0.0:
        a = max(_ALPHA_SLOPE * n + _ALPHA_INTERCEPT, 1e-10)
        b = max(_BETA_SLOPE * n + _BETA_INTERCEPT, 1e-10)
        return _ln_beta_binom(k, n, a, b)

    if af == 1.0:
        a = max(_BETA_SLOPE * n + _BETA_INTERCEPT, 1e-10)
        b = max(_ALPHA_SLOPE * n + _ALPHA_INTERCEPT, 1e-10)
        return _ln_beta_binom(k, n, a, b)

    if af == 0.5:
        w = np.clip(_W_SLOPE * n + _W_INTERCEPT, 1e-10, 1.0 - 1e-10)
        a1 = max(_ALPHA1_SLOPE * n + _ALPHA1_INTERCEPT, 1e-10)
        a2 = max(_ALPHA2_SLOPE * n + _ALPHA2_INTERCEPT, 1e-10)
        ln_c1 = np.log(w) + _ln_beta_binom(k, n, a1, a1)
        ln_c2 = np.log(1.0 - w) + _ln_beta_binom(k, n, a2, a2)
        return float(logsumexp([ln_c1, ln_c2]))

    raise ValueError(
        f"ProSolo only supports diploid AF in {{0, 0.5, 1}}, got {af}"
    )


def lodato_af(m: int, C: int) -> float:
    """
    Map multiplicity/copy-number to nearest Lodato AF in {0, 0.5, 1}.

    The Lodato model only supports diploid allele frequencies.  For arbitrary
    m/C we snap to the nearest supported value.

    Parameters
    ----------
    m : int
        Multiplicity (variant copies per cell).
    C : int
        Total copy number at the locus.

    Returns
    -------
    float
        One of 0.0, 0.5, or 1.0.
    """
    vaf = m / C
    if vaf <= 0.25:
        return 0.0
    elif vaf <= 0.75:
        return 0.5
    else:
        return 1.0


def sample_lodato(af: float, n: int, rng: np.random.Generator) -> int:
    """
    Sample k alt reads from the Lodato amplification-bias distribution.

    Uses direct beta-binomial sampling (much faster than PMF enumeration).

    Parameters
    ----------
    af : {0.0, 0.5, 1.0}
        True allele frequency (diploid).
    n : int
        Total read count (coverage).
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    int
        Number of alt reads.
    """
    if n == 0:
        return 0

    if af == 0.0:
        a = max(_ALPHA_SLOPE * n + _ALPHA_INTERCEPT, 1e-10)
        b = max(_BETA_SLOPE * n + _BETA_INTERCEPT, 1e-10)
        p = rng.beta(a, b)
        return int(rng.binomial(n, p))

    if af == 1.0:
        a = max(_BETA_SLOPE * n + _BETA_INTERCEPT, 1e-10)
        b = max(_ALPHA_SLOPE * n + _ALPHA_INTERCEPT, 1e-10)
        p = rng.beta(a, b)
        return int(rng.binomial(n, p))

    if af == 0.5:
        w = np.clip(_W_SLOPE * n + _W_INTERCEPT, 1e-10, 1.0 - 1e-10)
        a1 = max(_ALPHA1_SLOPE * n + _ALPHA1_INTERCEPT, 1e-10)
        a2 = max(_ALPHA2_SLOPE * n + _ALPHA2_INTERCEPT, 1e-10)
        if rng.random() < w:
            p = rng.beta(a1, a1)
        else:
            p = rng.beta(a2, a2)
        return int(rng.binomial(n, p))

    raise ValueError(
        f"sample_lodato only supports af in {{0, 0.5, 1}}, "
        f"got {af}"
    )


def ln_count_likelihood(k_obs: int, n_obs: int, af: float, e: float) -> float:
    """
    Log likelihood of k_obs alt reads out of n_obs total, given allele
    frequency af and base error rate e.

    Per-read model (ProSolo, uniform quality):
        P(alt read | af) = af·(1−e) + (1−af)·e/3
        P(ref read | af) = (1−af)·(1−e) + af·e/3

    Product over independent reads with count data:
        L(af) = P(alt|af)^k · P(ref|af)^(n−k)
    """
    if n_obs == 0:
        return 0.0
    p_alt = af * (1.0 - e) + (1.0 - af) * e / 3.0
    p_ref = (1.0 - af) * (1.0 - e) + af * e / 3.0
    p_alt = max(p_alt, 1e-300)
    p_ref = max(p_ref, 1e-300)
    return float(k_obs * np.log(p_alt) + (n_obs - k_obs) * np.log(p_ref))


def _write_reference(path: str) -> None:
    """Write a minimal reference FASTA (one contig, all A's) and index it."""
    ref_seq = "A" * _CONTIG_LEN
    with open(path, "w") as f:
        f.write(">chr1\n")
        for i in range(0, _CONTIG_LEN, 80):
            f.write(ref_seq[i:i + 80] + "\n")
    pysam.faidx(path)


def _write_candidate_vcf(vcf_path: str) -> None:
    """Write a single-site candidate VCF (bgzipped + tabix-indexed)."""
    with open(vcf_path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##contig=<ID=chr1,length={_CONTIG_LEN}>\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        f.write(f"chr1\t{_VAR_POS_0BASED + 1}\t.\tA\tT\t.\t.\t.\n")
    gz_path = vcf_path + ".gz"
    pysam.tabix_compress(vcf_path, gz_path, force=True)
    pysam.tabix_index(gz_path, preset="vcf", force=True)


def _write_bam(
    path: str, sample: str, rg_id: str, k_alt: int,
    n_total: int, base_qual: int
) -> None:
    """Write a BAM with k_alt alt reads and (n_total - k_alt) ref reads."""
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": _CONTIG_LEN}],
        "RG": [{"ID": rg_id, "SM": sample}],
    })
    unsorted = path.replace(".bam", "_unsorted.bam")
    qual_char = chr(base_qual + 33)
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for i in range(n_total):
            a = pysam.AlignedSegment(out.header)
            a.query_name = f"{sample}_read_{i}"
            seq = list("A" * _READ_LEN)
            if i < k_alt:
                seq[49] = "T"
            a.query_sequence = "".join(seq)
            a.flag = 0
            a.reference_id = 0
            a.reference_start = _READ_START
            a.mapping_quality = 60
            a.cigar = [(0, _READ_LEN)]  # 100M
            a.query_qualities = pysam.qualitystring_to_array(
                qual_char * _READ_LEN
            )
            a.set_tag("RG", rg_id)
            out.write(a)
    pysam.sort("-o", path, unsorted)
    pysam.index(path)
    os.remove(unsorted)


def _parse_prosolo_bcf(bcf_path: str) -> dict:
    """Parse ProSolo BCF output and return posteriors + P(alt present).

    ProSolo emits PHRED-scaled probabilities as INFO fields. We convert these
    to a linear scale and sum the five alt-present events.
    """
    save = pysam.set_verbosity(0)
    vcf = pysam.VariantFile(bcf_path)
    pysam.set_verbosity(save)
    rec = next(vcf)

    posteriors = {}
    for field in ("PROB_HOM_REF", "PROB_ADO_TO_REF", "PROB_ADO_TO_ALT",
                  "PROB_HOM_ALT", "PROB_ERR_ALT", "PROB_HET", "PROB_ERR_REF"):
        phred_val = rec.info[field][0]
        posteriors[field] = 10.0 ** (-phred_val / 10.0)
    prob_alt = sum(posteriors[e] for e in _ALT_PRESENT_EVENTS)
    vcf.close()
    return {"prob_alt": float(prob_alt), "posteriors": posteriors}


def prosolo_call_native(
    k_sc: int, n_sc: int, k_b: int, n_b: int,
    base_error_rate: float = BASE_ERROR_RATE_DEFAULT,
) -> dict:
    """
    Call the native ProSolo binary on synthetic BAMs built from read counts.

    Parameters
    ----------
    k_sc : int
        Alt reads in the single cell.
    n_sc : int
        Total reads in the single cell.
    k_b : int
        Alt reads in the bulk sample.
    n_b : int
        Total reads in the bulk sample.
    base_error_rate : float
        Sequencing base-call error rate.  Mapped to a uniform Phred base
        quality: Q = -10·log10(error_rate).  Default 0.01 → Q20.

    Returns
    -------
    dict
        ``prob_alt``  — P(alt allele present in cell | data)
        ``posteriors`` — dict mapping PROB_* event name → linear probability
    """
    base_qual = max(0, round(-10.0 * np.log10(base_error_rate)))
    tmpdir = tempfile.mkdtemp(prefix="prosolo_native_")
    try:
        ref_path = os.path.join(tmpdir, "ref.fa")
        vcf_path = os.path.join(tmpdir, "candidates.vcf")
        bulk_bam = os.path.join(tmpdir, "bulk.bam")
        sc_bam = os.path.join(tmpdir, "sc.bam")
        out_vcf = os.path.join(tmpdir, "output.vcf")

        _write_reference(ref_path)
        _write_candidate_vcf(vcf_path)
        _write_bam(bulk_bam, "BULK", "bulk", k_b, n_b, base_qual)
        _write_bam(sc_bam, "SC", "sc", k_sc, n_sc, base_qual)

        result = subprocess.run(
            [
                "prosolo", "single-cell-bulk",
                "--omit-indels",
                "--omit-fragment-evidence",
                "-c", vcf_path + ".gz",
                "-o", out_vcf,
                sc_bam,
                bulk_bam,
                ref_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"prosolo exited with code {result.returncode}: "
                f"{result.stderr}"
            )
        return _parse_prosolo_bcf(out_vcf)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _prosolo_call_native_worker(args: tuple) -> dict:
    """Worker function for prosolo_call_native_batch (picklable)."""
    k_sc, n_sc, k_b, n_b, base_error_rate = args
    return prosolo_call_native(k_sc, n_sc, k_b, n_b, base_error_rate)


def prosolo_call_native_batch(
    cells: list[tuple[int, int]], k_b: int, n_b: int,
    base_error_rate: float = BASE_ERROR_RATE_DEFAULT, n_workers: int = 1,
) -> list[dict]:
    """
    Run prosolo_call_native in parallel across multiple cells.

    Parameters
    ----------
    cells : list of (k_sc, n_sc)
        Alt and total read counts for each single cell.
    k_b : int
        Alt reads in the bulk sample (shared across all cells).
    n_b : int
        Total reads in the bulk sample.
    base_error_rate : float
        Sequencing base-call error rate.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    list[dict]
        One result dict per cell, same format as prosolo_call_native().
    """
    tasks = [(k_sc, n_sc, k_b, n_b, base_error_rate) for k_sc, n_sc in cells]
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            return pool.map(_prosolo_call_native_worker, tasks)
    return [_prosolo_call_native_worker(t) for t in tasks]
