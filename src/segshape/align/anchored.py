"""Anchored event-to-kmer alignment (segshape Step 5).

Per-read tight 4-corner bracketed semi-global Viterbi DP that maps the events
emitted by ``segshape segment`` (one row per find_peaks event) to a per-position
kmer model derived from the reference, with per-read scale/shift calibration.

Inputs (per sample):
  - ``3_alignment/subevents.parquet``     from ``segshape segment``
  - ``3_alignment/dorado.extract_mv.csv`` from ``segshape dorado-extract``
  - reference fasta + contig name
  - ONT kmer table (auto-resolved per --rna chemistry; bundled in the wheel)

Outputs (under ``3_alignment/``):
  - ``alignment.csv``      read_idx, event_idx, pos_idx
  - ``scale.csv``          per-read calibration + DP corner indices + ll
  - ``pos_kmer_table.csv`` per-position refined kmer mu/sigma (control only)

DP geometry — both axes run 3' → 5':

      kmer dim j:    j=0 (3' of ref) ─────────────────── j=L (5' of ref)
   event dim i:
   i=0 (3' of read)   V[0,0]                           V[0,L]
                                  inner DP V[i,j]
   i=N (5' of read)   V[N,0]                           V[N,L]

Per-read 4-corner windows from dorado mv (event axis) and minimap2 (kmer axis):
  k_seed = signal_idx_to_event_idx(mv_trans_start, event_starts)
  k_end  = signal_idx_to_event_idx(mv_trans_end,   event_starts)
  j_min  = max(0,         ref_len - ref_end + edge_pad)         (3' anchor)
  j_max  = min(L_ext - 1, ref_len - k - ref_start + edge_pad)   (5' anchor)

k-mer size + edge_pad are per chemistry (driven by --rna):
  RNA002: k=5, edge_pad=0  → L_ext = ref_len - 4, dead-zone 2+2 bases.
  RNA004: k=9, edge_pad=2  → L_ext = ref_len - 4, dead-zone 2+2 bases.

edge_pad=2 lets RNA004 cover the same SHAPE-callable range as RNA002. The
extra 2 kmer positions on each side overhang the ref end (1-2 X wildcards);
their (μ, σ) is moment-matched over all 4^|X| matching full 9-mers, which
naturally inflates σ at the edges and down-weights them in the DP.

After extension the center-base mapping is uniform across chemistries
(1-indexed, the SHAPE / RNAfold / .ct convention — written as
``ref_center_base_pos`` in alignment.csv):

    ref_center_pos = ref_len - (pos_idx + (k-1)//2 - edge_pad)
                   = ref_len - 2 - pos_idx        (both RNA002 & RNA004)

so pos_idx ∈ [0, L_ext-1] maps to ref base position [3, ref_len-2] (1-based).
Dead-zone = positions 1,2 (5' end) + ref_len-1, ref_len (3' end) = 4 nt.

Inner transitions:
  match : V[i-1, j-1] + ll(em[i-1] | mu[j-1]) + EPS
  stay  : V[i-1, j  ] + ll(em[i-1] | mu[j-1]) + EPS  (multi events / one kmer)
  skip  : V[i,   j-1] - skip_penalty                 (one kmer / no event)

Cells outside the kmer band [j_min - δ_k, j_max + δ_k] stay at NEG_INF (this is
the kmer-end HI cap that prevents phantom matches past mm2's alignment edge).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:
    sys.stderr.write("ERROR: numba required. conda install -c conda-forge numba\n")
    raise

NEG_INF = -1e18
LOG_2PI = np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------- #
# ONT kmer table loading                                                       #
#                                                                              #
# All bundled tables use **5'→3' canonical** k-mer keys (matching nanopolish / #
# dorado / ONT upstream convention). The two pA tables go with raw            #
# subevents.parquet (mean_pa ≈ 40–130); the two normalized-level tables go    #
# with subevents.norm.parquet (mean_pa ≈ z, median≈0). The signal domain is   #
# auto-detected in `_detect_norm_from_parquet` from the mean_pa range — no   #
# explicit `--norm` CLI flag.                                                  #
# See data/kmer_models/README.md for upstream provenance and direction proofs. #
# ---------------------------------------------------------------------------- #

# (rna, norm) → (filename, k, edge_pad)
KMER_MODEL_FILES: Dict[Tuple[str, str], Tuple[str, int, int]] = {
    ('002', 'none'):    ('ont_rna002_template_median69pA_5_to_3.model',         5, 0),
    ('002', 'med-mad'): ('ont_rna002_5mer_levels_v1_with_stdv_5_to_3.txt',      5, 0),
    ('004', 'none'):    ('f5c_rna004_9mer_template_5_to_3.csv',                 9, 2),
    ('004', 'med-mad'): ('ont_rna004_9mer_levels_v1_with_stdv_5_to_3.txt',      9, 2),
}

def _load_kmer_table(path: str) -> Dict[str, Tuple[float, float]]:
    """Load a 5'→3' canonical k-mer table. Auto-detect by extension/format:

      *.model   ONT-style TSV with columns
                  ``kmer  level_mean  level_stdv  sd_mean  sd_stdv ...``
                  → returns (level_mean, level_stdv).
      *.csv     CSV with optional ``#`` comments, columns
                  ``model_kmer,model_mean,model_stdv``
                  → returns (model_mean, model_stdv).
      *.txt     TSV/whitespace, 3 columns ``kmer  level_mean  level_stdv``
                  (locally augmented with σ_z from a paired pA table; see
                  ``data/kmer_models/build_levels_with_stdv.py``)
                  → returns (level_mean, level_stdv).
    """
    p = str(path)
    if p.endswith('.model'):
        df = pd.read_csv(p, sep='\t', comment='#')
        return {r.kmer: (float(r.level_mean), float(r.level_stdv))
                for r in df.itertuples()}
    if p.endswith('.csv'):
        df = pd.read_csv(p, comment='#')
        df.columns = ['kmer', 'mean', 'stdv'][:len(df.columns)]
        return {r.kmer: (float(r.mean), float(r.stdv)) for r in df.itertuples()}
    if p.endswith('.txt'):
        df = pd.read_csv(p, sep=r'\s+', header=None, comment='#')
        if df.shape[1] != 3:
            raise ValueError(
                f"{path}: expected 3 whitespace-separated columns "
                f"(kmer/level_mean/level_stdv), got {df.shape[1]}. "
                "Upstream ONT levels_v1 files (μ only, 2-col) are not "
                "accepted; run data/kmer_models/build_levels_with_stdv.py "
                "to derive the σ_z column first.")
        df.columns = ['kmer', 'level_mean', 'level_stdv']
        return {r.kmer: (float(r.level_mean), float(r.level_stdv))
                for r in df.itertuples()}
    raise ValueError(f"unrecognized kmer-table format: {path}")


def _resolve_xkmer(km: str, table: Dict[str, Tuple[float, float]]
                   ) -> Tuple[float, float]:
    """Look up a possibly X-containing kmer in ``table``.

    Pure (no X): direct lookup.
    With X (edge kmer): enumerate all 4^|X| matching full kmers and treat
    them as a uniform mixture, returning the matched-moments single-Gaussian
    approximation:

        μ = mean(μ_k)
        σ² = mean(σ_k² + μ_k²) − μ²

    This yields a wider σ at edges, naturally down-weighting them in the DP."""
    if 'X' not in km:
        return table[km]
    positions = [i for i, c in enumerate(km) if c == 'X']
    chars = list(km)
    mus: List[float] = []
    s2s: List[float] = []
    from itertools import product
    for combo in product('ACGT', repeat=len(positions)):
        for p, b in zip(positions, combo):
            chars[p] = b
        full = ''.join(chars)
        if full in table:
            m, s = table[full]
            mus.append(m)
            s2s.append(s * s)
    if not mus:
        raise KeyError(f"no full-kmer matches for X-kmer {km!r}")
    mus_a = np.asarray(mus, dtype=np.float64)
    s2s_a = np.asarray(s2s, dtype=np.float64)
    mu = float(mus_a.mean())
    var = float((s2s_a + mus_a * mus_a).mean() - mu * mu)
    sigma = float(np.sqrt(max(var, 1e-12)))
    return mu, sigma


def load_kmer_table(rna: str, norm: str = 'none',
                    override: Optional[str] = None
                    ) -> Tuple[Dict[str, Tuple[float, float]], int, int]:
    """Return ``(kmer_table, k, edge_pad)`` for the given chemistry + norm.

    rna  : '002' (5-mer, edge_pad=0) or '004' (9-mer, edge_pad=2).
    norm : 'none' → pA-scale tables(`.model`/`.csv`,paired with `subevents.parquet`)
           'med-mad' → normalized-level tables (`.txt`, paired with
           `subevents.norm.parquet`).
    override : explicit kmer-table path; format auto-detected by extension.
               k inferred from first key length, edge_pad chosen so
               ``(k-1)//2 - edge_pad == 2`` (uniform 2+2 dead-zone target).
               For 5-mers that's edge_pad=0; 9-mers → 2.
    """
    from segshape.data import kmer_model_path
    if override:
        path = override
        table = _load_kmer_table(path)
        if not table:
            raise ValueError(f"empty kmer table: {path}")
        k = len(next(iter(table)))
        edge_pad = max(0, (k - 1) // 2 - 2)
    else:
        key = (rna, norm)
        if key not in KMER_MODEL_FILES:
            raise ValueError(
                f"no kmer table for (rna={rna!r}, norm={norm!r}); "
                f"valid combos: {sorted(KMER_MODEL_FILES.keys())}")
        fname, k, edge_pad = KMER_MODEL_FILES[key]
        path = kmer_model_path(fname)
        table = _load_kmer_table(path)
    return table, k, edge_pad


def _detect_rna_from_extract_mv(path: str) -> Optional[str]:
    """Best-effort RNA chemistry detection from the dorado_version comment.

    ``segshape dorado-extract`` writes a ``# ... dorado_version=X.Y.Z ...``
    line as the first row. Returns '002' for dorado 0.9.x, '004' for 1.4+
    or 1.5, and None otherwise (caller must pass --rna)."""
    try:
        with open(path) as f:
            head = f.readline()
    except OSError:
        return None
    if not head.startswith('#'):
        return None
    if 'dorado_version=0.9' in head:
        return '002'
    for v in ('dorado_version=1.4', 'dorado_version=1.5',
              'dorado_version=1.6'):
        if v in head:
            return '004'
    return None


# ---------------------------------------------------------------------------- #
# Reference / pos-kmer helpers                                                 #
# ---------------------------------------------------------------------------- #

def read_reference_fasta(path: str, contig: str) -> str:
    cur, seqs = None, {}
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith('>'):
                cur = line[1:].split()[0]
                seqs[cur] = []
            else:
                seqs[cur].append(line)
    if contig not in seqs:
        raise KeyError(f"contig '{contig}' not in {path}; have: {list(seqs)}")
    return ''.join(seqs[contig]).upper().replace('U', 'T')


def extract_pos_kmers(ref_5to3: str, k: int, edge_pad: int = 0) -> List[str]:
    """Slide a k-window over the 5'→3' reference and emit k-mer strings in
    **5'→3' canonical form**, but indexed in **3'→5' positional order**
    (``pos_kmers[0]`` is the 3'-end k-mer, ``pos_kmers[L-1]`` is the 5'-end).

    The result is byte-identical at the (μ, σ) level to the legacy
    ``ref[::-1]``-then-slide pipeline used with 3'→5' literal-reversed tables
    — only the k-mer string representation differs (5'→3' here vs reversed
    there). Use with a 5'→3' canonical k-mer table (nanopolish/dorado/ONT
    upstream convention; see data/kmer_models/README.md).

    ``edge_pad`` extends the kmer axis by that many positions at each
    physical end of the reference. Edge k-mers (overhanging the fasta)
    carry ``'X'`` wildcards at the off-ref positions; their (μ, σ) is
    moment-matched over all 4^|X| matching full k-mers in
    ``init_pos_kmer_arrays`` / ``_resolve_xkmer``. List length =
    ``len(ref) - k + 1 + 2*edge_pad``.
    """
    n = len(ref_5to3)
    out: List[str] = []
    for j in range(-edge_pad, n - k + 1 + edge_pad):
        # pos_idx j=0 is the 3'-most k-mer.  As j grows we slide toward
        # the 5' end of the reference. To get the k-mer string at pos j in
        # 5'→3' canonical form, start its first base at ref position
        # (n - k - j) and read k bases forward (5'→3'). Out-of-bounds
        # positions get an 'X' wildcard.
        start = n - k - j
        chars = [ref_5to3[i] if 0 <= i < n else 'X'
                 for i in range(start, start + k)]
        out.append(''.join(chars))
    return out


def init_pos_kmer_arrays(pos_kmers: List[str],
                         ont_table: Dict[str, Tuple[float, float]],
                         max_stdv: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cold-start (μ, σ) per kmer position. Handles both pure kmers
    (direct lookup) and edge kmers with X wildcards (mixture moment match
    via ``_resolve_xkmer``)."""
    L = len(pos_kmers)
    mu = np.empty(L, dtype=np.float64)
    sigma = np.empty(L, dtype=np.float64)
    for i, km in enumerate(pos_kmers):
        try:
            m, s = _resolve_xkmer(km, ont_table)
        except KeyError as e:
            raise KeyError(f"kmer '{km}' at pos {i}: {e}") from None
        mu[i] = m
        sigma[i] = min(s, max_stdv)
    return mu, sigma


def load_pos_kmer_table_csv(path: str, expected_L: int, max_stdv: float
                            ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    if len(df) != expected_L:
        raise ValueError(
            f"pos_kmer_table {len(df)} rows, ref expects {expected_L}")
    return (df['mean'].to_numpy(np.float64),
            np.minimum(df['stdv'].to_numpy(np.float64), max_stdv),
            df['kmer'].tolist())


def _validate_pos_kmer_table_domain(pos_sigma: np.ndarray,
                                    ont_table: Dict[str, Tuple[float, float]],
                                    path: str) -> None:
    """Sanity check that a loaded ``pos_kmer_table.csv`` is in the same
    signal domain (pA vs z) as the current run's ONT k-mer table.

    The CSV has no domain metadata, so this compares median σ between the
    loaded table and the ONT table. Domain typical medians are well-
    separated (pA σ ≈ 3, z σ ≈ 0.17; ratio ~18×), so a ratio > 5× is a
    confident mismatch signal. Mixing domains here silently produces
    garbage alignments — bail loudly instead.

    Common cause: ``control`` ran on raw pA but treated is now aligning
    against subevents.norm.parquet (or vice versa)."""
    if pos_sigma.size == 0:
        return
    loaded_med = float(np.median(pos_sigma))
    ont_sig = np.fromiter((s for _, s in ont_table.values()),
                          dtype=np.float64, count=len(ont_table))
    ont_med = float(np.median(ont_sig))
    # Symmetric ratio so the order doesn't matter; guard against 0
    lo = max(min(loaded_med, ont_med), 1e-9)
    ratio = max(loaded_med, ont_med) / lo
    if ratio > 5.0:
        loaded_dom = 'pA' if loaded_med > 1.0 else 'z'
        ont_dom    = 'pA' if ont_med    > 1.0 else 'z'
        sys.exit(
            f"\n  ⚠ pos_kmer_table domain mismatch:\n"
            f"    {path}\n"
            f"    loaded median(σ)  = {loaded_med:.4f}  (looks like {loaded_dom} domain)\n"
            f"    current ONT median(σ) = {ont_med:.4f}  (looks like {ont_dom} domain)\n"
            f"    ratio = {ratio:.1f}×.  Re-run the control step on a parquet\n"
            f"    of the same domain ({ont_dom}) before this treated alignment.")


# ---------------------------------------------------------------------------- #
# Pipeline I/O                                                                 #
# ---------------------------------------------------------------------------- #

def load_extract_mv(path: str) -> pd.DataFrame:
    """Load ``dorado.extract_mv.csv`` and keep the columns alignment uses.

    Filters to ``is_child==0`` rows with valid mv/ref bounds. Drops
    duplicated read_id rows (observed: ~0.8% of non-child rows in some
    BAMs). The resulting frame is keyed on read_id."""
    df = pd.read_csv(path, comment='#',
                     usecols=['read_id', 'is_child',
                              'mv_trans_start', 'mv_trans_end',
                              'ref_start', 'ref_end', 'ref_len'])
    keep = (df.is_child == 0) & (df.mv_trans_start >= 0) & \
           (df.mv_trans_end > df.mv_trans_start) & \
           (df.ref_start >= 0) & (df.ref_end > df.ref_start)
    df = df[keep].copy()
    df['read_id'] = df['read_id'].astype(str)
    n_pre = len(df)
    df = df.drop_duplicates('read_id', keep='first')
    n_dup = n_pre - len(df)
    if n_dup:
        sys.stderr.write(
            f"  load_extract_mv: dropped {n_dup} duplicate read_id rows "
            f"({100 * n_dup / n_pre:.2f}%)\n")
    return df.set_index('read_id')


def load_subevents_parquet(path: str
                            ) -> Tuple[List[str],
                                       Dict[str, Tuple[np.ndarray,
                                                       np.ndarray,
                                                       np.ndarray,
                                                       np.ndarray]]]:
    """Stream-friendly load of ``subevents.parquet``.

    Returns:
        order   : unique read_id list in first-appearance order
                  (deterministic given segment's worker dispatch).
        events  : {read_id: (mean_pa, start_sample, dwell, event_idx)}
                  arrays sorted by event_idx, dtypes (f8, i8, i8, i8).
                  `dwell = end_sample - start_sample` (sample count per
                  subevent) — consumed by `--length-weight` alignment.
    """
    df = pd.read_parquet(
        path, columns=['read_id', 'event_idx',
                       'start_sample', 'end_sample', 'mean_pa'])
    df['read_id'] = df['read_id'].astype(str)
    # First-appearance order (avoids hashing-induced shuffle from groupby)
    order: List[str] = []
    seen: set = set()
    for rid in df['read_id'].to_numpy():
        if rid not in seen:
            seen.add(rid)
            order.append(rid)
    df = df.sort_values(['read_id', 'event_idx'], kind='stable')
    out: Dict[str, Tuple[np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray]] = {}
    n_nan_total = 0
    n_all_nan_reads = 0
    for rid, g in df.groupby('read_id', sort=False):
        em = g['mean_pa'].to_numpy(np.float64)
        # segment writes mean_pa = NaN for non-physical pA<0 subevents (the
        # row is kept so subevents.parquet preserves the [start_sample,
        # end_sample) tiling of signal time). Filter here so the DP sees a
        # dense, finite mean array — emission cost on NaN would propagate
        # through Numba ll accumulation. raw_idx (event_idx) keeps its
        # original per-read labels so alignment.csv references match the
        # parquet rows that are still present.
        mask = ~np.isnan(em)
        n_kept = int(mask.sum())
        if n_kept < len(em):
            n_nan_total += int((~mask).sum())
        if n_kept == 0:
            n_all_nan_reads += 1
        starts = g['start_sample'].to_numpy(np.int64)[mask]
        ends = g['end_sample'].to_numpy(np.int64)[mask]
        dwell = (ends - starts).astype(np.int64)
        out[rid] = (
            em[mask],
            starts,
            dwell,
            g['event_idx'].to_numpy(np.int64)[mask],
        )
    if n_nan_total:
        print(f"  filtered {n_nan_total:,} NaN-mean subevents "
              f"(pA<0 markers from segment)")
    if n_all_nan_reads:
        # Distinct from `build_reads`'s n_skipped_short — these are reads
        # where every subevent was pA<0 (extreme sensor / pore failure).
        # They land in the dict as empty arrays; build_reads skips them
        # via min_events but the diagnostic is useful at this layer too.
        print(f"  ⚠ {n_all_nan_reads} read(s) had ALL subevents NaN "
              f"(complete pA<0 read); will be skipped downstream")
    return order, out


def signal_idx_to_event_idx(signal_idx: int, event_starts: np.ndarray) -> int:
    """Locate the ordinal index of the first event whose ``start_sample``
    is ``>= signal_idx``. After segment's NaN filter, ``event_starts`` is
    a non-decreasing subsequence of pod5 sample positions; `searchsorted`
    handles the gaps correctly (a `signal_idx` that falls inside a
    filtered NaN event's [s_bad, e_bad) interval returns the next kept
    event's ordinal). Returns ``len(event_starts)`` when ``signal_idx``
    is past every kept event — caller in ``build_reads`` checks for this
    and skips the read."""
    if signal_idx < 0 or len(event_starts) == 0:
        return 0
    return int(np.searchsorted(event_starts, signal_idx, side='left'))


# ---------------------------------------------------------------------------- #
# Anchored Viterbi DP                                                          #
#                                                                              #
# Entry box  : i in [e_start_lo, e_start_hi]  AND  j in [k_start_lo, k_start_hi]
# Exit  box  : i in [e_end_lo,   e_end_hi  ]  AND  j in [k_end_lo,   k_end_hi  ]
# Inner band : j in [k_band_lo,  k_band_hi]   (cells outside stay NEG_INF)     #
#                                                                              #
# Anchored cost (linear) at entry:  -bc * (|i - k_seed| + |j - j_min|)         #
# Anchored cost (linear) at exit :  -bc * (|i - k_end | + |j - (j_max+1)|)     #
# bc=0 reproduces uniform-zero "free in window" behavior.                      #
# ---------------------------------------------------------------------------- #

@njit(cache=True, fastmath=True)
def _anchored_viterbi(em: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                      log_sigma: np.ndarray, scale: float, shift: float,
                      v_scale: float,
                      eps: float, skip_penalty: float, boundary_cost: float,
                      weights: np.ndarray,
                      k_seed: int, k_end: int,
                      j_min: int, j_max: int,
                      e_start_lo: int, e_start_hi: int,
                      e_end_lo: int, e_end_hi: int,
                      k_start_lo: int, k_start_hi: int,
                      k_end_lo: int, k_end_hi: int,
                      k_band_lo: int, k_band_hi: int):
    N = em.shape[0]
    L = mu.shape[0]
    j_exit_anchor = j_max + 1     # half-open exit kmer index
    log_v_scale = np.log(v_scale)             # scalar; pulled out of inner loop
    use_weights = weights.shape[0] == N

    V = np.full((N + 1, L + 1), NEG_INF)
    # bp: 0=match, 1=stay, 2=skip, 3=entry-init sentinel (back-track stop)
    bp = np.zeros((N + 1, L + 1), dtype=np.int8)

    # ---- Entry box init ----
    e_lo = e_start_lo if e_start_lo > 0 else 0
    e_hi = e_start_hi if e_start_hi < N + 1 else N + 1
    j_lo = k_start_lo if k_start_lo > 0 else 0
    j_hi = k_start_hi if k_start_hi < L + 1 else L + 1
    for i in range(e_lo, e_hi):
        for j in range(j_lo, j_hi):
            di = i - k_seed
            if di < 0:
                di = -di
            dj = j - j_min
            if dj < 0:
                dj = -dj
            cur = -boundary_cost * (di + dj)
            if cur > V[i, j]:
                V[i, j] = cur
                bp[i, j] = 3            # entry-init sentinel

    # ---- Forward DP, restricted to kmer band [k_band_lo, k_band_hi] ----
    # emission noise scale = v_scale * sigma[j-1]; v_scale=1 reproduces
    # the legacy behaviour bit-identically (log_v_scale=0).
    bj_lo = k_band_lo if k_band_lo > 1 else 1
    bj_hi = k_band_hi + 1 if k_band_hi + 1 < L + 1 else L + 1
    for i in range(1, N + 1):
        e = scale * em[i - 1] + shift
        w_i = weights[i - 1] if use_weights else 1.0
        Vi_prev = V[i - 1]
        Vi = V[i]
        bpi = bp[i]
        for j in range(bj_lo, bj_hi):
            d = (e - mu[j - 1]) / (v_scale * sigma[j - 1])
            ll = (-0.5 * LOG_2PI - log_sigma[j - 1] - log_v_scale
                  - 0.5 * d * d + eps)
            # Length-weighted emission: w_i = 1 in legacy mode (size-0
            # `weights`), w_i > 0 when --length-weight is enabled (e.g.
            # capped sample-count). Multiplies emission ONLY, not the
            # stay/skip transition penalties.
            ll = ll * w_i

            best = Vi_prev[j - 1] + ll
            best_bp = 0
            stay = Vi_prev[j] + ll
            if stay > best:
                best = stay
                best_bp = 1
            skip = Vi[j - 1] - skip_penalty
            if skip > best:
                best = skip
                best_bp = 2

            # Only commit cell if a transition produced a finite improvement,
            # OR the cell was already initialized by entry-box. Cells that
            # would stay at NEG_INF stay NEG_INF.
            if best > Vi[j]:
                Vi[j] = best
                bpi[j] = best_bp

    # ---- Exit box search ----
    e2_lo = e_end_lo if e_end_lo > 0 else 0
    e2_hi = e_end_hi + 1 if e_end_hi + 1 < N + 1 else N + 1
    j2_lo = k_end_lo if k_end_lo > 0 else 0
    j2_hi = k_end_hi + 1 if k_end_hi + 1 < L + 1 else L + 1

    best_i = e2_lo
    best_j = j2_lo
    best_ll = V[best_i, best_j]
    best_score = NEG_INF
    found = False
    for i in range(e2_lo, e2_hi):
        for j in range(j2_lo, j2_hi):
            v = V[i, j]
            if v <= NEG_INF / 2:    # unreached
                continue
            di = i - k_end
            if di < 0:
                di = -di
            dj = j - j_exit_anchor
            if dj < 0:
                dj = -dj
            score = v - boundary_cost * (di + dj)
            if (not found) or score > best_score:
                best_score = score
                best_ll = v
                best_i = i
                best_j = j
                found = True

    # ---- Backtrack ----
    alignment = np.full(N, -1, dtype=np.int64)
    if not found:
        return 0, 0, best_i, best_j, NEG_INF, alignment

    i, j = best_i, best_j
    while i > 0 and j > 0:
        if V[i, j] <= NEG_INF / 2:        # unreachable cell, abort
            break
        b = bp[i, j]
        if b == 3:                         # entry-init sentinel -> stop
            break
        if b == 0:
            alignment[i - 1] = j - 1
            i -= 1
            j -= 1
        elif b == 1:
            alignment[i - 1] = j - 1
            i -= 1
        elif b == 2:
            j -= 1
        else:                              # unset (shouldn't reach here)
            break

    return i, j, best_i, best_j, best_ll, alignment


# ---------------------------------------------------------------------------- #
# Per-read iterative align <-> scale                                           #
# ---------------------------------------------------------------------------- #

def fit_scale_shift(events: np.ndarray, mu_aligned: np.ndarray,
                    sigma_aligned: Optional[np.ndarray],
                    shift_bounds: Tuple[float, float],
                    v_scale_bounds: Tuple[float, float] = (0.5, 3.0),
                    mode: str = 'shift_only',
                    max_n: int = 800,
                    rng: Optional[np.random.Generator] = None
                    ) -> Tuple[float, float, float, float, float, float]:
    """Per-read calibration. Returns
    ``(scale, shift, v_scale, scale_raw, shift_raw, v_scale_raw)``.
    ``scale = scale_raw = 1`` always (slope is locked to 1 in every
    supported mode; only shift and optional v_scale are fit).

    Modes:
      'off'           — return identity (1, 0, 1) with NaN raw fields. The DP
                        runs without any calibration.
      'shift_only'    — fit shift = ``median(μ - event)``; ``v_scale = 1``.
                        This is the default; bit-identical to the pre-2026-05
                        per-read calibration.
      'shift_v_scale' — sequential closed-form: first ``shift = median(μ - event)``
                        (robust, independent of v), then ML
                        ``v_scale² = mean((event + shift - μ_aligned)² / σ_aligned²)``.
                        Per-read multiplicative calibration on the model σ
                        (nanopolish's ``var_scale``). Requires ``sigma_aligned``.

    The legacy 2-parameter Theil-Sen mode was removed in 2026-05 because it
    consistently fit a model-shape mismatch rather than per-read drift; see
    docs/attention.md for the diagnosis."""
    if mode == 'off':
        return (1.0, 0.0, 1.0,
                float('nan'), float('nan'), float('nan'))
    # n<30 is gated upstream by `valid.sum() >= 30` in align_one_read; this
    # function expects pre-filtered inputs.
    if mode == 'shift_only':
        intercept_raw = float(np.median(mu_aligned - events))
        v_raw = 1.0
    elif mode == 'shift_v_scale':
        if sigma_aligned is None:
            raise ValueError(
                "'shift_v_scale' requires sigma_aligned (per-aligned-position "
                "model σ); pass sigma[alignment[valid]] from the caller.")
        intercept_raw = float(np.median(mu_aligned - events))
        # Sequential closed-form ML: shift first (independent of v), then v
        # given shift. No joint optimisation, so no Theil-Sen-style coupling.
        residual = events + intercept_raw - mu_aligned
        v_sq = float(np.mean((residual / sigma_aligned) ** 2))
        v_raw = float(np.sqrt(v_sq)) if v_sq > 0 and np.isfinite(v_sq) else 1.0
    else:
        raise ValueError(
            f"unknown fit mode: {mode!r}. Supported: 'off', 'shift_only', "
            "'shift_v_scale'. ('theil_sen' was removed; see fit_scale_shift "
            "docstring.)")
    intercept_c = float(np.clip(intercept_raw, shift_bounds[0], shift_bounds[1]))
    v_c = float(np.clip(v_raw, v_scale_bounds[0], v_scale_bounds[1]))
    return 1.0, intercept_c, v_c, 1.0, intercept_raw, v_raw


def align_one_read(em: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                   log_sigma: np.ndarray, eps: float, skip_penalty: float,
                   boundary_cost: float,
                   k_seed: int, k_end: int, j_min: int, j_max: int,
                   delta_event: int, delta_kmer: int,
                   n_inner_iters: int,
                   shift_bounds: Tuple[float, float],
                   v_scale_bounds: Tuple[float, float] = (0.5, 3.0),
                   fit_mode: str = 'shift_only',
                   ll_tol: float = 0.1,
                   rng: Optional[np.random.Generator] = None,
                   init_shift: float = 0.0,
                   init_v_scale: float = 1.0,
                   weights: Optional[np.ndarray] = None):
    N = len(em)
    L = len(mu)
    j_exit_anchor = j_max + 1

    # Per-read 4-corner windows (clipped to legal DP table indices).
    e_start_lo = max(0,     k_seed - delta_event)
    e_start_hi = min(N + 1, k_seed + delta_event + 1)
    e_end_lo   = max(0,     k_end  - delta_event)
    e_end_hi   = min(N,     k_end  + delta_event)
    k_start_lo = max(0,     j_min  - delta_kmer)
    k_start_hi = min(L + 1, j_min  + delta_kmer + 1)
    k_end_lo   = max(0,     j_exit_anchor - delta_kmer)
    k_end_hi   = min(L,     j_exit_anchor + delta_kmer)
    k_band_lo  = k_start_lo
    k_band_hi  = k_end_hi

    # Slope is locked to 1.0 in all supported modes (no theil_sen).
    scale, shift, v_scale = 1.0, init_shift, init_v_scale
    scale_raw, shift_raw, v_scale_raw = 1.0, init_shift, init_v_scale
    last_ll = -np.inf
    k_start = j_start = j_end = best_end = 0
    ll = -np.inf
    alignment = None
    n_iter = 0

    # `weights` is the per-subevent length-weight vector. Empty / None
    # = legacy uniform-weight emission (bit-identical with pre-2026-05).
    if weights is None or len(weights) != N:
        weights_arr = np.zeros(0, dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64)

    # Loop ordering: DP → convergence check → (skip update on last iter) →
    # update calibration. This guarantees the (scale, shift, v_scale)
    # emitted in scale.csv match the DP that produced ``alignment``.
    for it in range(n_inner_iters):
        k_start, j_start, best_end, j_end, ll, alignment = _anchored_viterbi(
            em, mu, sigma, log_sigma, scale, shift, v_scale,
            eps, skip_penalty, boundary_cost,
            weights_arr,
            k_seed, k_end, j_min, j_max,
            e_start_lo, e_start_hi, e_end_lo, e_end_hi,
            k_start_lo, k_start_hi, k_end_lo, k_end_hi,
            k_band_lo, k_band_hi)
        n_iter = it + 1
        if fit_mode == 'off':
            break
        if it > 0 and abs(ll - last_ll) < ll_tol:
            break
        last_ll = ll
        if it == n_inner_iters - 1:
            break
        valid = alignment >= 0
        if valid.sum() >= 30:
            x = em[valid]
            aligned_idx = alignment[valid]
            y = mu[aligned_idx]
            z = sigma[aligned_idx] if fit_mode == 'shift_v_scale' else None
            (scale, shift, v_scale,
             scale_raw, shift_raw, v_scale_raw) = fit_scale_shift(
                x, y, z, shift_bounds, v_scale_bounds,
                mode=fit_mode, rng=rng)

    return (k_start, j_start, best_end, j_end, ll,
            scale, shift, v_scale,
            scale_raw, shift_raw, v_scale_raw,
            alignment, n_iter)


# ---------------------------------------------------------------------------- #
# Outer driver                                                                 #
# ---------------------------------------------------------------------------- #

@dataclass
class ReadData:
    read_idx: int
    read_id: str
    k_seed: int
    k_end:  int
    j_min:  int
    j_max:  int
    event_means: np.ndarray
    raw_idx: np.ndarray              # event_idx column from subevents.parquet
    event_dwells: Optional[np.ndarray] = None  # per-subevent sample count
    # alignment outputs
    k_start:   int = -1
    j_start:   int = -1
    j_end:     int = -1
    best_end:  int = -1
    ll: float = -np.inf
    scale:       float = 1.0
    shift:       float = 0.0
    v_scale:     float = 1.0
    scale_raw:   float = 1.0
    shift_raw:   float = 0.0
    v_scale_raw: float = 1.0
    n_iter: int = 0
    alignment: Optional[np.ndarray] = None


def build_reads(events_order: List[str],
                events_dict: Dict[str, Tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]],
                mv_df: pd.DataFrame, ref_len: int, k: int, edge_pad: int,
                min_events: int = 100
                ) -> Tuple[List[ReadData], int, int, int, int]:
    """Pair subevents.parquet rows with mv anchors to produce ReadData list.

    With edge_pad > 0 the kmer-axis is extended by edge_pad cells on each
    side (X-wildcard edge kmers); j_min / j_max are shifted accordingly so
    the entry/exit anchors still mark the boundary of the mm2-mapped
    region in the new (extended) j coordinate system.

    Returns ``(reads, n_total, n_skipped_no_mv, n_skipped_short,
    n_skipped_anchor_oob)``. The last bucket counts reads whose
    mv_trans_start lands past every kept event (pathological — would
    leave the DP entry box clipped to the table edge with a degenerate
    alignment outcome)."""
    n_total = len(events_order)
    L_ext = ref_len - k + 1 + 2 * edge_pad
    reads: List[ReadData] = []
    n_skipped_no_mv = 0
    n_skipped_short = 0
    n_skipped_anchor_oob = 0
    for read_idx, rid in enumerate(events_order):
        if rid not in mv_df.index:
            n_skipped_no_mv += 1
            continue
        em, starts, dwell, raw_idx = events_dict[rid]
        if len(em) < min_events:
            n_skipped_short += 1
            continue
        row = mv_df.loc[rid]
        k_seed = signal_idx_to_event_idx(int(row.mv_trans_start), starts)
        k_end  = signal_idx_to_event_idx(int(row.mv_trans_end),   starts)
        # k_seed == len(em) means signal_idx_to_event_idx couldn't find any
        # kept event with start_sample >= mv_trans_start (every event is
        # before the entry anchor). The DP entry box would clip to the
        # table edge and produce a degenerate / failed alignment; skip
        # rather than feed it through.
        if k_seed >= len(em):
            n_skipped_anchor_oob += 1
            continue
        if k_end <= k_seed:
            k_end = min(len(em), k_seed + 1)
        j_min = max(0,         ref_len - int(row.ref_end)        + edge_pad)
        j_max = min(L_ext - 1, ref_len - k - int(row.ref_start)  + edge_pad)
        if j_max <= j_min:
            j_max = min(L_ext - 1, j_min + 1)
        reads.append(ReadData(
            read_idx=read_idx, read_id=rid,
            k_seed=k_seed, k_end=k_end, j_min=j_min, j_max=j_max,
            event_means=em.astype(np.float64),
            event_dwells=dwell.astype(np.int64),
            raw_idx=raw_idx,
        ))
    return (reads, n_total, n_skipped_no_mv, n_skipped_short,
            n_skipped_anchor_oob)


def _make_length_weights(dwell: Optional[np.ndarray], mode: str,
                         cap: float = 30.0) -> Optional[np.ndarray]:
    """Map a per-subevent dwell vector to a likelihood-weight vector.

      mode='none'    → None (legacy uniform weighting)
      mode='n'       → w_i = n_i                      (raw count)
      mode='sqrt_n'  → w_i = sqrt(n_i)                (info-theoretic √)
      mode='capped'  → w_i = min(n_i, cap)            (default — clips
                                                       stalls so a single
                                                       outlier doesn't
                                                       dominate the DP)
    """
    if dwell is None or mode == 'none':
        return None
    n = np.asarray(dwell, dtype=np.float64)
    if mode == 'n':
        return n
    if mode == 'sqrt_n':
        return np.sqrt(np.maximum(n, 0.0))
    if mode == 'capped':
        return np.minimum(n, cap)
    raise ValueError(f"unknown length-weight mode: {mode!r}")


def align_all_reads(reads: List[ReadData], pos_mu: np.ndarray,
                    pos_sigma: np.ndarray, eps: float, skip_penalty: float,
                    boundary_cost: float,
                    delta_event: int, delta_kmer: int,
                    n_inner_iters: int,
                    shift_bounds: Tuple[float, float],
                    v_scale_bounds: Tuple[float, float] = (0.5, 3.0),
                    fit_mode: str = 'shift_only',
                    ll_tol: float = 0.1,
                    rng: Optional[np.random.Generator] = None,
                    length_weight_mode: str = 'none',
                    length_weight_cap: float = 30.0) -> None:
    log_sigma = np.log(pos_sigma)
    t0 = time.time()
    n = len(reads)
    for r_idx, r in enumerate(reads):
        try:
            weights = _make_length_weights(
                r.event_dwells, length_weight_mode, length_weight_cap)
            (k_start, j_start, best_end, j_end, ll,
             scale, shift, v_scale,
             scale_raw, shift_raw, v_scale_raw,
             align, n_iter) = align_one_read(
                r.event_means, pos_mu, pos_sigma, log_sigma, eps,
                skip_penalty, boundary_cost,
                r.k_seed, r.k_end, r.j_min, r.j_max,
                delta_event, delta_kmer,
                n_inner_iters, shift_bounds, v_scale_bounds,
                fit_mode=fit_mode, ll_tol=ll_tol, rng=rng,
                init_shift=r.shift,
                init_v_scale=r.v_scale,
                weights=weights)
            r.k_start = int(k_start)
            r.j_start = int(j_start)
            r.best_end = int(best_end)
            r.j_end = int(j_end)
            r.ll = float(ll)
            r.scale = float(scale)
            r.shift = float(shift)
            r.v_scale = float(v_scale)
            r.scale_raw = float(scale_raw)
            r.shift_raw = float(shift_raw)
            r.v_scale_raw = float(v_scale_raw)
            r.n_iter = n_iter
            r.alignment = align
        except Exception as e:
            sys.stderr.write(
                f"  read {r.read_id} (read_idx={r.read_idx}) failed: {e}\n")
            r.alignment = None
        if (r_idx + 1) % 200 == 0 or r_idx + 1 == n:
            elapsed = time.time() - t0
            rate = (r_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"    {r_idx + 1}/{n} reads  "
                  f"({elapsed:.0f}s, {rate:.1f} rd/s)")


def update_pos_mu(reads: List[ReadData], ont_mu_pos: np.ndarray,
                  kappa: float) -> Tuple[np.ndarray, np.ndarray]:
    L = len(ont_mu_pos)
    sums = np.zeros(L, dtype=np.float64)
    counts = np.zeros(L, dtype=np.int64)
    for r in reads:
        if r.alignment is None:
            continue
        valid = r.alignment >= 0
        if not valid.any():
            continue
        scaled = r.scale * r.event_means[valid] + r.shift
        positions = r.alignment[valid]
        np.add.at(sums, positions, scaled)
        np.add.at(counts, positions, 1)
    new_mu = (sums + kappa * ont_mu_pos) / (counts + kappa)
    return new_mu, counts


# ---------------------------------------------------------------------------- #
# Output                                                                       #
# ---------------------------------------------------------------------------- #

def _scale_shift_clip_summary(reads: List[ReadData],
                              shift_bounds: Tuple[float, float],
                              v_scale_bounds: Tuple[float, float],
                              fit_mode: str
                              ) -> Tuple[int, int, int, str]:
    """Count PASS reads whose raw shift / v_scale hit the configured clip
    bounds. A clipped raw signals that ``fit_scale_shift`` is fitting a
    systematic bias rather than per-read calibration.

    Returns ``(n_pass, n_shift_clip, n_v_scale_clip, summary_str)``.
    ``summary_str`` is empty when ``fit_mode == 'off'`` or no PASS reads.
    PASS = read produced ≥1 aligned event (matches write_outputs's
    qc_tag='PASS' criterion); FAIL reads (DP returned all-(-1) alignment)
    and pre-DP-skipped reads are excluded from the denominator so
    clip-rate fractions reflect the actually-calibrated population."""
    if fit_mode == 'off':
        return 0, 0, 0, ""
    eps = 1e-6
    sh_lo, sh_hi = shift_bounds
    v_lo, v_hi = v_scale_bounds
    pass_reads = [r for r in reads
                  if r.alignment is not None
                  and (r.alignment >= 0).any()
                  and not (np.isnan(r.shift_raw) or np.isnan(r.v_scale_raw))]
    n_pass = len(pass_reads)
    if n_pass == 0:
        return 0, 0, 0, ""
    n_sh = sum(1 for r in pass_reads
               if r.shift_raw <= sh_lo + eps or r.shift_raw >= sh_hi - eps)
    parts = [f"shift clipped {n_sh}/{n_pass} ({100 * n_sh / n_pass:.1f}%)"]
    n_v = 0
    if fit_mode == 'shift_v_scale':
        n_v = sum(1 for r in pass_reads
                  if r.v_scale_raw <= v_lo + eps
                  or r.v_scale_raw >= v_hi - eps)
        parts.append(
            f"v_scale clipped {n_v}/{n_pass} ({100 * n_v / n_pass:.1f}%)")
    s = "  calibration clip-rate — " + ", ".join(parts)
    return n_pass, n_sh, n_v, s


def write_outputs(reads: List[ReadData], events_order: List[str],
                  out_dir: str, ref_len: int, k: int, edge_pad: int) -> None:
    """Flat per-sample output: one alignment.csv + one scale.csv covering
    every read_id in subevents.parquet (in events_order).

    alignment.csv columns:
        read_idx, event_idx, pos_idx, ref_center_base_pos

    where ``ref_center_base_pos`` is the **1-indexed** 5'→3' reference base
    position that this kmer is centered on (the SHAPE / RNAfold / .ct
    convention). Computed as

        ref_center_base_pos = ref_len - (pos_idx + (k-1)//2 - edge_pad)

    With our chemistry-aware edge_pad (RNA002 → 0, RNA004 → 2) this
    simplifies to ``ref_len - 2 - pos_idx`` for both chemistries; range is
    [3, ref_len-2], dead-zone is positions 1,2 (5' end) + ref_len-1, ref_len
    (3' end)."""
    os.makedirs(out_dir, exist_ok=True)
    align_path = os.path.join(out_dir, 'alignment.csv')
    scale_path = os.path.join(out_dir, 'scale.csv')
    anchor_off = (k - 1) // 2 - edge_pad        # = 2 for both RNA002 & RNA004
    by_idx = {r.read_idx: r for r in reads}
    with open(align_path, 'w', newline='') as fa, \
            open(scale_path, 'w', newline='') as fs:
        wa = csv.writer(fa)
        ws = csv.writer(fs)
        wa.writerow(['read_idx', 'event_idx', 'pos_idx',
                     'ref_center_base_pos'])
        ws.writerow(['read_idx', 'read_id', 'qc_tag', 'k_seed',
                     'k_start', 'best_end', 'j_start', 'j_end',
                     'n_events',
                     'scale', 'shift', 'v_scale',
                     'scale_raw', 'shift_raw', 'v_scale_raw',
                     'n_iter', 'll'])
        nan = float('nan')
        for read_idx, rid in enumerate(events_order):
            r = by_idx.get(read_idx)
            # Three failure regimes, all written with qc_tag='FAIL' and NaN
            # placeholders so scale.csv stays float-typed end-to-end. The
            # specific reason (no_mv / short / oob / DP-fail) is logged in
            # stdout via build_reads counters; downstream consumers filter
            # on `qc_tag == 'PASS'`.
            if r is None:
                # pre-DP skip: not_in_mv / short_events / anchor_oob.
                ws.writerow([read_idx, rid, 'FAIL', -1, -1, -1, -1, -1, 0,
                             nan, nan, nan, nan, nan, nan, 0, nan])
                continue
            if r.alignment is None:
                # exception inside align_one_read (rare; logged to stderr).
                ws.writerow([read_idx, r.read_id, 'FAIL', r.k_seed,
                             -1, -1, -1, -1, len(r.event_means),
                             nan, nan, nan, nan, nan, nan, 0, nan])
                continue
            if not (r.alignment >= 0).any():
                # DP returned all-(-1): no reachable exit-box cell.
                ws.writerow([read_idx, r.read_id, 'FAIL', r.k_seed,
                             -1, -1, -1, -1, len(r.event_means),
                             nan, nan, nan, nan, nan, nan, r.n_iter, nan])
                continue
            ws.writerow([read_idx, r.read_id, 'PASS', r.k_seed,
                         r.k_start, r.best_end, r.j_start, r.j_end,
                         len(r.event_means),
                         round(r.scale, 5), round(r.shift, 4),
                         round(r.v_scale, 5),
                         round(r.scale_raw, 5), round(r.shift_raw, 4),
                         round(r.v_scale_raw, 5),
                         r.n_iter, round(r.ll, 2)])
            for ev_i, pos in enumerate(r.alignment):
                if pos >= 0:
                    pos_i = int(pos)
                    ref_center_pos = ref_len - (pos_i + anchor_off)
                    wa.writerow([read_idx, int(r.raw_idx[ev_i]), pos_i,
                                 ref_center_pos])
    print(f"  Wrote {align_path}")
    print(f"  Wrote {scale_path}")


def write_pos_kmer_table(path: str, pos_kmers: List[str], pos_mu: np.ndarray,
                         pos_sigma: np.ndarray, counts: np.ndarray) -> None:
    pd.DataFrame({
        'pos': np.arange(len(pos_kmers)),
        'kmer': pos_kmers,
        'mean': np.round(pos_mu, 3),
        'stdv': np.round(pos_sigma, 3),
        'n_obs': counts.astype(int),
    }).to_csv(path, index=False)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------- #
# CLI                                                                          #
# ---------------------------------------------------------------------------- #

def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # ---- Sample addressing ----
    p.add_argument('--root-dir', required=True,
                   help='Root folder containing {dataset}/{sample}/...')
    p.add_argument('--dataset', required=True)
    p.add_argument('--sample', choices=['control', 'treated'], required=True)
    p.add_argument('--reference-file', required=True,
                   help='fasta basename under {root_dir}/{dataset}/reference/')
    p.add_argument('--contig', required=True)

    # ---- I/O overrides (default = 3_alignment/ layout) ----
    p.add_argument('--subevents-parquet', dest='subevents_parquet',
                   default=None,
                   help='default: {sample_dir}/3_alignment/subevents.parquet')
    p.add_argument('--mv-csv', default=None,
                   help='default: {sample_dir}/3_alignment/dorado.extract_mv.csv')
    p.add_argument('--out-dir', default=None,
                   help='default: {sample_dir}/3_alignment/')

    # ---- Chemistry / kmer model ----
    p.add_argument('--rna', choices=['002', '004'], default=None,
                   help='RNA chemistry. Auto-detected from extract_mv.csv '
                        'dorado_version comment if omitted.')
    # Note: signal normalization (raw pA vs med-mad z-domain) is auto-detected
    # from the subevents.parquet's mean_pa range — pA lands in [40, 130], z
    # near 0. No CLI flag needed; the chosen k-mer table follows automatically.
    p.add_argument('--ont-kmer-file', default=None,
                   help='Override bundled ONT kmer table.')
    p.add_argument('--pos-kmer-table', default=None,
                   help='control: optional warm start; treated: REQUIRED. '
                        'For treated, defaults to '
                        '{dataset}/control/3_alignment/pos_kmer_table.csv if '
                        'that file exists.')

    # ---- Iterations ----
    p.add_argument('--n-outer-iters', type=int, default=3,
                   help='control kmer-table refinement rounds '
                        '(treated forced to 1)')
    p.add_argument('--n-inner-iters', type=int, default=3,
                   help='inner DP↔calibration rounds. Iter 0 runs '
                        'uncalibrated (init scale=1, shift=0, v_scale=1) '
                        'so for chemistries with a systematic shift '
                        '(RNA004 ~ +10 pA, see attention.md §4) keep ≥ 2 '
                        '— otherwise the alignment that lands in '
                        'alignment.csv reflects an uncalibrated DP and '
                        'the per-read shift in scale.csv has not been '
                        'applied to the path.')
    p.add_argument('--ll-tol', type=float, default=0.1)

    # ---- Anchored windows ----
    p.add_argument('--delta-event', type=int, default=50,
                   help='per-read event-axis tolerance (events) on both '
                        'entry (mv_trans_start) and exit (mv_trans_end) anchors.')
    p.add_argument('--delta-kmer', type=int, default=15,
                   help='per-read kmer-axis tolerance (kmers) on both '
                        'entry (j_min from ref_end) and exit (j_max+1 from '
                        'ref_start) anchors. Cells outside [j_min - dk, '
                        'j_max + dk] stay NEG_INF.')

    # ---- Penalty / kmer model knobs ----
    p.add_argument('--epsilon', type=float, default=None,
                   help='match-likelihood offset. default = mean(log_sigma) '
                        '+ 0.5*log(2pi) + 0.5')
    p.add_argument('--skip-penalty', type=float, default=50.0)
    p.add_argument('--boundary-cost', type=float, default=0.0,
                   help='linear cost per cell of slack from each anchor.')
    p.add_argument('--max-stdv', type=float, default=5.0)
    p.add_argument('--sigma-override', type=float, default=None)
    p.add_argument('--sigma-multiplier', type=float, default=1.5,
                   help='multiplicative scaler on σ_k after table load. '
                        'Preserves per-kmer σ heterogeneity (every σ_k is '
                        'multiplied by the same factor). Default 1.5 (the '
                        'production setting) dilates the bundled k-mer table '
                        'σ, which underestimates observed per-read residuals '
                        '(e.g. z-domain σ_z median ≈ 0.17), widening the DP '
                        'tolerance. Applied before --max-stdv cap; '
                        'overridden by --sigma-override.')
    p.add_argument('--kappa', type=float, default=50.0)
    # ---- Length-weighted emission ----
    p.add_argument('--length-weight', choices=('none', 'capped',
                                                'sqrt_n', 'n'),
                   default='capped',
                   help='per-subevent emission weighting. '
                        '`capped` (default) = min(dwell, '
                        '--length-weight-cap) — weights each subevent by its '
                        'dwell while taming stall outliers. '
                        '`none` = legacy uniform weighting. '
                        '`sqrt_n` = sqrt(dwell) — information-theoretic '
                        'half-weighting. `n` = raw sample count.')
    p.add_argument('--length-weight-cap', type=float, default=30.0,
                   help='upper bound on the per-subevent weight when '
                        '--length-weight=capped (default: 30 ≈ kmer '
                        'dwell q99 in clean reads; prevents a single '
                        'stall from dominating the DP).')

    # ---- Shift / v_scale ----
    # (--scale-bounds removed in 2026-05: slope is locked to 1.0 in every
    #  supported fit-mode after theil_sen retirement; see fit_scale_shift.)
    p.add_argument('--shift-bounds', type=float, nargs=2,
                   default=[-30.0, 30.0], metavar=('LO', 'HI'),
                   help="bounds on the per-read intercept (μ − event).")
    p.add_argument('--v-scale-bounds', type=float, nargs=2,
                   default=[0.5, 3.0], metavar=('LO', 'HI'),
                   help="bounds on the per-read multiplicative σ-scale "
                        "(nanopolish-style var_scale). Only used when "
                        "--fit-mode shift_v_scale. Default upper bound "
                        "raised from 2.0 to 3.0 in 2026-05 after baseline "
                        "sweeps showed bsub_16S/treated saturating 47%% of "
                        "reads at 2.0 (v_scale_raw reached 4+ on noisier reads).")
    p.add_argument('--fit-mode',
                   choices=['off', 'shift_only', 'shift_v_scale'],
                   default='shift_only',
                   help="per-read calibration. 'shift_only' (default) fits "
                        "the intercept only — equivalent to the legacy "
                        "behaviour, scale=1, v_scale=1. 'shift_v_scale' "
                        "additionally fits a multiplicative σ-scale "
                        "(nanopolish-style var_scale) per read by closed-form "
                        "ML against pos_sigma. 'off' skips calibration. "
                        "('theil_sen' was removed in 2026-05 — see "
                        "fit_scale_shift docstring.)")

    # ---- Misc ----
    p.add_argument('--min-events', type=int, default=100,
                   help='skip reads with fewer than this many events.')
    p.add_argument('--max-reads', type=int, default=0,
                   help='debug: cap aligned reads; 0 = all.')
    p.add_argument('--save-iters', action='store_true',
                   help='write pos_kmer_table_iter{n}.csv after each outer '
                        'iter (control only).')
    p.add_argument('--seed', type=int, default=0)
    return p


# Raw-pA default mirrored by add_arguments; used to detect un-overridden
# shift-bounds in `_apply_norm_defaults`. skip_penalty does NOT need rescaling
# under norm because the bundled `_with_stdv` σ_z tables restore pA-equivalent
# DP geometry (cost magnitudes ≈ raw-pA path, see docs/alignment_norm.md §2.x).
RAW_PA_SHIFT_BOUNDS = [-30.0, 30.0]


def _detect_norm_from_parquet(path: str, sample_n: int = 50_000) -> str:
    """Infer the signal-domain ('none' = raw pA, 'med-mad' = z) from the
    mean_pa column of ``subevents.parquet``. Reads only the first row group
    (or up to ``sample_n`` rows) to keep this fast on multi-GB files.

    Decision rule (mean_pa percentiles on the sample):
      |median| < 5  AND  q95(|val|) < 10            → 'med-mad'  (z domain, ~0±3)
      30 < median < 150  AND  q95(|val|) < 200      → 'none'     (raw pA, ~85 typical)
      otherwise                                     → ValueError (ambiguous)

    The decision gap between the two domains is ~6× the upper bound on z, so
    a parquet falling in neither bucket signals a real anomaly (truncated
    file, mis-merged data, non-standard chemistry) worth bailing on rather
    than guessing."""
    df = pd.read_parquet(path, columns=['mean_pa'])
    if len(df) > sample_n:
        df = df.iloc[:sample_n]
    vals = df['mean_pa'].dropna().to_numpy()
    if len(vals) == 0:
        raise ValueError(f"{path}: no finite mean_pa rows")
    med = float(np.median(vals))
    q95 = float(np.percentile(np.abs(vals), 95))
    if abs(med) < 5 and q95 < 10:
        return 'med-mad'
    if 30 < med < 150 and q95 < 200:
        return 'none'
    raise ValueError(
        f"{path}: cannot infer signal domain — median(mean_pa)={med:.3f}, "
        f"q95(|mean_pa|)={q95:.3f}. Expected median≈0 + q95<10 for med-mad "
        "z-domain, or 30<median<150 for raw pA.")


def _apply_norm_defaults(args, norm: str, rna: str) -> None:
    """Under z-domain ('med-mad') the pA-tuned `shift_bounds` default stops
    making sense (z scale is ~18× smaller). σ_z from the bundled `_with_stdv`
    tables restores pA-equivalent DP geometry — so `skip_penalty` and
    `epsilon` auto-defaults carry over unchanged.

    `shift_bounds` is **chemistry-specific** because empirical |shift| q95
    differs ~5× between 002 and 004 (RNA002 polyA-biased tpp datasets reach
    q95=0.72 σ; RNA004 9-mer riboswitch q95=0.16 σ).

      rna='002'  → [-1.0, 1.0]   (covers RNA002 worst-case q95 + safety)
      rna='004'  → [-0.3, 0.3]   (5× tighter, fits 9-mer's narrow shift dist)

    `fit_mode shift_v_scale` is always downgraded to `shift_only` under z-
    domain because per-read MAD is already removed upstream — re-fitting
    v_scale absorbs modification signal (docs/alignment_norm.md §2.3)."""
    if norm != 'med-mad':
        return
    if list(args.shift_bounds) == RAW_PA_SHIFT_BOUNDS:
        old = list(args.shift_bounds)
        if rna == '004':
            args.shift_bounds = [-0.3, 0.3]
        else:                       # '002' and any future 5-mer chemistries
            args.shift_bounds = [-1.0, 1.0]
        print(f"  norm domain: shift-bounds auto-adjusted {old} → "
              f"{args.shift_bounds}  (chemistry rna{rna}; "
              f"pass --shift-bounds to override)")
    if args.fit_mode == 'shift_v_scale':
        sys.stderr.write(
            "  ⚠ WARNING: --fit-mode shift_v_scale on a z-domain parquet would\n"
            "    re-fit per-read σ on top of the segment-stage MAD normalization,\n"
            "    absorbing modification signal as a scale residual. Auto-\n"
            "    downgrading to 'shift_only'. Pass --fit-mode shift_only\n"
            "    explicitly to silence this warning.\n")
        args.fit_mode = 'shift_only'


def run(args):
    rng = np.random.default_rng(args.seed)
    sample_dir = os.path.join(args.root_dir, "datasets", args.dataset, args.sample)
    align_dir = os.path.join(sample_dir, '3_alignment')

    # ---- Resolve I/O paths ----
    # Default prefers subevents.norm.parquet when both files exist (newer
    # / recommended path); fall back to raw pA subevents.parquet otherwise.
    if args.subevents_parquet:
        subevents_path = args.subevents_parquet
    else:
        norm_path = os.path.join(align_dir, 'subevents.norm.parquet')
        raw_path  = os.path.join(align_dir, 'subevents.parquet')
        subevents_path = norm_path if os.path.isfile(norm_path) else raw_path
    mv_csv = args.mv_csv or os.path.join(align_dir, 'dorado.extract_mv.csv')
    out_dir = args.out_dir or align_dir
    for label, p in [('subevents.parquet', subevents_path),
                     ('extract_mv.csv', mv_csv)]:
        if not os.path.isfile(p):
            sys.exit(f"missing {label}: {p}")

    # ---- Detect signal domain from parquet content ----
    norm = _detect_norm_from_parquet(subevents_path)
    print(f"Subevents  : {subevents_path}")
    print(f"Norm       : {norm}  (auto-detected from mean_pa range)")

    # ---- RNA chemistry resolution (must precede kmer-table load: it sets k) ----
    rna = args.rna or _detect_rna_from_extract_mv(mv_csv)
    if rna is None:
        sys.exit("--rna {002,004} could not be auto-detected from "
                 f"{mv_csv}; pass --rna explicitly.")
    print(f"RNA        : {rna}  (auto-detected)" if not args.rna
          else f"RNA        : {rna}")

    ont_table, k, edge_pad = load_kmer_table(rna, norm=norm,
                                              override=args.ont_kmer_file)
    print(f"ONT table  : {len(ont_table)} kmers, k={k}, edge_pad={edge_pad}, "
          f"norm={norm}")
    if norm == 'med-mad':
        sig_vals = np.fromiter((s for _, s in ont_table.values()),
                               dtype=np.float64, count=len(ont_table))
        print(f"  norm domain: σ_k from 3-col table  "
              f"[{sig_vals.min():.3f}, {sig_vals.max():.3f}]  "
              f"median={float(np.median(sig_vals)):.3f}  "
              "(DP geometry ≈ raw-pA equivalent)")
    _apply_norm_defaults(args, norm, rna)

    # ---- Reference (k + edge_pad now known) ----
    ref_path = os.path.join(args.root_dir, "datasets", args.dataset,
                            'reference', args.reference_file)
    ref_seq = read_reference_fasta(ref_path, args.contig)
    pos_kmers = extract_pos_kmers(ref_seq, k=k, edge_pad=edge_pad)
    L = len(pos_kmers)              # = L_ext
    ref_len = len(ref_seq)
    n_edge = 2 * edge_pad
    print(f"Reference  : {args.contig}  len={ref_len} nt  "
          f"L_ext={L} positions ({n_edge} edge X-kmer)")
    # ont_mu_pos is only consumed by update_pos_mu's kappa shrinkage,
    # which runs in the 'control' branch only — skip the (cheap but
    # pure dead-code-on-treated) computation otherwise.
    if args.sample == 'control':
        ont_mu_pos = np.array(
            [_resolve_xkmer(km, ont_table)[0] for km in pos_kmers],
            dtype=np.float64)
    else:
        ont_mu_pos = None

    # ---- Init pos_mu / pos_sigma ----
    pos_kmer_path = args.pos_kmer_table
    if args.sample == 'treated' and pos_kmer_path is None:
        # Auto-find sibling control's pos_kmer_table.csv
        sibling = os.path.join(args.root_dir, "datasets", args.dataset,
                               'control', '3_alignment', 'pos_kmer_table.csv')
        if os.path.isfile(sibling):
            pos_kmer_path = sibling
            print(f"pos_kmer_table (auto from control): {sibling}")

    if args.sample == 'treated':
        if not pos_kmer_path:
            sys.exit("--pos-kmer-table is required for sample=treated "
                     "(or run control first to produce one).")
        pos_mu, pos_sigma, loaded = load_pos_kmer_table_csv(
            pos_kmer_path, L, args.max_stdv)
        if loaded != pos_kmers:
            mismatch = sum(1 for a, b in zip(pos_kmers, loaded) if a != b)
            sys.exit(f"pos_kmer_table kmers don't match reference "
                     f"(mismatches={mismatch})")
        _validate_pos_kmer_table_domain(pos_sigma, ont_table, pos_kmer_path)
        print(f"Loaded frozen pos_kmer_table: {pos_kmer_path}")
        n_outer = 1
    else:
        if pos_kmer_path:
            pos_mu, pos_sigma, loaded = load_pos_kmer_table_csv(
                pos_kmer_path, L, args.max_stdv)
            if loaded != pos_kmers:
                mismatch = sum(1 for a, b in zip(pos_kmers, loaded) if a != b)
                sys.exit(f"pos_kmer_table kmers don't match reference "
                         f"(mismatches={mismatch})")
            _validate_pos_kmer_table_domain(pos_sigma, ont_table, pos_kmer_path)
            print(f"Loaded warm-start pos_kmer_table: {pos_kmer_path}")
        else:
            pos_mu, pos_sigma = init_pos_kmer_arrays(
                pos_kmers, ont_table, args.max_stdv)
            print("Initialized pos_kmer_table from ONT (cold start)")
        n_outer = args.n_outer_iters

    # ---- sigma scaling / override / epsilon ----
    # Multiplier dilates σ_k uniformly (preserves per-kmer heterogeneity);
    # applied AFTER the table's per-position σ is built (cold or warm start)
    # and BEFORE the constant override. The --max-stdv cap was already
    # applied inside init_pos_kmer_arrays / load_pos_kmer_table_csv, so
    # multiplier > 1 can push σ above max_stdv — re-cap here.
    if args.sigma_multiplier != 1.0:
        pre = float(np.median(pos_sigma))
        pos_sigma = np.minimum(pos_sigma * args.sigma_multiplier, args.max_stdv)
        post = float(np.median(pos_sigma))
        print(f"sigma multiplier: × {args.sigma_multiplier}  "
              f"(median σ {pre:.4f} → {post:.4f}; capped at --max-stdv "
              f"{args.max_stdv})")
    if args.sigma_override is not None:
        pos_sigma = np.full_like(pos_sigma, args.sigma_override)
        print(f"sigma override: ALL positions -> {args.sigma_override}")
    if args.epsilon is None:
        eps = float(np.mean(np.log(pos_sigma)) + 0.5 * LOG_2PI + 0.5)
        print(f"epsilon (auto): {eps:.3f}")
    else:
        eps = float(args.epsilon)
        print(f"epsilon (manual): {eps:.3f}")
    shift_bounds = (float(args.shift_bounds[0]), float(args.shift_bounds[1]))
    v_scale_bounds = (float(args.v_scale_bounds[0]),
                      float(args.v_scale_bounds[1]))
    print(f"shift clip = {shift_bounds}, v_scale clip = {v_scale_bounds}")
    print(f"skip_penalty = {args.skip_penalty}, "
          f"boundary_cost = {args.boundary_cost}")
    print(f"delta_event = {args.delta_event}, "
          f"delta_kmer = {args.delta_kmer}")
    print(f"fit_mode    = {args.fit_mode}")

    # ---- Load extract_mv ----
    mv_df = load_extract_mv(mv_csv)
    print(f"Loaded extract_mv: {mv_csv}  ({len(mv_df)} non-child valid rows)")

    # ---- Load subevents.parquet ----
    events_order, events_dict = load_subevents_parquet(subevents_path)
    print(f"  subevents: {len(events_order)} unique reads")

    # ---- Build ReadData ----
    reads, n_total, n_no_mv, n_short, n_anchor_oob = build_reads(
        events_order, events_dict, mv_df, ref_len, k, edge_pad,
        min_events=args.min_events)
    if args.max_reads:
        reads = reads[:args.max_reads]
    skip_parts = [f"{n_no_mv} not-in-mv", f"{n_short} short-events"]
    if n_anchor_oob:
        skip_parts.append(f"{n_anchor_oob} anchor-out-of-bounds")
    print(f"Usable reads: {len(reads)} / {n_total}  "
          f"(skipped {', '.join(skip_parts)})")
    print(f"Output     : {out_dir}")

    # ---- Outer iterations ----
    counts = np.zeros(L, dtype=np.int64)
    for outer in range(n_outer):
        print(f"\n=== Outer iter {outer + 1}/{n_outer} ({args.sample}) ===")
        align_all_reads(reads, pos_mu, pos_sigma, eps, args.skip_penalty,
                        args.boundary_cost,
                        args.delta_event, args.delta_kmer,
                        args.n_inner_iters,
                        shift_bounds, v_scale_bounds,
                        fit_mode=args.fit_mode, ll_tol=args.ll_tol, rng=rng,
                        length_weight_mode=args.length_weight,
                        length_weight_cap=args.length_weight_cap)
        if args.sample == 'control':
            new_mu, counts = update_pos_mu(reads, ont_mu_pos, args.kappa)
            delta = np.abs(new_mu - pos_mu).mean()
            covered = int((counts > 0).sum())
            print(f"  pos_mu update: mean |Δμ| = {delta:.4f} pA  "
                  f"covered={covered}/{L}  "
                  f"median n_obs={int(np.median(counts))}")
            pos_mu = new_mu
            if args.save_iters:
                snap = os.path.join(
                    out_dir, f'pos_kmer_table_iter{outer + 1}.csv')
                os.makedirs(out_dir, exist_ok=True)
                write_pos_kmer_table(
                    snap, pos_kmers, pos_mu, pos_sigma, counts)

    # ---- Final realign so output matches the shipped pos_kmer_table ----
    if args.sample == 'control' and n_outer > 0:
        print("\n=== Final realign with converged pos_kmer_table ===")
        align_all_reads(reads, pos_mu, pos_sigma, eps, args.skip_penalty,
                        args.boundary_cost,
                        args.delta_event, args.delta_kmer,
                        args.n_inner_iters,
                        shift_bounds, v_scale_bounds,
                        fit_mode=args.fit_mode, ll_tol=args.ll_tol, rng=rng,
                        length_weight_mode=args.length_weight,
                        length_weight_cap=args.length_weight_cap)
        _, counts = update_pos_mu(reads, ont_mu_pos, args.kappa)

    # ---- Calibration clip-rate diagnostic ----
    n_pass, n_sh, n_v, summary = _scale_shift_clip_summary(
        reads, shift_bounds, v_scale_bounds, args.fit_mode)
    if summary:
        print(summary)
        # 30% threshold: lowered from 50% (2026-05) because bsub_16S/treated
        # showed 47% v_scale clip without triggering the warning, and the
        # clip-rate denominator now correctly excludes FAIL reads, so a
        # given fraction is more meaningful.
        if n_pass > 0 and n_sh / n_pass > 0.3:
            sys.stderr.write(
                "\n  ⚠ WARNING: > 30% of PASS reads hit the shift clip\n"
                "    bound — pore baselines vary more than --shift-bounds\n"
                f"    {shift_bounds!r} permits. Consider widening the\n"
                "    bounds (e.g. --shift-bounds -50 50) or use --fit-mode\n"
                "    off if calibration is unreliable.\n\n")
        if n_pass > 0 and n_v / n_pass > 0.3:
            sys.stderr.write(
                "\n  ⚠ WARNING: > 30% of PASS reads hit the v_scale clip\n"
                f"    bound {v_scale_bounds!r} — per-read σ-scale calibration\n"
                "    is consistently outside the bounds, suggesting a\n"
                "    model-σ mismatch rather than per-read drift. Consider\n"
                "    widening --v-scale-bounds or use --fit-mode shift_only.\n\n")

    # ---- Write outputs ----
    print("\nWriting outputs...")
    write_outputs(reads, events_order, out_dir, ref_len, k, edge_pad)
    if args.sample == 'control':
        write_pos_kmer_table(
            os.path.join(out_dir, 'pos_kmer_table.csv'),
            pos_kmers, pos_mu, pos_sigma, counts)
    print("Done.")


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Anchored alignment via dorado moves and minimap2 '
                    '+ per-read scaling (control/treated).')
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main() or 0)
