"""Per-read pod5 → peaks → subevents parquet.

For each read in the pod5 folder whose read_id appears in
``dorado.extract_mv.csv`` (the dorado-extract whitelist):

  1. Read pod5 signal in pA, clip to ``[called_start, called_end)`` =
     ``[ts, ns)`` — the basecaller-consumed signal interval exported by
     dorado-extract. This range contains polyA + 5'/3' soft-clip +
     transcript, but excludes the adapter / open-pore stretch dorado
     already trimmed at ``ts``. The pre/post-``mv_trans_*`` padding
     gives the downstream Viterbi DP a real bilateral entry/exit
     buffer (so ``[k_seed − delta_event, k_seed + delta_event]`` can
     actually look at events before mv_trans_start).
  2. Compute slope → smooth → ``scipy.signal.find_peaks(distance=...)``.
  3. Emit one **subevent** per inter-peak segment with both
     ``mean_pa`` and ``std_pa`` computed on the **same** trimmed sample
     set (``--trim`` proportion on each end).

The "subevent" name (vs nanopolish's "event" = per-base) reflects that our
find_peaks segmentation produces a finer subdivision: ~3-7 subevents per
basecalled base.

Output parquet schema (sample positions are *absolute* pod5 coordinates so
they directly join with BAM ``mv_trans_*`` / ``polya_*`` columns)::

    read_id        string
    event_idx      int32
    start_sample   int64
    end_sample     int64
    mean_pa        float32   ← mean of trimboth(pA samples, --trim) in [s, e)
    std_pa         float32   ← std  of trimboth(pA samples, --trim) in [s, e)

Both ``mean_pa`` and ``std_pa`` are computed on the *same* sample set
(``trimboth(seg, --trim)``). **Default ``--trim 0.1``** drops 10 %
from each end — find_peaks places subevent boundaries on slope peaks,
and the samples nearest each boundary are in transition between
neighbouring base levels, inflating both the mean (toward the
neighbour) and std (boundary jitter). Pass ``--trim 0`` to disable
(raw ``seg.mean()`` / ``seg.std()``).

The trim default was bumped 0.0 → 0.1 in 2026-05 after a head-to-head
miR17-92 retest scored every (cell, method) combination on five
metrics: trim=0.1 won on Reactivity Spearman (+0.026), alignment
LL/event (+0.01-0.06), modification-signal Δv (+0.034), centroid MCC
(+5.71 pp at the best config), and structural AUC (+0.014). Boundary
samples turned out to be transition jitter, not modification signal.
See ``docs/attention.md §3`` for the full audit.

Trim only affects the FINAL ``mean_pa`` / ``std_pa`` emitted to the
parquet. All internal decisions inside ``_resplit_high_std_subevents``
(``--resplit-std`` threshold, BM, R², post-split sanity) use **raw**
std on the raw signal. Concretely: a subevent whose parquet
``std_pa = 2.9`` (trimmed) can still trigger ``--resplit-std 3.0`` if
its raw std is 3.0+. The two scales differ by ~5-10 % depending on
subevent length. See ``_trim_stats`` and
``_resplit_high_std_subevents`` for details.

The actual ``--trim`` value used for a given output file is recorded in
the parquet file-level metadata under key ``trim`` so downstream code can
verify it.

Streaming write (pyarrow ParquetWriter, 100k-row batches) keeps peak memory
small even for the largest datasets (RNA004 riboswitch_*, ~20M subevent rows).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pod5
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import find_peaks
from scipy.stats import trimboth

from segshape import __version__


DEFAULT_SUBEVENTS_PARQUET = "subevents.parquet"
DEFAULT_SUBEVENTS_NORM_PARQUET = "subevents.norm.parquet"
DEFAULT_EXTRACT_CSV = "dorado.extract_mv.csv"
DEFAULT_TRIM = 0.1
NORM_CHOICES = ("none", "med-mad")
DEFAULT_NORM = "med-mad"

# Module-level handle for the dorado-extract whitelist, set by the parent
# before spawning workers. With fork start method (Linux default), workers
# inherit this without pickling — saves ~120 MB IPC for the 1 M-entry case.
_WORKER_WINDOW: dict[str, tuple[int, int]] | None = None
# (peak_distance, smooth_box, trim,
#  resplit_std, resplit_distance, resplit_smooth, resplit_max_pieces,
#  resplit_min_piece_len, resplit_bm_th, resplit_r2_th, norm)
_WORKER_PARAMS: tuple[int, int, float, float, int, int, int, int, float, float, str] = (
    10, 3, DEFAULT_TRIM, 0.15, 5, 1, 2, 4, 1.5, 0.70, DEFAULT_NORM)


def med_mad_normalisation(sig: np.ndarray) -> tuple[float, float]:
    """Per-read median-MAD location/scale estimate.

    Mirrors ``nanofm/src/nanofm/data_pre/signal_reader.py::med_mad_normalisation``
    (the bonito-style nanopore signal normalization). Returns ``(shift, scale)``
    such that the normalized signal is ``(sig - shift) / scale``:

      shift = median(sig)
      mad   = median(|sig - shift|)
      scale = 1.4826 × mad        (with a 1e-6 floor to avoid div-by-zero)

    The 1.4826 factor makes ``scale`` an unbiased estimator of σ for Gaussian
    data. Robust to up to ~25-50% outliers (modifications, segmentation
    artefacts) by design of the median/MAD pair.

    Applied to the full per-read pA signal (before clipping to [ts, ns)) so
    polyA / adapter samples participate in the stats — this matches bonito
    and nanofm convention and keeps the normalization read-independent of
    where the basecaller decided to cut.
    """
    shift = float(np.median(sig))
    mad = float(np.median(np.abs(sig - shift)))
    scale = max(mad * 1.4826, 1e-6)
    return shift, scale

EVENTS_SCHEMA = pa.schema([
    ("read_id",      pa.string()),
    ("event_idx",    pa.int32()),
    ("start_sample", pa.int64()),
    ("end_sample",   pa.int64()),
    ("mean_pa",      pa.float32()),
    ("std_pa",       pa.float32()),
])

_BATCH_SIZE = 100_000


def smooth(y: np.ndarray, box_pts: int) -> np.ndarray:
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode="same")


def run_slope(y: np.ndarray, win_size_arr=(1, 2)) -> np.ndarray:
    """Vectorised symmetric-window OLS slope per sample. Mirrors the pipeline
    in segment/peaks.py so peak positions are bit-identical to the legacy
    ``segshape segmentation peaks`` step (now folded into ``segshape segment``)."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    res = np.zeros(n, dtype=np.float64)
    for w in win_size_arr:
        kernel = np.arange(-w, w + 1, dtype=np.float64)
        denom = float((kernel ** 2).sum())
        conv = np.correlate(y, kernel, mode="same") / denom
        if w > 0:
            conv[:w] = 0
            conv[n - w:] = 0
        res += conv
    return res


def _trim_stats(seg: np.ndarray, trim: float) -> tuple[float, float]:
    """Mean and std of ``seg`` after sort-trim of ``trim`` fraction from
    each end. Same sample set for both stats (fixes the mean/std
    asymmetry the legacy implementation had).

    Short-segment behaviour (for default ``trim = 0.1``):

      ============  =========================================  ==================
      len(seg) n    `trimboth` keeps                            returned stats
      ============  =========================================  ==================
      0             —                                           ``(NaN, NaN)``
      1             1 sample (``int(0.1)=0`` → nothing trimmed) ``(value, 0)``
      2 … 9         all samples (``floor(n·0.1)=0`` each side)  raw mean / std
      10 … 19       n − 2 samples (1 trimmed each side)         trimmed stats
      20 … 29       n − 4 samples                                trimmed stats
      ≥ 30          n − ``2·int(n·0.1)`` samples                 trimmed stats
      ============  =========================================  ==================

    So at the default ``trim``, **subevents with fewer than 10 samples
    are statistically identical to the raw case** — trimboth has nothing
    to trim away. Only longer subevents see a real noise-floor reduction.

    Fallback guards:

      * ``trim <= 0`` → raw mean / std directly.
      * ``2 · int(n · trim) >= n`` (would empty or invert the slice for
        very large ``trim``) → fall back to raw so callers always get
        finite numbers.

    Single-sample subevents (``n = 1``) return ``std = 0`` (by numpy
    convention for ``ddof = 0``). Downstream code that reads
    ``std_pa = 0`` should treat such rows as artefactual; the canonical
    way the pipeline avoids them is to drop ``peaks[0]`` when it's
    closer than ``peak_distance`` to sample 0."""
    if trim <= 0:
        return float(seg.mean()), float(seg.std())
    n = len(seg)
    if 2 * int(n * trim) >= n:                  # would empty the slice
        return float(seg.mean()), float(seg.std())
    seg_trim = trimboth(seg, trim)
    return float(seg_trim.mean()), float(seg_trim.std())


def _bm_score(seg: np.ndarray) -> float:
    """Bimodality score: |mean(L) − mean(R)| / pooled_within_std.

    Large when the subevent has two halves with clearly different
    means relative to within-half noise (Welch-t-like). Used by the
    shape-gated resplit to distinguish "two-kmer stack" from
    "single noisy kmer"."""
    h = len(seg) // 2
    if h < 4 or len(seg) - h < 4:
        return 0.0
    L = seg[:h].astype(np.float64)
    R = seg[h:].astype(np.float64)
    sL, sR = float(L.std()), float(R.std())
    pooled = float(np.sqrt((sL * sL + sR * sR) / 2.0))
    if pooled < 1e-6:
        return float('inf')
    return abs(float(L.mean()) - float(R.mean())) / pooled


def _linear_r2(seg: np.ndarray) -> float:
    """R^2 of OLS fit ``signal ~ sample_idx``. High = ramp-like
    (linear drift), low = plateau-like. Used by the shape gate to
    EXCLUDE transition fragments (which look linear) from resplit."""
    n = len(seg)
    if n < 4:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = seg.astype(np.float64)
    xm, ym = x.mean(), y.mean()
    sxy = float(((x - xm) * (y - ym)).sum())
    sxx = float(((x - xm) ** 2).sum())
    syy = float(((y - ym) ** 2).sum())
    if sxx <= 0 or syy <= 0:
        return 0.0
    slope = sxy / sxx
    ss_res = syy - slope * sxy
    return max(0.0, 1.0 - ss_res / syy)


def _resplit_high_std_subevents(
    clip: np.ndarray, bounds: np.ndarray, *,
    resplit_std: float, resplit_distance: int, resplit_smooth: int,
    resplit_max_pieces: int = 2,
    resplit_min_piece_len: int = 4,
    resplit_bm_th: float = 1.5,
    resplit_r2_th: float = 0.70,
) -> np.ndarray:
    """Single-pass shape-gated resplit of suspicious subevents.

    For each subevent in ``bounds``, a candidate split is attempted
    only when ALL of these hold (the "shape gate"):

      1. ``len(seg) >= 2 * resplit_min_piece_len`` — long enough that
         halves give stable statistics.
      2. ``seg.std() > resplit_std`` — the subevent is noisier than
         expected. **Raw std** (not trimmed), so the threshold scale
         matches an unprocessed signal view. The ``trim`` argument is
         intentionally NOT used inside resplit; it is applied only to
         the final emitted ``mean_pa`` / ``std_pa`` in ``_signal_to_events``.
      3. ``bm_score(seg) >= resplit_bm_th`` (default 1.5) — the two
         halves' means differ by ≥ 1.5 × pooled within-std, i.e. the
         subevent is **bimodal**. Excludes single-kmer subevents that
         just happen to be noisy.
      4. ``linear_r2(seg) <= resplit_r2_th`` (default 0.70) — the
         signal is NOT a linear ramp. Excludes transition fragments
         (case (c) in the audit), which would otherwise pass the
         bimodal check but should not be split.

    When the gate passes, place at most ``resplit_max_pieces − 1`` new
    boundaries at the strongest inner slope peaks (``find_peaks`` at
    ``resplit_distance``). Per benchmark on simulated nanopore signal
    (segment_simulate/) the practical sweet spot is
    ``resplit_max_pieces = 2`` (one extra cut), so the default
    changed from 3 to 2 in 2026-05.

    Post-split verification: each piece must NOT itself pass the
    shape gate (otherwise we'd recurse on a still-bimodal half — but
    we don't recurse). Subevents that fail any gate check are left
    unchanged.

    Audit history (2026-05) compared three triggers across noise
    levels 0.5–2.0 on simulated reads:
      - std-only (the original):  pure_rate +0.9 pp vs no-resplit but
        trans_frag rate climbs to 0.15 % (transitions cut).
      - strict shape (BM ≥ 2.5, R² ≤ 0.55): too conservative; almost
        no splits fire; gains < 0.1 pp.
      - loose shape (BM ≥ 1.5, R² ≤ 0.70, the current default):
        matches std-only on pure_rate / recall, lowers trans_frag by
        30 %, and runs ~2× faster than std-only because most
        subevents fail the BM check early."""
    if resplit_std <= 0 or len(bounds) < 2 or resplit_max_pieces < 2:
        return bounds
    max_new_bounds = resplit_max_pieces - 1
    min_piece_len = max(2, int(resplit_min_piece_len))
    new_bounds = [int(bounds[0])]
    n_resplit = 0
    for k in range(len(bounds) - 1):
        s, e = int(bounds[k]), int(bounds[k + 1])
        new_bounds.append(e)
        if e - s < 2 * min_piece_len:               # too short to attempt
            continue
        seg = clip[s:e]
        sd = float(seg.std())
        if not np.isfinite(sd) or sd <= resplit_std:
            continue
        # Shape gate — bimodal & non-ramp
        if _bm_score(seg) < resplit_bm_th:
            continue
        if _linear_r2(seg) > resplit_r2_th:
            continue
        sub_slope = smooth(np.abs(run_slope(seg)), max(1, resplit_smooth))
        sub_peaks, _ = find_peaks(sub_slope, distance=resplit_distance)
        viable = [int(p) for p in sub_peaks
                  if min_piece_len <= int(p) <= (e - s) - min_piece_len]
        if not viable:
            continue
        if len(viable) > max_new_bounds:
            viable = sorted(viable,
                            key=lambda p: -float(sub_slope[p]))[:max_new_bounds]
            viable = sorted(viable)
        # Post-split sanity: every piece must FAIL the shape gate
        # individually — i.e. each one looks like a single kmer now.
        cuts = [0] + viable + [e - s]
        gate_recurses = False
        for kk in range(len(cuts) - 1):
            piece = seg[cuts[kk]:cuts[kk + 1]]
            if len(piece) < 2 * min_piece_len:
                continue
            piece_sd = float(piece.std())
            if (np.isfinite(piece_sd)
                    and piece_sd > resplit_std
                    and _bm_score(piece) >= resplit_bm_th
                    and _linear_r2(piece) <= resplit_r2_th):
                gate_recurses = True
                break
        if gate_recurses:
            continue
        for p in viable:
            new_bounds.insert(-1, s + p)
            n_resplit += 1
    if n_resplit == 0:
        return bounds
    return np.array(sorted(set(new_bounds)), dtype=np.int64)


def _signal_to_events(
    clip: np.ndarray, *, peak_distance: int, smooth_box: int, abs_offset: int,
    trim: float = DEFAULT_TRIM,
    resplit_std: float = 0.0,
    resplit_distance: int = 5,
    resplit_smooth: int = 1,
    resplit_max_pieces: int = 2,
    resplit_min_piece_len: int = 4,
    resplit_bm_th: float = 1.5,
    resplit_r2_th: float = 0.70,
    norm: str = DEFAULT_NORM,
) -> list[tuple[int, int, int, float, float]]:
    """Per-subevent statistics:
      mean_pa  = mean of trimboth(seg, ``trim``)
      std_pa   = std  of trimboth(seg, ``trim``)   (same sample set as mean)

    Both stats are computed on the same trimmed sample set so they describe
    the steady-state core of the subevent without boundary contamination.
    See module docstring for rationale; default ``trim`` = 0.0 (no trim;
    use trimboth's full sample set on each subevent).

    When ``resplit_std > 0``, after the first ``find_peaks`` pass any
    subevent whose trimmed std exceeds the threshold is locally re-cut
    with finer parameters (``resplit_distance`` / ``resplit_smooth``).
    See ``_resplit_high_std_subevents`` docstring for details.
    """
    if len(clip) < 4:
        return []
    slope = smooth(np.abs(run_slope(clip)), smooth_box)
    peaks, _ = find_peaks(slope, distance=peak_distance)
    # find_peaks(distance=...) enforces a min gap between *consecutive*
    # peaks, but not between sample 0 and peaks[0]. When the signal spikes
    # near sample 0 the leading slice [0, peaks[0]) can be 1-3 samples and
    # falls into _trim_stats's short-seg fallback (raw mean/std on n<=4),
    # which yielded std_pa=0 in the very-first subevent of every read on
    # 2026-05 audit (14/14 samples affected, 100% of std=0 rows landed in
    # event_idx=0). Merge by dropping peaks[0]: first emitted subevent
    # then runs [0, peaks[1]) which is guaranteed >= peak_distance long.
    if len(peaks) >= 1 and peaks[0] < peak_distance:
        peaks = peaks[1:]
    bounds = np.concatenate(([0], peaks, [len(clip)])).astype(np.int64)
    # NOTE: trim is intentionally NOT passed to resplit — internal
    # gates use raw std on the raw signal. trim only applies when we
    # emit mean_pa / std_pa below.
    bounds = _resplit_high_std_subevents(
        clip, bounds,
        resplit_std=resplit_std,
        resplit_distance=resplit_distance,
        resplit_smooth=resplit_smooth,
        resplit_max_pieces=resplit_max_pieces,
        resplit_min_piece_len=resplit_min_piece_len,
        resplit_bm_th=resplit_bm_th,
        resplit_r2_th=resplit_r2_th)
    out = []
    for k in range(len(bounds) - 1):
        s, e = int(bounds[k]), int(bounds[k + 1])
        if e <= s:
            continue
        seg = clip[s:e]
        m, sd = _trim_stats(seg, trim)
        # Mark non-physical pA<0 subevents as NaN (sensor / calibration
        # artefacts; ONT pore current is positive on intact pores). 2026-05
        # audit: <=10 rows per (dataset, sample), all in mid-read with no
        # boundary clustering. We keep the row (preserving the [start_sample,
        # end_sample) tiling of the signal time axis) but blank both stats
        # so downstream DP/mod-calling never feeds the bad value into a
        # Gaussian emission (~((mu-m)/s)^2 of order 1e3-1e4) or a per-pos
        # mean. anchored._load_events filters NaN at load time so the DP
        # sees a dense mean array; mod-calling joins on (read_id, event_idx)
        # and never references the NaN rows because alignment.csv only
        # contains aligned (non-skipped) events.
        # `m < 0` alone misses NaN (any NaN comparison returns False), so
        # stats from an all-NaN raw-signal slice (sensor glitch) leak through
        # with mismatched mean/std. Use isfinite() to catch NaN, ±inf, and
        # negative-pA in one shot — all three get blanked to (NaN, NaN).
        # Also flag std == 0 (single-sample subevents or pathologically
        # constant slice): downstream Gaussian likelihood divides by std,
        # so std=0 would yield ±inf likelihood. anchored._load_events
        # filters NaN, so blanking is the cleanest opt-out.
        # When ``norm == "med-mad"``, the m<0 guard is dropped: normalized
        # mean values legitimately span both signs around 0. We add an
        # outlier guard ``abs(m) > 50`` instead (50σ is unreachable for
        # any healthy signal — median/MAD normalization keeps the bulk
        # within a few σ; any |m| ≥ 50 indicates a sensor glitch or
        # near-zero MAD that blew up the scale factor).
        bad_negative = (norm != "med-mad") and m < 0
        bad_outlier  = (norm == "med-mad") and abs(m) > 50
        if (not np.isfinite(m)) or bad_negative or bad_outlier or sd == 0:
            m = float('nan')
            sd = float('nan')
        out.append((k, abs_offset + s, abs_offset + e, m, sd))
    return out


def _load_whitelist(extract_csv: Path) -> dict[str, tuple[int, int]]:
    """Per-read signal window for segmentation.

    Returns ``{read_id: (called_start, called_end)}`` = ``(ts, ns)``, the
    basecaller-consumed interval (polyA + 5'/3' soft clip + transcript;
    excludes adapter / open-pore). Using [ts, ns) instead of
    [mv_trans_start, mv_trans_end) gives the downstream Viterbi DP a
    real bilateral entry/exit buffer (``k_seed > 0``), so the
    ``[k_seed - delta_event, k_seed + delta_event]`` entry box can
    actually search events before mv_trans_start.
    """
    df = pd.read_csv(extract_csv, comment="#",
                     usecols=["read_id", "called_start", "called_end"])
    n_in = len(df)
    df = df[(df.called_start >= 0) & (df.called_end > df.called_start)]
    n_skipped = n_in - len(df)
    if n_skipped:
        print(f"  skipped {n_skipped:,} reads with invalid called_start/end",
              file=sys.stderr)
    return dict(zip(df.read_id, zip(df.called_start, df.called_end)))


def _process_one_pod5(
    pod5_path: Path,
    window: dict[str, tuple[int, int]],
    *,
    peak_distance: int,
    smooth_box: int,
    trim: float = DEFAULT_TRIM,
    resplit_std: float = 0.0,
    resplit_distance: int = 5,
    resplit_smooth: int = 1,
    resplit_max_pieces: int = 2,
    resplit_min_piece_len: int = 4,
    resplit_bm_th: float = 1.5,
    resplit_r2_th: float = 0.70,
    norm: str = DEFAULT_NORM,
) -> tuple[int, int, dict[str, list]]:
    """Iterate one pod5 file, segment whitelisted reads. Return
    (n_seen, n_kept, columnar_dict). Used by both sequential and parallel
    paths.

    When ``norm == "med-mad"``, each read's full pA signal is z-score-like
    normalized via :func:`med_mad_normalisation` BEFORE clipping to
    [ts, ns) and BEFORE segmentation. Peak positions are scale-invariant
    (find_peaks here uses only ``distance``, not ``prominence``/``height``),
    so segmentation boundaries are bit-identical to the raw-pA case; only
    the emitted ``mean_pa`` / ``std_pa`` values change unit (now in
    ``MAD``-scaled normalized units, ~ σ-units against ONT v1 normalized
    k-mer models). The column names stay ``mean_pa`` / ``std_pa`` for
    schema compatibility; the ``norm`` field in parquet metadata tells
    downstream consumers what unit the values are in.
    """
    cols: dict[str, list] = {col: [] for col in EVENTS_SCHEMA.names}
    n_seen = 0
    n_kept = 0
    with pod5.Reader(pod5_path) as reader:
        for read in reader.reads():
            n_seen += 1
            rid = str(read.read_id)
            w = window.get(rid)
            if w is None:
                continue
            s_lo, s_hi = int(w[0]), int(w[1])
            full = np.asarray(read.signal_pa, dtype=np.float32)
            s_hi = min(s_hi, len(full))
            if s_lo >= s_hi:
                continue
            if norm == "med-mad":
                shift, scale = med_mad_normalisation(full)
                full = ((full - shift) / scale).astype(np.float32)
            events = _signal_to_events(
                full[s_lo:s_hi],
                peak_distance=peak_distance,
                smooth_box=smooth_box,
                abs_offset=s_lo,
                trim=trim,
                resplit_std=resplit_std,
                resplit_distance=resplit_distance,
                resplit_smooth=resplit_smooth,
                resplit_max_pieces=resplit_max_pieces,
                resplit_min_piece_len=resplit_min_piece_len,
                resplit_bm_th=resplit_bm_th,
                resplit_r2_th=resplit_r2_th,
                norm=norm,
            )
            if not events:
                continue
            for ev in events:
                cols["read_id"].append(rid)
                cols["event_idx"].append(ev[0])
                cols["start_sample"].append(ev[1])
                cols["end_sample"].append(ev[2])
                cols["mean_pa"].append(ev[3])
                cols["std_pa"].append(ev[4])
            n_kept += 1
    return n_seen, n_kept, cols


def _worker_segment(args: tuple[str, str]) -> tuple[str, str | None, int, int]:
    """Pool worker. Reads ``_WORKER_WINDOW`` / ``_WORKER_PARAMS`` from the
    inherited (fork) module globals; returns (pod5_basename, shard_path, n_seen, n_kept).
    The shard path is None when no events were produced for this pod5 file.
    """
    pod5_path_str, tmp_dir_str = args
    pod5_path = Path(pod5_path_str)
    (peak_distance, smooth_box, trim,
     resplit_std, resplit_distance, resplit_smooth,
     resplit_max_pieces, resplit_min_piece_len,
     resplit_bm_th, resplit_r2_th, norm) = _WORKER_PARAMS
    n_seen, n_kept, cols = _process_one_pod5(
        pod5_path, _WORKER_WINDOW,
        peak_distance=peak_distance, smooth_box=smooth_box, trim=trim,
        resplit_std=resplit_std,
        resplit_distance=resplit_distance,
        resplit_smooth=resplit_smooth,
        resplit_max_pieces=resplit_max_pieces,
        resplit_min_piece_len=resplit_min_piece_len,
        resplit_bm_th=resplit_bm_th,
        resplit_r2_th=resplit_r2_th,
        norm=norm)
    if not cols["read_id"]:
        return (pod5_path.name, None, n_seen, n_kept)
    shard = Path(tmp_dir_str) / f"shard.{pod5_path.stem}.parquet"
    pq.write_table(pa.table(cols, schema=EVENTS_SCHEMA), shard,
                   compression="snappy")
    return (pod5_path.name, str(shard), n_seen, n_kept)


def segment_pod5_folder(
    pod5_folder: str | Path,
    extract_csv: str | Path,
    output: str | Path,
    *,
    peak_distance: int = 10,
    smooth_box: int = 3,
    trim: float = DEFAULT_TRIM,
    resplit_std: float = 0.0,
    resplit_distance: int = 5,
    resplit_smooth: int = 1,
    resplit_max_pieces: int = 2,
    resplit_min_piece_len: int = 4,
    resplit_bm_th: float = 1.5,
    resplit_r2_th: float = 0.70,
    norm: str = DEFAULT_NORM,
    force: bool = False,
    n_workers: int = 1,
    drop_dup_reads: bool = True,
) -> Path:
    """Stream per-read subevents from pod5 folder filtered by extract_csv whitelist.

    Returns the output parquet path.

    With ``n_workers == 1`` (default) processes pod5 files sequentially and
    streams rows into a single ``pyarrow.ParquetWriter``. With ``n_workers > 1``
    each pod5 file is processed by a worker that writes its own shard parquet
    in a temp dir; the main process concatenates the shards into the final
    ``subevents.parquet`` (also streaming, no full load into RAM). Linux fork
    start method shares the whitelist dict zero-copy across workers.

    The output parquet records ``trim`` and ``segshape_version`` in
    file-level metadata so downstream readers can detect the convention."""
    pod5_folder = Path(pod5_folder).resolve()
    extract_csv = Path(extract_csv).resolve()
    output = Path(output).resolve()

    if not pod5_folder.is_dir():
        raise NotADirectoryError(pod5_folder)
    if not extract_csv.is_file():
        raise FileNotFoundError(f"extract CSV not found: {extract_csv}")
    if output.exists() and not force:
        raise FileExistsError(
            f"{output} already exists; pass force=True / --force to overwrite")
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if not (0.0 <= trim < 0.5):
        raise ValueError(f"--trim must be in [0, 0.5); got {trim!r}")
    if resplit_std < 0:
        raise ValueError(f"--resplit-std must be >= 0; got {resplit_std!r}")
    if resplit_distance < 2:
        raise ValueError(
            f"--resplit-distance must be >= 2; got {resplit_distance!r}")
    if resplit_smooth < 1:
        raise ValueError(
            f"--resplit-smooth must be >= 1; got {resplit_smooth!r}")
    if resplit_max_pieces < 2:
        raise ValueError(
            f"--resplit-max-pieces must be >= 2 (need at least 1 new "
            f"boundary to make a meaningful split); got "
            f"{resplit_max_pieces!r}")
    if resplit_min_piece_len < 2:
        raise ValueError(
            f"--resplit-min-piece-len must be >= 2 (single-sample "
            f"pieces have undefined std); got {resplit_min_piece_len!r}")
    if resplit_bm_th < 0:
        raise ValueError(
            f"--resplit-bm-th must be >= 0; got {resplit_bm_th!r}")
    if not (0.0 <= resplit_r2_th <= 1.0):
        raise ValueError(
            f"--resplit-r2-th must be in [0, 1]; got {resplit_r2_th!r}")
    if norm not in NORM_CHOICES:
        raise ValueError(
            f"--norm must be one of {NORM_CHOICES}; got {norm!r}")

    pod5_files = sorted(pod5_folder.glob("*.pod5"))
    if not pod5_files:
        raise FileNotFoundError(f"no *.pod5 in {pod5_folder}")

    print(f"pod5 folder: {pod5_folder}  ({len(pod5_files)} files)",
          file=sys.stderr)
    print(f"extract csv: {extract_csv}", file=sys.stderr)
    print(f"output     : {output}", file=sys.stderr)
    print(f"params     : peak_distance={peak_distance} smooth_box={smooth_box} "
          f"trim={trim} resplit_std={resplit_std} "
          f"(distance={resplit_distance} smooth={resplit_smooth} "
          f"max_pieces={resplit_max_pieces} "
          f"min_piece_len={resplit_min_piece_len}) "
          f"norm={norm} n_workers={n_workers}",
          file=sys.stderr)

    window = _load_whitelist(extract_csv)
    print(f"  whitelist: {len(window):,} reads", file=sys.stderr)

    output.parent.mkdir(parents=True, exist_ok=True)

    if n_workers == 1:
        n_reads, n_events, n_seen, n_dup_dropped = _segment_sequential(
            pod5_files, window, output,
            peak_distance=peak_distance, smooth_box=smooth_box, trim=trim,
            resplit_std=resplit_std,
            resplit_distance=resplit_distance,
            resplit_smooth=resplit_smooth,
            resplit_max_pieces=resplit_max_pieces,
            resplit_min_piece_len=resplit_min_piece_len,
            resplit_bm_th=resplit_bm_th,
            resplit_r2_th=resplit_r2_th,
            norm=norm,
            drop_dup_reads=drop_dup_reads)
    else:
        n_reads, n_events, n_seen, n_dup_dropped = _segment_parallel(
            pod5_files, window, output,
            peak_distance=peak_distance, smooth_box=smooth_box, trim=trim,
            resplit_std=resplit_std,
            resplit_distance=resplit_distance,
            resplit_smooth=resplit_smooth,
            resplit_max_pieces=resplit_max_pieces,
            resplit_min_piece_len=resplit_min_piece_len,
            resplit_bm_th=resplit_bm_th,
            resplit_r2_th=resplit_r2_th,
            norm=norm,
            n_workers=n_workers,
            drop_dup_reads=drop_dup_reads)

    if n_reads == 0:
        # writer closed an empty file; remove it so retries are clean.
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"no events produced. {len(window)} reads in whitelist, "
            f"{n_seen} reads scanned in pod5 folder; intersection empty. "
            f"Check that the extract CSV matches this pod5 folder.")

    print(
        f"wrote {output}  ({n_reads:,}/{n_seen:,} reads matched whitelist, "
        f"{n_events:,} events, "
        f"{n_events / max(1, n_reads):.1f} events/read"
        f"{f', {n_dup_dropped:,} duplicate read_id occurrences dropped' if n_dup_dropped else ''})",
        file=sys.stderr,
    )
    return output


class _NoOpPbar:
    """Fallback when tqdm is not installed and the caller wants a manual
    progress bar (kwargs-only, no iterable)."""
    def update(self, n=1): pass
    def close(self): pass


def _tqdm(iterable=None, **kw):
    """Wrap an iterable with tqdm (iterator form), OR open a manual progress
    bar (no iterable, only kwargs). Falls back to no-op if tqdm is missing."""
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable if iterable is not None else _NoOpPbar()
    return tqdm(iterable, **kw) if iterable is not None else tqdm(**kw)


def _schema_with_metadata(trim: float, resplit_std: float,
                          resplit_max_pieces: int = 2,
                          resplit_min_piece_len: int = 4,
                          resplit_bm_th: float = 1.5,
                          resplit_r2_th: float = 0.70,
                          norm: str = DEFAULT_NORM) -> pa.Schema:
    """EVENTS_SCHEMA tagged with the file-level metadata recorded under each
    output parquet so downstream readers can verify the trim / norm
    convention. ``norm`` records the per-read signal normalization scheme
    applied before segmentation:

      - ``"none"``: raw pA. ``mean_pa`` / ``std_pa`` columns are in pA.
      - ``"med-mad"``: per-read ``(sig - median) / (1.4826 × MAD)`` applied
        to the full pA signal before clipping/find_peaks. ``mean_pa`` /
        ``std_pa`` are in normalized σ-equivalent units (compatible with
        ONT v1 normalized k-mer level tables).
    """
    return EVENTS_SCHEMA.with_metadata({
        b"trim":                  str(trim).encode(),
        b"resplit_std":           str(resplit_std).encode(),
        b"resplit_max_pieces":    str(resplit_max_pieces).encode(),
        b"resplit_min_piece_len": str(resplit_min_piece_len).encode(),
        b"resplit_bm_th":         str(resplit_bm_th).encode(),
        b"resplit_r2_th":         str(resplit_r2_th).encode(),
        b"norm":                  norm.encode(),
        b"segshape_version":      __version__.encode(),
        b"segshape_tool":         b"segshape segment",
        b"clip_range":            b"[called_start, called_end)",
    })


def _segment_sequential(
    pod5_files: list[Path],
    window: dict[str, tuple[int, int]],
    output: Path,
    *,
    peak_distance: int,
    smooth_box: int,
    trim: float,
    resplit_std: float = 0.0,
    resplit_distance: int = 5,
    resplit_smooth: int = 1,
    resplit_max_pieces: int = 2,
    resplit_min_piece_len: int = 4,
    resplit_bm_th: float = 1.5,
    resplit_r2_th: float = 0.70,
    norm: str = DEFAULT_NORM,
    drop_dup_reads: bool = True,
) -> tuple[int, int, int, int]:
    # The same read_id can appear in >1 pod5 file (chunked split-write).
    # When drop_dup_reads=True, only the first occurrence is written; the
    # second copy's rows are dropped (n_dup_dropped counts those rows).
    unique_ids: set[str] = set()
    n_events = 0
    n_seen_total = 0
    n_dup_dropped = 0
    schema = _schema_with_metadata(trim, resplit_std, resplit_max_pieces,
                                   resplit_min_piece_len,
                                   resplit_bm_th, resplit_r2_th,
                                   norm=norm)
    with pq.ParquetWriter(output, schema, compression="snappy") as writer:
        for p in _tqdm(pod5_files, desc="segmenting", unit="pod5"):
            n_seen, _n_kept, cols = _process_one_pod5(
                p, window, peak_distance=peak_distance,
                smooth_box=smooth_box, trim=trim,
                resplit_std=resplit_std,
                resplit_distance=resplit_distance,
                resplit_smooth=resplit_smooth,
                resplit_max_pieces=resplit_max_pieces,
                resplit_min_piece_len=resplit_min_piece_len,
                resplit_bm_th=resplit_bm_th,
                resplit_r2_th=resplit_r2_th,
                norm=norm)
            n_seen_total += n_seen
            if not cols["read_id"]:
                continue
            if drop_dup_reads:
                keep = [rid not in unique_ids for rid in cols["read_id"]]
                if not all(keep):
                    n_dup_dropped += sum(1 for k in keep if not k)
                    cols = {k: [v for v, m in zip(cols[k], keep) if m]
                            for k in cols}
                    if not cols["read_id"]:
                        continue
            writer.write_table(pa.table(cols, schema=schema))
            unique_ids.update(cols["read_id"])
            n_events += len(cols["read_id"])
    return len(unique_ids), n_events, n_seen_total, n_dup_dropped


def _segment_parallel(
    pod5_files: list[Path],
    window: dict[str, tuple[int, int]],
    output: Path,
    *,
    peak_distance: int,
    smooth_box: int,
    trim: float,
    resplit_std: float = 0.0,
    resplit_distance: int = 5,
    resplit_smooth: int = 1,
    resplit_max_pieces: int = 2,
    resplit_min_piece_len: int = 4,
    resplit_bm_th: float = 1.5,
    resplit_r2_th: float = 0.70,
    norm: str = DEFAULT_NORM,
    n_workers: int,
    drop_dup_reads: bool = True,
) -> tuple[int, int, int, int]:
    """Parallel: each pod5 file → worker → shard parquet → main concats."""
    global _WORKER_WINDOW, _WORKER_PARAMS
    _WORKER_WINDOW = window
    _WORKER_PARAMS = (peak_distance, smooth_box, trim,
                      resplit_std, resplit_distance, resplit_smooth,
                      resplit_max_pieces, resplit_min_piece_len,
                      resplit_bm_th, resplit_r2_th, norm)

    tmp_dir = Path(tempfile.mkdtemp(prefix=".tmp_seg_shards_",
                                    dir=output.parent))
    # The same read_id can appear in >1 pod5 file (chunked split-write).
    # When drop_dup_reads=True, the concat step keeps only the first
    # occurrence of each read_id across shards and drops the rest.
    unique_ids: set[str] = set()
    n_events = 0
    n_seen_total = 0
    n_dup_dropped = 0
    shards: list[str] = []
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            tasks = [(str(p), str(tmp_dir)) for p in pod5_files]
            pbar = _tqdm(total=len(tasks), desc=f"segmenting (j={n_workers})",
                         unit="pod5")
            for name, shard, n_seen, _n_kept in pool.imap_unordered(
                    _worker_segment, tasks):
                if hasattr(pbar, "update"):
                    pbar.update(1)
                n_seen_total += n_seen
                if shard is not None:
                    shards.append(shard)
            if hasattr(pbar, "close"):
                pbar.close()

        # Streaming concat: read each shard, write to final parquet.
        # Sort by name for deterministic row order even with imap_unordered.
        schema = _schema_with_metadata(trim, resplit_std, resplit_max_pieces,
                                       resplit_min_piece_len,
                                       resplit_bm_th, resplit_r2_th,
                                       norm=norm)
        with pq.ParquetWriter(output, schema,
                              compression="snappy") as writer:
            for shard in _tqdm(sorted(shards), desc="concat shards",
                               unit="shard"):
                pf = pq.ParquetFile(shard)
                for batch in pf.iter_batches(batch_size=_BATCH_SIZE):
                    if drop_dup_reads:
                        rids = batch.column("read_id").to_pylist()
                        keep = [rid not in unique_ids for rid in rids]
                        if not all(keep):
                            n_dup_dropped += sum(1 for k in keep if not k)
                            batch = batch.filter(pa.array(keep))
                            if batch.num_rows == 0:
                                continue
                    writer.write_batch(batch)
                    n_events += batch.num_rows
                    unique_ids.update(batch.column("read_id").to_pylist())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # Reset module globals so library use (or test re-runs) start clean.
        _WORKER_WINDOW = None
        _WORKER_PARAMS = (10, 3, DEFAULT_TRIM, 0.15, 5, 1, 2, 4, 1.5, 0.70,
                          DEFAULT_NORM)

    return len(unique_ids), n_events, n_seen_total, n_dup_dropped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    """Workflow vs explicit-path resolution. Each of pod5-folder /
    extract-csv / output can be overridden individually; whatever is left
    blank is filled from --root-dir/--dataset/--sample defaults."""
    pod5_folder = args.pod5_folder
    extract_csv = args.extract_csv
    output = args.output

    if pod5_folder and extract_csv and output:
        return pod5_folder, extract_csv, output

    if not (args.root_dir and args.dataset and args.sample):
        missing = [name for name, val in [
            ("--pod5-folder", pod5_folder),
            ("--extract-csv", extract_csv),
            ("--output", output),
        ] if not val]
        raise SystemExit(
            f"ERROR: missing path(s) {missing}. Either pass them explicitly, "
            f"or pass --root-dir/--dataset/--sample to derive defaults.")

    sample_dir = Path(args.root_dir) / "datasets" / args.dataset / args.sample
    pod5_folder = pod5_folder or str(sample_dir / "1_raw_signal" / "pod5")
    extract_csv = extract_csv or str(sample_dir / "3_alignment" / DEFAULT_EXTRACT_CSV)
    # When --norm med-mad and no explicit --output, default to
    # subevents.norm.parquet so the normalized and raw-pA outputs sit
    # side by side and are not confused with each other.
    default_basename = (DEFAULT_SUBEVENTS_NORM_PARQUET
                        if getattr(args, "norm", DEFAULT_NORM) == "med-mad"
                        else DEFAULT_SUBEVENTS_PARQUET)
    output = output or str(sample_dir / "3_alignment" / default_basename)
    return pod5_folder, extract_csv, output


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pod5-folder",
        help="path to a folder of *.pod5 files "
             "(default: <root>/datasets/<DATASET>/<SAMPLE>/1_raw_signal/pod5)",
    )
    parser.add_argument(
        "--extract-csv",
        help="path to dorado.extract_mv.csv (default: "
             "<root>/datasets/<DATASET>/<SAMPLE>/3_alignment/dorado.extract_mv.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        help="output subevents.parquet path (default: "
             "<root>/datasets/<DATASET>/<SAMPLE>/3_alignment/subevents.parquet)",
    )
    parser.add_argument("--root-dir",
                        help="project root for path derivation")
    parser.add_argument("--dataset", help="dataset name")
    parser.add_argument("--sample", help="sample name (control | treated)")
    parser.add_argument(
        "--peak-distance", type=int, default=10, metavar="N",
        help="scipy.signal.find_peaks distance parameter in samples "
             "(default: 10).",
    )
    parser.add_argument(
        "--smooth", type=int, default=3, metavar="N",
        help="box-car smoothing width applied to |slope| before find_peaks "
             "(default: 3).",
    )
    parser.add_argument(
        "--trim", type=float, default=DEFAULT_TRIM, metavar="P",
        help=f"per-subevent sample trim proportion on each end "
             f"(default: {DEFAULT_TRIM}). Both mean_pa and std_pa are "
             f"computed on the same trimmed sample set, so the std drops "
             f"the boundary-transition jitter that find_peaks puts at the "
             f"event edges. P=0 disables trimming (raw seg.mean / seg.std). "
             f"Allowed range: [0, 0.5).",
    )
    parser.add_argument(
        "--resplit-std", type=float, default=None, metavar="STD",
        help="re-split any subevent whose **raw** std exceeds this "
             "threshold via a second find_peaks pass — catches fast "
             "within-subevent transitions the first pass merged. Units "
             "follow --norm: sigma when --norm med-mad, pA when --norm "
             "none. The default is norm-aware: 0.15 (med-mad; ~2.5-3 "
             "raw pA) or 3.0 (none). NOTE the threshold is compared "
             "against RAW std, while the emitted `std_pa` column is "
             "TRIMMED std (per --trim); the two scales differ ~5-10 %% "
             "so a parquet `std_pa = 2.9` can still trigger a 3.0 "
             "threshold. Pass 0 to disable re-split.",
    )
    parser.add_argument(
        "--resplit-distance", type=int, default=5, metavar="N",
        help="find_peaks distance for the re-split pass; smaller than the "
             "first pass to catch finer transitions (default: 5).",
    )
    parser.add_argument(
        "--resplit-smooth", type=int, default=1, metavar="N",
        help="box-car smoothing for the re-split pass; smaller than the "
             "first pass to preserve sharp transitions (default: 1).",
    )
    parser.add_argument(
        "--resplit-max-pieces", type=int, default=2, metavar="N",
        help="cap on how many pieces a single suspicious subevent can be "
             "split into (default: 2 = exactly 1 extra cut). Sub-peaks "
             "ranked by slope amplitude — only the strongest are kept. "
             "Default lowered from 3 to 2 in 2026-05 after simulator "
             "audit showed max_pieces=3 gave no measurable gain.",
    )
    parser.add_argument(
        "--resplit-min-piece-len", type=int, default=4, metavar="N",
        help="minimum samples per piece in a re-split (default: 4). A "
             "subevent shorter than 2*N samples is never attempted, and "
             "candidate sub-boundaries that would yield a piece shorter "
             "than N are discarded. Lowered from 6 in 2026-05 to allow "
             "splitting short bimodal subevents (10-11 samples) that the "
             "old floor blocked.",
    )
    parser.add_argument(
        "--resplit-bm-th", type=float, default=1.5, metavar="X",
        help="bimodality score threshold (default 1.5). A subevent is "
             "considered for re-split only if its first half mean and "
             "second half mean differ by ≥ X × pooled within-half std "
             "(Welch-t-like). Excludes single-kmer noisy subevents that "
             "would otherwise pass on std alone. Set 0 to disable the "
             "shape gate (legacy std-only behaviour).",
    )
    parser.add_argument(
        "--resplit-r2-th", type=float, default=0.70, metavar="X",
        help="linearity R² ceiling (default 0.70). A subevent whose "
             "OLS fit `signal ~ sample_idx` has R² > X is treated as a "
             "transition ramp and skipped — splitting a ramp would "
             "create phantom transition fragments. Set 1.0 to disable.",
    )
    parser.add_argument(
        "--norm", choices=NORM_CHOICES, default=DEFAULT_NORM,
        help=f"per-read signal normalization applied BEFORE find_peaks "
             f"and trimmed-mean computation (default: {DEFAULT_NORM!r}). "
             f"'none' = raw pA (mean_pa/std_pa columns are in pA, "
             f"compatible with the legacy RNA002 pA k-mer model). "
             f"'med-mad' = bonito-style median/MAD z-score "
             f"`(sig - median) / (1.4826 × MAD)` on the full per-read "
             f"signal; mean_pa/std_pa columns become unitless σ-equivalent "
             f"and pair with ONT v1 normalized k-mer models. When "
             f"--norm med-mad is set and --output is not specified, the "
             f"default output basename becomes "
             f"'{DEFAULT_SUBEVENTS_NORM_PARQUET}' so it sits beside the "
             f"raw-pA '{DEFAULT_SUBEVENTS_PARQUET}'. Peak positions are "
             f"scale-invariant (find_peaks here uses only `distance`), so "
             f"segmentation boundaries are bit-identical to the raw case; "
             f"only the emitted statistics change unit.",
    )
    parser.add_argument(
        "-j", "--n-workers", type=int, default=1, metavar="N",
        help="number of pod5-file-level worker processes "
             "(default: 1 = sequential streaming). Each worker writes a "
             "shard parquet; the main process concatenates shards into the "
             "final subevents.parquet. SLURM users: pass -j ${SLURM_CPUS_PER_TASK}.",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="overwrite an existing subevents.parquet",
    )
    parser.add_argument(
        "--drop-dup-reads", action=argparse.BooleanOptionalAction, default=True,
        help="keep only the first occurrence of each read_id across pod5 "
             "files (default: on). The same read_id can appear in >1 pod5 "
             "when MinKNOW chunk-writes a read twice across a restart "
             "(e.g. tetra/treated: 299/41,328 reads = 0.72%%); without "
             "dedup the parquet ends up with N copies of every event for "
             "those reads. Pass --no-drop-dup-reads to keep all copies.",
    )


def run(args: argparse.Namespace) -> int:
    pod5_folder, extract_csv, output = _resolve_paths(args)
    # --resplit-std lives in the same coordinate as --norm, so its default is
    # norm-aware: 0.15 sigma under med-mad (production default), 3.0 pA under
    # none. An explicit --resplit-std (including 0 to disable) always wins.
    resplit_std = args.resplit_std
    if resplit_std is None:
        resplit_std = 0.15 if args.norm == "med-mad" else 3.0
    try:
        segment_pod5_folder(
            pod5_folder, extract_csv, output,
            peak_distance=args.peak_distance,
            smooth_box=args.smooth,
            trim=args.trim,
            resplit_std=resplit_std,
            resplit_distance=args.resplit_distance,
            resplit_smooth=args.resplit_smooth,
            resplit_max_pieces=args.resplit_max_pieces,
            resplit_min_piece_len=args.resplit_min_piece_len,
            resplit_bm_th=args.resplit_bm_th,
            resplit_r2_th=args.resplit_r2_th,
            norm=args.norm,
            force=args.force,
            n_workers=args.n_workers,
            drop_dup_reads=args.drop_dup_reads,
        )
    except (FileNotFoundError, FileExistsError, NotADirectoryError,
            ValueError, RuntimeError) as e:
        raise SystemExit(f"ERROR [{type(e).__name__}]: {e}") from None
    return 0
