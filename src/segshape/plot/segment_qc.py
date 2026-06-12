"""Audit-plot ``subevents.parquet`` overlaid on raw pod5 signal.

For each (dataset, sample), pick (or specify) one read; zoom into a
window of its pod5 signal trace; overlay all subevent boundaries +
mean_pa horizontal segments. Useful for sanity-checking segment output:
peak placement, mean_pa level, std_pa width.

Window selection:

  - Default mode (no explicit zoom flags):
      [mv_trans_start - --zoom-pre-pad, mv_trans_start + --zoom-samples)
      ``--zoom-pre-pad`` shows pre-mv_trans_start signal context (polyA
      tail / soft-clip — these samples have no subevent because segment
      clips to mv_trans_start, so only signal_pa is drawn there).
  - Explicit mode (``--zoom-start S --zoom-end E``):
      absolute pod5 sample positions; overrides the default mode.

Modes of invocation:

  - ``--dataset / --sample`` together: render that one combo.
  - Both omitted: batch-render all (dataset, sample) under
    ``<root-dir>/datasets/``.
  - Adding ``--read-id`` picks a specific read instead of the random
    eligible one (requires ``--dataset / --sample``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_ZOOM_SAMPLES = 500    # ~15 RNA bases ≈ ~45 events at peak_distance=10
DEFAULT_ZOOM_PRE_PAD = 50
DEFAULT_SEED = 42


# --- helpers ----------------------------------------------------------------

def _pick_random_read(extract_csv: Path, rng: np.random.Generator
                      ) -> Optional[pd.Series]:
    """Random eligible read with mv_trans_start >= 0 and ref_cov >= 0.5
    (loose filter — just avoid totally pathological reads)."""
    df = pd.read_csv(extract_csv, comment="#")
    df["ref_cov"] = (df.ref_end - df.ref_start) / df.ref_len
    pool = df[(df.mv_trans_start >= 0) & (df.ref_cov >= 0.5)]
    if len(pool) == 0:
        return None
    idx = rng.integers(0, len(pool))
    return pool.iloc[int(idx)]


def _pick_random_reads(extract_csv: Path, n: int,
                        rng: np.random.Generator) -> list[pd.Series]:
    """Pick `n` distinct eligible reads (without replacement). Returns
    fewer than `n` if the pool is smaller. Returns [] when no reads
    pass the filter."""
    df = pd.read_csv(extract_csv, comment="#")
    df["ref_cov"] = (df.ref_end - df.ref_start) / df.ref_len
    pool = df[(df.mv_trans_start >= 0) & (df.ref_cov >= 0.5)]
    if len(pool) == 0:
        return []
    n = min(n, len(pool))
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool.iloc[int(i)] for i in idx]


def _random_zoom_from_middle(s_lo: int, s_hi: int, signal_len: int,
                              zoom_samples: int, margin: int,
                              rng: np.random.Generator
                              ) -> Optional[tuple[int, int]]:
    """Pick a random zoom window of width `zoom_samples` whose center
    falls inside [s_lo + margin + W/2, s_hi - margin - W/2]. Returns
    None when the aligned region is too short to host such a window."""
    half = zoom_samples // 2
    c_lo = s_lo + margin + half
    c_hi = s_hi - margin - half
    if c_hi <= c_lo:
        return None
    center = int(rng.integers(c_lo, c_hi + 1))
    abs_lo = max(0, center - half)
    abs_hi = min(signal_len, abs_lo + zoom_samples)
    if abs_hi <= abs_lo:
        return None
    return abs_lo, abs_hi


def _signal_for_read(pod5_folder: Path, read_id: str,
                     pod5_index_path: Path) -> np.ndarray:
    import pod5
    idx = pd.read_parquet(pod5_index_path)
    fname = idx[idx.read_id == read_id]["filename"].iloc[0]
    with pod5.Reader(pod5_folder / fname) as r:
        rec = next(r.reads(selection=[read_id]))
        return np.asarray(rec.signal_pa, dtype=np.float32)


def _subevents_for_read(subevents_path: Path, read_id: str) -> pd.DataFrame:
    """Pyarrow filter pushdown to load only one read's rows."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(subevents_path)
    cols = ["read_id", "event_idx", "start_sample", "end_sample",
            "mean_pa", "std_pa"]
    rows = []
    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        df = batch.to_pandas()
        m = df["read_id"] == read_id
        if m.any():
            rows.append(df[m])
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.concat(rows, ignore_index=True)
    return df.sort_values("event_idx").reset_index(drop=True)


def _trim_metadata(subevents_path: Path) -> str:
    """File-level ``trim`` value (segshape segment writes it). '?' on miss."""
    import pyarrow.parquet as pq
    try:
        meta = pq.ParquetFile(subevents_path).schema_arrow.metadata
        if meta and b"trim" in meta:
            return meta[b"trim"].decode()
    except Exception:
        pass
    return "?"


def _norm_metadata(subevents_path: Path) -> str:
    """File-level ``norm`` value ('none' default; 'med-mad' when segment
    applied bonito-style per-read normalization). Older parquets without
    the field return 'none' so legacy callers stay raw-pA."""
    import pyarrow.parquet as pq
    try:
        meta = pq.ParquetFile(subevents_path).schema_arrow.metadata
        if meta and b"norm" in meta:
            return meta[b"norm"].decode()
    except Exception:
        pass
    return "none"


def _plot_panel(ax, signal_pa, subev, *, mv_trans_start, x_window,
                plot_std=False, norm: str = "none"):
    abs_lo, abs_hi = x_window
    abs_hi = min(abs_hi, len(signal_pa))
    x = np.arange(abs_lo, abs_hi)
    ax.plot(x, signal_pa[abs_lo:abs_hi],
            color="lightgray", linewidth=1.0,
            label="Raw signal (pA)", zorder=1)
    ax.set_xlim(abs_lo, abs_hi)

    in_window = subev[(subev["end_sample"] > abs_lo)
                      & (subev["start_sample"] < abs_hi)]
    drew_mean_legend = False
    drew_std_legend = False
    for _, r in in_window.iterrows():
        s, e = int(r["start_sample"]), int(r["end_sample"])
        s_c, e_c = max(s, abs_lo), min(e, abs_hi)
        if abs_lo <= s < abs_hi:
            ax.axvline(s, color="lightsteelblue", alpha=0.5,
                       linewidth=0.4, zorder=2)
        m = float(r["mean_pa"])
        unit = "σ" if norm == "med-mad" else "pA"
        if plot_std and "std_pa" in r and np.isfinite(r["std_pa"]):
            sd = float(r["std_pa"])
            kw = dict(color="C0", alpha=0.18, zorder=3, linewidth=0)
            if not drew_std_legend:
                kw["label"] = f"Subevent mean ± std ({unit})"
                drew_std_legend = True
            ax.fill_between([s_c, e_c], m - sd, m + sd, **kw)
        kw = dict(color="C0", linewidth=2.0, alpha=0.95, zorder=4)
        if not drew_mean_legend:
            kw["label"] = f"Subevent mean ({unit})"
            drew_mean_legend = True
        ax.hlines(m, s_c, e_c, **kw)

    # Only label mv_trans_start in the legend when it actually falls
    # inside the visible window — otherwise the legend entry is a lie
    # (the line is clipped off-screen).
    mv_kw = dict(color="firebrick", linestyle="--",
                 linewidth=0.8, alpha=0.7, zorder=3)
    if abs_lo <= mv_trans_start < abs_hi:
        mv_kw["label"] = "mv_trans_start"
    ax.axvline(mv_trans_start, **mv_kw)
    ax.set_xlabel("Absolute index of raw pod5 sample")
    if norm == "med-mad":
        ax.set_ylabel("Normalized signal "
                      "((sig − median)/(1.4826·MAD), σ-units)")
    else:
        ax.set_ylabel("Current signal (pA)")
    # Y-axis grid only (horizontal lines for pA reference). Vertical
    # gridlines removed — they collided visually with the cool-blue
    # subevent boundary axvlines and made boundaries hard to see.
    ax.grid(axis="y", which="major", color="#d4b48a", linestyle=":",
            linewidth=0.5, alpha=0.55)


def _resolve_window(s_lo: int, s_hi: int, signal_len: int,
                    *, zoom_start: Optional[int], zoom_end: Optional[int],
                    zoom_pre_pad: int, zoom_samples: int,
                    zoom_anchor: str
                    ) -> Optional[tuple[int, int]]:
    """Pick the window. Returns (abs_lo, abs_hi) or None on empty.

    Default mode is anchored on one of the mv_trans bounds:
      - 'start' (default): [s_lo - pre_pad, s_lo + samples)
      - 'mid':             [mid - samples//2, mid + samples//2)
                           where mid = (s_lo + s_hi) // 2 — useful for
                           scanning the body of the aligned region
                           rather than just the polyA-adjacent head.
      - 'end':             [s_hi - samples, s_hi + pre_pad)

    Explicit ``--zoom-start S --zoom-end E`` overrides this entirely.
    """
    if zoom_start is not None or zoom_end is not None:
        abs_lo = max(0, int(zoom_start) if zoom_start is not None else 0)
        abs_hi = min(signal_len,
                     int(zoom_end) if zoom_end is not None else signal_len)
    elif zoom_anchor == 'start':
        abs_lo = max(0, s_lo - zoom_pre_pad)
        abs_hi = min(signal_len, s_lo + zoom_samples)
    elif zoom_anchor == 'mid':
        mid = (s_lo + s_hi) // 2
        half = zoom_samples // 2
        abs_lo = max(0, mid - half)
        abs_hi = min(signal_len, mid + half)
    elif zoom_anchor == 'end':
        abs_lo = max(0, s_hi - zoom_samples)
        abs_hi = min(signal_len, s_hi + zoom_pre_pad)
    else:
        raise ValueError(f"unknown zoom_anchor: {zoom_anchor!r}")
    if abs_hi <= abs_lo:
        return None
    return abs_lo, abs_hi


# --- core render ------------------------------------------------------------

def render(label: str, sample_dir: Path, out_dir: Path,
           rng: np.random.Generator, *,
           read_id_override: Optional[str] = None,
           zoom_start: Optional[int] = None,
           zoom_end: Optional[int] = None,
           zoom_pre_pad: int = DEFAULT_ZOOM_PRE_PAD,
           zoom_samples: int = DEFAULT_ZOOM_SAMPLES,
           zoom_anchor: str = "start",
           subevents_file: str = "subevents.parquet",
           plot_std: bool = False) -> int:
    """Render one PNG for one (sample_dir, read). Returns 0 on success,
    non-zero on skip/error. Side-effect: writes PNG into ``out_dir``.

    ``subevents_file`` is the basename inside <sample_dir>/3_alignment/
    (e.g. 'subevents.parquet' or 'subevents.trimmed.parquet'). The
    chosen file is recorded in the panel title and the output filename.

    ``plot_std`` toggles a translucent ±std_pa band drawn behind each
    mean_pa segment, useful for comparing within-subevent noise across
    different segment configurations (untrimmed vs trimmed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pod5_folder = sample_dir / "1_raw_signal" / "pod5"
    pod5_index = sample_dir / "1_raw_signal" / "pod5.index"
    extract_csv = sample_dir / "3_alignment" / "dorado.extract_mv.csv"
    subevents_path = sample_dir / "3_alignment" / subevents_file
    for path, name in [(pod5_folder, "pod5/"), (pod5_index, "pod5.index"),
                       (extract_csv, "dorado.extract_mv.csv"),
                       (subevents_path, subevents_file)]:
        if not path.exists():
            print(f"  {label}: missing {name} at {path}, skipping")
            return 1

    if read_id_override is not None:
        df = pd.read_csv(extract_csv, comment="#")
        df["ref_cov"] = (df.ref_end - df.ref_start) / df.ref_len
        match = df[df.read_id == read_id_override]
        if len(match) == 0:
            print(f"  {label}: read_id {read_id_override[:8]}.. "
                  f"not in extract csv, skipping")
            return 1
        row = match.iloc[0]
    else:
        row = _pick_random_read(extract_csv, rng)
        if row is None:
            print(f"  {label}: no eligible read, skipping")
            return 1

    rid = str(row.read_id)
    s_lo = int(row.mv_trans_start)
    s_hi = int(row.mv_trans_end)
    full = _signal_for_read(pod5_folder, rid, pod5_index)
    # When the chosen subevents parquet was produced with --norm med-mad,
    # its mean_pa / std_pa columns are in σ-units. To keep the raw pA
    # signal and the subevent overlays in the same coordinate system, apply
    # the same per-read median-MAD transform here BEFORE the panel draws.
    # The transform exactly mirrors segment/events.py:med_mad_normalisation
    # (full signal stats, 1e-6 scale floor).
    norm = _norm_metadata(subevents_path)
    if norm == "med-mad":
        from segshape.segment.events import med_mad_normalisation
        shift, scale = med_mad_normalisation(full)
        full = ((full - shift) / scale).astype(np.float32)
    subev = _subevents_for_read(subevents_path, rid)
    if subev.empty:
        print(f"  {label}: read_id {rid[:8]}.. not in subevents.parquet, "
              f"skipping")
        return 1

    win = _resolve_window(s_lo, s_hi, len(full),
                          zoom_start=zoom_start, zoom_end=zoom_end,
                          zoom_pre_pad=zoom_pre_pad,
                          zoom_samples=zoom_samples,
                          zoom_anchor=zoom_anchor)
    if win is None:
        lo = zoom_start if zoom_start is not None else 'auto'
        hi = zoom_end if zoom_end is not None else 'auto'
        print(f"  {label}: empty window [{lo}, {hi}), skipping")
        return 1
    abs_lo, abs_hi = win

    n_in_window = ((subev["end_sample"] > abs_lo)
                   & (subev["start_sample"] < abs_hi)).sum()
    trim = _trim_metadata(subevents_path)
    title = (f"{label}  |  read_id={rid[:8]}...{rid[-4:]}  |  "
             f"seq_len={int(row.seq_len)} ref_cov={row.ref_cov:.2f}  |  "
             f"source={subevents_file} (trim={trim}, norm={norm})  |  "
             f"n_subevents_total={len(subev)}  |  "
             f"window=samples [{abs_lo}, {abs_hi}) "
             f"({n_in_window} subevents shown)"
             f"{'  |  +std band' if plot_std else ''}")

    fig, ax = plt.subplots(figsize=(16, 5))
    _plot_panel(ax, full, subev, mv_trans_start=s_lo,
                x_window=(abs_lo, abs_hi), plot_std=plot_std, norm=norm)
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=8)

    # Tag the subevents source in the filename when it's not the
    # default plain 'subevents.parquet', so untrimmed vs trimmed runs
    # produce distinct PNGs in the same out_dir.
    src_tag = ""
    if subevents_file != "subevents.parquet":
        # 'subevents.trimmed.parquet' -> '_trimmed'
        stem = subevents_file.removesuffix(".parquet")
        if stem.startswith("subevents."):
            src_tag = "_" + stem[len("subevents."):]
        else:
            src_tag = "_" + stem
    std_tag = "_std" if plot_std else ""

    out_dir.mkdir(parents=True, exist_ok=True)
    if (read_id_override is not None
            or zoom_start is not None or zoom_end is not None):
        # Filename format: <label>_<rid8>_<lo>_<hi>[<src_tag>][<std_tag>].png
        # where <label> = '<dataset>_<sample>'. Used for explicit reads
        # and for batch mode --reads-per-sample.
        out = out_dir / f"{label}_{rid[:8]}_{abs_lo}_{abs_hi}{src_tag}{std_tag}.png"
    elif zoom_anchor != "start":
        # Suffix non-default anchor in filename so 'start' / 'mid' / 'end'
        # batch runs of the same (dataset, sample) produce distinct PNGs.
        out = out_dir / f"{label}_{zoom_anchor}{src_tag}{std_tag}.png"
    else:
        out = out_dir / f"{label}{src_tag}{std_tag}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print(f"  {label}: n_subevents_total={len(subev):>5}  "
          f"in_window={n_in_window:>3}  →  {out.name}")
    return 0


# --- CLI --------------------------------------------------------------------

def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root-dir", required=True,
                   help="project root containing the datasets/ subtree.")
    p.add_argument("--dataset", default=None,
                   help="dataset name (e.g. bsub_16S). If omitted (and "
                        "--sample also omitted), batch-renders all "
                        "(dataset, sample) combos under <root>/datasets/.")
    p.add_argument("--sample", default=None, choices=["control", "treated"],
                   help="sample name. Must be paired with --dataset.")
    p.add_argument("--read-id", default=None,
                   help="specific read_id to render (requires --dataset/"
                        "--sample). Default: random eligible read.")
    p.add_argument("--out-dir", default=None,
                   help="output directory for PNGs. "
                        "Default: <root-dir>/figures/segment/")
    win = p.add_argument_group(
        "window selection",
        "Default mode: [mv_trans_start - pre_pad, mv_trans_start + samples). "
        "Explicit mode (overrides default): --zoom-start S --zoom-end E "
        "with absolute pod5 sample positions.")
    win.add_argument("--zoom-start", type=int, default=None,
                     help="absolute pod5 sample index for window start "
                          "(inclusive).")
    win.add_argument("--zoom-end", type=int, default=None,
                     help="absolute pod5 sample index for window end "
                          "(exclusive).")
    win.add_argument("--zoom-pre-pad", type=int, default=DEFAULT_ZOOM_PRE_PAD,
                     metavar="N",
                     help=f"samples shown before mv_trans_start in default "
                          f"mode (default {DEFAULT_ZOOM_PRE_PAD}).")
    win.add_argument("--zoom-samples", type=int, default=DEFAULT_ZOOM_SAMPLES,
                     metavar="M",
                     help=f"window length in samples in default mode "
                          f"(default {DEFAULT_ZOOM_SAMPLES}; "
                          f"~15 RNA bases ≈ ~45 events).")
    win.add_argument("--zoom-anchor", choices=["start", "mid", "end"],
                     default="start",
                     help="default-mode anchor: 'start' (around "
                          "mv_trans_start, default), 'mid' (around the "
                          "midpoint of the aligned region), 'end' "
                          "(around mv_trans_end). Ignored if "
                          "--zoom-start/--zoom-end is given.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"random seed for read pick (default {DEFAULT_SEED}).")
    p.add_argument("--reads-per-sample", type=int, default=1, metavar="N",
                   help="batch-mode only: how many reads to render per "
                        "(dataset, sample). Default 1. When N > 1 each "
                        "read also gets a random zoom window centered "
                        "inside its aligned region (width=--zoom-samples; "
                        "see --zoom-margin), and the filename becomes "
                        "<dataset>_<sample>_<rid8>_<lo>_<hi>.png.")
    p.add_argument("--zoom-margin", type=int, default=100, metavar="M",
                   help="batch-mode --reads-per-sample only: minimum "
                        "samples of slack from each end of the aligned "
                        "region when picking the random zoom center. "
                        "Default 100. Picked center must satisfy "
                        "[s_lo + M + W/2, s_hi - M - W/2].")
    p.add_argument("--subevents-file", default="subevents.parquet",
                   help="basename inside <sample_dir>/3_alignment/. "
                        "Default 'subevents.parquet'. Set to "
                        "'subevents.trimmed.parquet' to plot the trimmed "
                        "segment output. Recorded in the title and the "
                        "output filename so untrimmed vs trimmed runs "
                        "produce distinct PNGs in the same out_dir.")
    p.add_argument("--plot-std", action="store_true",
                   help="draw a translucent ±std_pa band behind each "
                        "subevent's mean_pa segment. Default off. The "
                        "filename gets a '_std' suffix when on, so plain "
                        "vs +std runs of the same read coexist.")


def run(args: argparse.Namespace) -> int:
    root_dir = Path(args.root_dir).resolve()
    datasets_dir = root_dir / "datasets"
    out_dir = (Path(args.out_dir) if args.out_dir
               else root_dir / "figures" / "segment")

    targeted = args.dataset is not None or args.sample is not None
    if targeted and (args.dataset is None or args.sample is None):
        print("error: --dataset and --sample must be given together",
              file=sys.stderr)
        return 2
    if not targeted and (args.read_id is not None
                         or args.zoom_start is not None
                         or args.zoom_end is not None):
        print("error: --read-id / --zoom-start / --zoom-end require "
              "--dataset and --sample", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)

    if targeted:
        d = datasets_dir / args.dataset / args.sample
        if not d.is_dir():
            print(f"error: not a directory: {d}", file=sys.stderr)
            return 2
        label = f"{args.dataset}_{args.sample}"
        print(f"rendering 1 sample → {out_dir}")
        return render(label, d, out_dir, rng,
                      read_id_override=args.read_id,
                      zoom_start=args.zoom_start,
                      zoom_end=args.zoom_end,
                      zoom_pre_pad=args.zoom_pre_pad,
                      zoom_samples=args.zoom_samples,
                      zoom_anchor=args.zoom_anchor,
                      subevents_file=args.subevents_file,
                      plot_std=args.plot_std)

    # Batch mode (no targeted args)
    if not datasets_dir.is_dir():
        print(f"error: {datasets_dir} does not exist", file=sys.stderr)
        return 2
    samples = []
    for ds in sorted(datasets_dir.iterdir()):
        if not ds.is_dir():
            continue
        for sm in ("control", "treated"):
            d = ds / sm
            if d.is_dir():
                samples.append((ds.name, sm, d))
    n_per = max(1, args.reads_per_sample)
    print(f"rendering {len(samples)} samples × {n_per} reads/sample "
          f"→ {out_dir}")
    print(f"random seed = {args.seed} (per-sample independent stream)\n")
    rc = 0
    for ds_name, sm_name, d in samples:
        label = f"{ds_name}_{sm_name}"
        # Per-sample independent rng so adding/removing a sample doesn't
        # shift the random pick of the others.
        sub_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        try:
            if n_per == 1:
                render(label, d, out_dir, sub_rng,
                       zoom_pre_pad=args.zoom_pre_pad,
                       zoom_samples=args.zoom_samples,
                       zoom_anchor=args.zoom_anchor,
                       subevents_file=args.subevents_file,
                       plot_std=args.plot_std)
                continue

            # Multi-read mode: pick N distinct reads, give each a random
            # zoom window centered inside its aligned region. Render each
            # via the explicit (read_id + zoom_start + zoom_end) path,
            # which triggers the dataset_sample_rid8_lo_hi.png filename.
            extract_csv = d / "3_alignment" / "dorado.extract_mv.csv"
            if not extract_csv.exists():
                print(f"  {label}: missing dorado.extract_mv.csv, skipping")
                continue
            picks = _pick_random_reads(extract_csv, n_per, sub_rng)
            if not picks:
                print(f"  {label}: no eligible read, skipping")
                continue
            # Probe pod5 once per read to get signal length for the
            # zoom-fitting check. We could load it lazily inside render
            # but we need signal length here to choose a fitting zoom.
            for r in picks:
                rid = str(r.read_id)
                s_lo, s_hi = int(r.mv_trans_start), int(r.mv_trans_end)
                # Use the aligned-region length as a proxy for signal
                # length (mv_trans_end <= signal_len always); if
                # _random_zoom_from_middle accepts, we're safe.
                win = _random_zoom_from_middle(
                    s_lo, s_hi, signal_len=s_hi,
                    zoom_samples=args.zoom_samples,
                    margin=args.zoom_margin, rng=sub_rng,
                )
                if win is None:
                    print(f"  {label}: read {rid[:8]}.. aligned region "
                          f"too short for zoom, skipping read")
                    continue
                zlo, zhi = win
                render(label, d, out_dir, sub_rng,
                       read_id_override=rid,
                       zoom_start=zlo, zoom_end=zhi,
                       zoom_pre_pad=args.zoom_pre_pad,
                       zoom_samples=args.zoom_samples,
                       zoom_anchor=args.zoom_anchor,
                       subevents_file=args.subevents_file,
                       plot_std=args.plot_std)
        except Exception as e:
            import traceback
            print(f"  {label}: ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            rc = 1
    return rc
