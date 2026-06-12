"""Plot the dorado mv table for one read.

Visualises a single read's raw signal with the basecaller frame and mv
table overlaid:

  - The full signal as a thin line.
  - Per-region shading along the time axis:
        [0, ts)              trimmed pre-window           (gray)
        [ts, mv_trans_start) polyA + 5' soft clip          (yellow)
        [mv_trans_start, mv_trans_end)  aligned bases     (no shade)
        [mv_trans_end, ns)   3' soft clip / trailing      (yellow)
        [ns, end of signal)  trimmed post-window          (gray)
  - Per-base mv "1" tick marks (thinned for long reads).
  - Optional polyA primary range (red), and secondary range (red dashed).

Inputs are pulled from the standard repo layout (--root-dir / --dataset /
--sample) or by explicit paths (--csv / --bam / --pod5-dir).

Outputs a PNG to --out (default: <read_id>.dorado_mv.png next to the CSV).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


# -- path resolution -------------------------------------------------------

def _resolve(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Returns (csv_path, bam_path, pod5_dir)."""
    if args.csv and args.bam and args.pod5_dir:
        return Path(args.csv), Path(args.bam), Path(args.pod5_dir)

    if not (args.root_dir and args.dataset and args.sample):
        raise SystemExit(
            "ERROR: provide --csv + --bam + --pod5-dir, or all of "
            "--root-dir + --dataset + --sample.")

    sample_dir = Path(args.root_dir) / "datasets" / args.dataset / args.sample
    csv_path = Path(args.csv) if args.csv else sample_dir / "3_alignment" / "dorado.extract_mv.csv"
    pod5_dir = Path(args.pod5_dir) if args.pod5_dir else sample_dir / "1_raw_signal" / "pod5"
    if args.bam:
        bam_path = Path(args.bam)
    else:
        import glob
        cands = sorted(glob.glob(str(sample_dir / "2_base_called" / "dorado-*" / "dorado.sorted.bam")))
        if len(cands) != 1:
            raise SystemExit(
                f"ERROR: expected exactly one dorado-*/dorado.sorted.bam under "
                f"{sample_dir}/2_base_called/, got {cands}. Pass --bam.")
        bam_path = Path(cands[0])

    for p, label in [(csv_path, "extract_mv.csv"), (bam_path, "BAM"), (pod5_dir, "pod5/")]:
        if not p.exists():
            raise SystemExit(f"ERROR: {label} not found: {p}")
    return csv_path, bam_path, pod5_dir


# -- data loaders ----------------------------------------------------------

def _load_csv_row(csv_path: Path, read_id: str) -> pd.Series:
    df = pd.read_csv(csv_path, comment="#")
    sub = df[df.read_id == read_id]
    if sub.empty:
        raise SystemExit(f"read_id {read_id!r} not in {csv_path}")
    if len(sub) > 1:
        raise SystemExit(f"read_id {read_id!r} has {len(sub)} rows in {csv_path}")
    return sub.iloc[0]


def _load_mv_from_bam(bam_path: Path, read_id: str) -> tuple[int, list[int]]:
    """Return (stride, mv_body) for the named primary forward record.
    Builds an in-memory name index (O(N) one-shot)."""
    import pysam
    bam = pysam.AlignmentFile(str(bam_path), "rb", check_sq=False)
    idx = pysam.IndexedReads(bam)
    idx.build()
    rec = next(idx.find(read_id))
    mv = rec.get_tag("mv")
    bam.close()
    return int(mv[0]), [int(x) for x in mv[1:]]


def _load_signal(pod5_dir: Path, read_id: str) -> np.ndarray:
    """Locate the .pod5 containing read_id via the sibling pod5.index, then
    fetch signal_pa."""
    import pod5
    from segshape.io.pod5_index import index_path_for, load_index, build_index

    idx_path = index_path_for(pod5_dir)
    if not idx_path.exists():
        print(f"  pod5 index missing at {idx_path}; building it now ...")
        build_index(pod5_dir)
    df = load_index(idx_path)
    hit = df[df.read_id == read_id]
    if hit.empty:
        raise SystemExit(f"read_id {read_id!r} not in pod5 index at {idx_path}")
    fname = hit.filename.iloc[0]
    with pod5.Reader(pod5_dir / fname) as r:
        rec = next(r.reads(selection=[read_id]))
        return np.array(rec.signal_pa, dtype=np.float32)


# -- plotting --------------------------------------------------------------

def _resolve_zoom(mode: str, mvts: int, mvte: int,
                  user_start: int | None, user_end: int | None,
                  random_window: int, seed: int) -> tuple[int, int, str]:
    """Compute (zoom_lo, zoom_hi, label_suffix) for the bottom panel."""
    aligned_len = mvte - mvts
    if mode == "aligned":
        return mvts, mvte, f"aligned [{mvts}, {mvte})  ({aligned_len} samples)"
    if mode == "user":
        if user_start is None or user_end is None:
            raise SystemExit("--zoom-mode user requires --plot-start and --plot-end")
        if not (0 <= user_start < user_end):
            raise SystemExit(f"invalid user window: [{user_start}, {user_end})")
        return user_start, user_end, f"user [{user_start}, {user_end})"
    if mode == "random":
        if aligned_len <= 0:
            raise SystemExit(f"empty aligned region: [{mvts}, {mvte})")
        win = min(random_window, aligned_len)
        rng = np.random.default_rng(seed)
        lo = int(rng.integers(mvts, mvte - win + 1)) if aligned_len > win else mvts
        return lo, lo + win, (
            f"random {win}-sample window [{lo}, {lo + win}) within "
            f"[{mvts}, {mvte}) (seed={seed})"
        )
    raise SystemExit(f"unknown --zoom-mode: {mode!r}")


def _plot(row: pd.Series, signal: np.ndarray, stride: int, mv_body: list[int],
          out_path: Path, max_ticks: int,
          zoom_mode: str, plot_start: int | None, plot_end: int | None,
          random_window: int, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rid = row.read_id
    ts = int(row.ts)
    ns = int(row.ns)
    mvts = int(row.mv_trans_start)
    mvte = int(row.mv_trans_end)
    pa_st = int(row.polya_start)
    pa_en = int(row.polya_end)
    pa_sst = int(row.polya_sec_start)
    pa_sen = int(row.polya_sec_end)
    n_sig = len(signal)

    # Per-base "1" positions in absolute signal index.
    ones_pos = ts + np.array([j for j, m in enumerate(mv_body) if m == 1],
                             dtype=np.int64) * stride

    fig, (ax_full, ax_zoom) = plt.subplots(
        2, 1, figsize=(13, 6.5), gridspec_kw={"height_ratios": [3, 2]})

    for ax in (ax_full, ax_zoom):
        ax.plot(np.arange(n_sig), signal, lw=0.4, color="black", rasterized=True)

    # Region shading (full panel).
    # Yellow = basecaller window outside the aligned slice = polyA + soft-clip
    #         (5' end up to mv_trans_start, 3' end from mv_trans_end to ns).
    # Red    = dorado-localized polyA range from pa:B:i (RNA004 only); a
    #         strict subset of the yellow region.
    ax_full.axvspan(0, ts,                       color="0.85", alpha=0.5, lw=0,
                    label="trimmed (outside ts/ns)")
    ax_full.axvspan(ns, n_sig,                   color="0.85", alpha=0.5, lw=0)
    ax_full.axvspan(ts, mvts,                    color="khaki", alpha=0.5, lw=0,
                    label="soft-clip")
    ax_full.axvspan(mvte, ns,                    color="khaki", alpha=0.5, lw=0)
    if pa_st >= 0 and pa_en > pa_st:
        ax_full.axvspan(pa_st, pa_en, color="red", alpha=0.25, lw=0,
                        label="polyA primary (pa:B:i)")
    if pa_sst >= 0 and pa_sen > pa_sst:
        ax_full.axvspan(pa_sst, pa_sen, color="red", alpha=0.15,
                        lw=0, hatch="//", label="polyA secondary")

    # Boundary lines + inline labels. ts/ns labels go to the TOP of the
    # axis; mv_trans_start/mv_trans_end go to the BOTTOM. That way, even
    # when the four x-positions are close (small soft-clip), the four text
    # labels live on two different vertical bands and never overlap.
    y_lo, y_hi = ax_full.get_ylim()
    pad = 0.02 * (y_hi - y_lo)
    for x, label, color, va, y in [
        (ts,   "ts",             "C0", "top",    y_hi - pad),
        (ns,   "ns",             "C0", "top",    y_hi - pad),
        (mvts, "mv_trans_start", "C3", "bottom", y_lo + pad),
        (mvte, "mv_trans_end",   "C3", "bottom", y_lo + pad),
    ]:
        ax_full.axvline(x, color=color, lw=0.8, ls="--")
        ax_full.text(x, y, f" {label}={x:,} ",
                     fontsize=7, va=va, color=color)

    ax_full.set_xlim(0, n_sig)
    ax_full.set_title(
        f"{rid}  |  ref={row.ref_name}  mapq={row.mapq}  "
        f"seq_len={row.seq_len}  stride={stride}  n_moves={row.n_moves}  "
        f"polya_tail={row.polya_tail}",
        fontsize=9)
    ax_full.set_ylabel("signal (pA)")
    ax_full.legend(loc="upper left", fontsize=7)

    # Bottom panel: zoom according to mode.
    zoom_lo, zoom_hi, zoom_label = _resolve_zoom(
        zoom_mode, mvts, mvte, plot_start, plot_end, random_window, seed)
    ax_zoom.set_xlim(zoom_lo, zoom_hi)
    in_zoom = ones_pos[(ones_pos >= zoom_lo) & (ones_pos < zoom_hi)]
    if len(in_zoom) > max_ticks:
        step = int(np.ceil(len(in_zoom) / max_ticks))
        in_zoom = in_zoom[::step]
        thin_suffix = f" (1/{step} thinned)"
    else:
        thin_suffix = ""
    for x in in_zoom:
        ax_zoom.axvline(x, color="C2", lw=0.3, alpha=0.6)
    ax_zoom.set_title(f"zoom: {zoom_label}  —  "
                      f"{len(in_zoom)} per-base ticks{thin_suffix}",
                      fontsize=9)
    ax_zoom.set_xlabel("signal sample index")
    ax_zoom.set_ylabel("signal (pA)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# -- CLI -------------------------------------------------------------------

def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--read-id", required=True, help="read_id to plot")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: "
                         "<root_dir>/figures/dorado-mv/<read_id>.png)")
    ap.add_argument("--max-ticks", type=int, default=300,
                    help="max per-base ticks drawn in the zoomed panel; "
                         "thinned if more aligned bases exist (default 300)")
    ap.add_argument("--zoom-mode", choices=["aligned", "user", "random"],
                    default="aligned",
                    help="bottom panel zoom: 'aligned' = full "
                         "[mv_trans_start, mv_trans_end); 'user' = "
                         "explicit [--plot-start, --plot-end); 'random' = a "
                         "random --random-window-sample slice within the "
                         "aligned region. Default 'aligned'.")
    ap.add_argument("--plot-start", type=int, default=None,
                    help="(zoom-mode user) signal index for left edge")
    ap.add_argument("--plot-end", type=int, default=None,
                    help="(zoom-mode user) signal index for right edge")
    ap.add_argument("--random-window", type=int, default=500,
                    help="(zoom-mode random) window size in samples (default 500)")
    ap.add_argument("--seed", type=int, default=0,
                    help="(zoom-mode random) RNG seed (default 0)")
    g = ap.add_argument_group("repo layout")
    g.add_argument("--root-dir", help="project root containing datasets/")
    g.add_argument("--dataset")
    g.add_argument("--sample", choices=["control", "treated"])
    g2 = ap.add_argument_group("explicit paths (override layout)")
    g2.add_argument("--csv", help="dorado.extract_mv.csv path")
    g2.add_argument("--bam", help="dorado.sorted.bam path")
    g2.add_argument("--pod5-dir", help="pod5/ folder with sibling pod5.index")
    return ap


def run(args: argparse.Namespace) -> int:
    csv_path, bam_path, pod5_dir = _resolve(args)
    print(f"csv  : {csv_path}")
    print(f"bam  : {bam_path}")
    print(f"pod5 : {pod5_dir}")

    row = _load_csv_row(csv_path, args.read_id)
    stride, mv_body = _load_mv_from_bam(bam_path, args.read_id)
    signal = _load_signal(pod5_dir, args.read_id)

    if args.out:
        out = Path(args.out)
    elif args.root_dir:
        out = Path(args.root_dir) / "figures" / "dorado-mv" / f"{args.read_id}.png"
    else:
        out = Path.cwd() / "figures" / "dorado-mv" / f"{args.read_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    _plot(row, signal, stride, mv_body, out,
          max_ticks=args.max_ticks,
          zoom_mode=args.zoom_mode,
          plot_start=args.plot_start, plot_end=args.plot_end,
          random_window=args.random_window, seed=args.seed)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    add_arguments(p)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
