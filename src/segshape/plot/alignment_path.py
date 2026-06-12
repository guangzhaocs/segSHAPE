"""Alignment-path PNG for a single read.

Plots the per-read DP path produced by ``segshape event-align`` (events on the
x-axis, kmer positions on the y-axis), with the per-read entry/exit anchor
boxes (mv-derived k_seed/k_end on event axis × mm2-derived j_min/j_max+1 on
kmer axis) and the EPS reference curve i ≈ j (slope 1, the "no skips, no
stays" diagonal that the DP would hit if events and kmers were 1:1).

Inputs (new flat 3_alignment/ layout):
  - ``alignment.csv``  read_idx, event_idx, pos_idx, ref_center_base_pos
  - ``scale.csv``      per-read DP corner indices + ll + scale/shift
  - ``subevents.parquet`` events per read (used for n_events and event_starts)

Output: one PNG per read. Reads with qc_tag != 'PASS' are skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument('--align-dir', required=True, type=Path,
                    help='3_alignment/ directory containing alignment.csv, '
                         'scale.csv, subevents.parquet')
    ap.add_argument('--read-id', default=None,
                    help='specific read_id; default auto-pick the first PASS '
                         'read with high kmer-axis coverage')
    ap.add_argument('--out', default=None, type=Path,
                    help='output PNG path. Default: '
                         '<align-dir>/alignment_path_<read_id_short>.png')
    ap.add_argument('--title-prefix', default='',
                    help='extra string prepended to the figure title '
                         '(e.g. dataset/sample for legibility)')
    return ap


def _pick_read(scale: pd.DataFrame, align: pd.DataFrame) -> str:
    """Auto-pick the first PASS read with the highest kmer-position coverage."""
    pass_reads = scale[scale.qc_tag == 'PASS']
    if pass_reads.empty:
        raise SystemExit("no PASS read found in scale.csv")
    cov = align.groupby('read_idx')['pos_idx'].nunique()
    pass_reads = pass_reads.assign(cov=pass_reads.read_idx.map(cov).fillna(0))
    best = pass_reads.sort_values('cov', ascending=False).iloc[0]
    return str(best.read_id)


def run(args) -> int:
    # Local matplotlib import keeps `segshape --help` cheap.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    align_dir: Path = args.align_dir
    align_path = align_dir / 'alignment.csv'
    scale_path = align_dir / 'scale.csv'
    for p in (align_path, scale_path):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    print(f"loading {scale_path}")
    scale = pd.read_csv(scale_path)
    print(f"loading {align_path}")
    align = pd.read_csv(align_path)

    rid = args.read_id or _pick_read(scale, align)
    sub_scale = scale[scale.read_id == rid]
    if sub_scale.empty:
        raise SystemExit(f"read_id {rid!r} not found in scale.csv")
    qc_tag = str(sub_scale['qc_tag'].iloc[0])
    if qc_tag != 'PASS':
        raise SystemExit(f"read {rid} has qc_tag={qc_tag!r} (not PASS)")

    # scale.csv has mixed-type columns (PASS rows numeric, placeholder rows
    # empty strings) → pandas infers object dtype. Cast each field explicitly.
    read_idx = int(sub_scale['read_idx'].iloc[0])
    ll       = float(sub_scale['ll'].iloc[0])
    shift    = float(sub_scale['shift'].iloc[0])
    scale_v  = float(sub_scale['scale'].iloc[0])
    n_events = int(sub_scale['n_events'].iloc[0])
    k_seed   = int(sub_scale['k_seed'].iloc[0])
    k_start  = int(sub_scale['k_start'].iloc[0])
    best_end = int(sub_scale['best_end'].iloc[0])
    j_start  = int(sub_scale['j_start'].iloc[0])
    j_end    = int(sub_scale['j_end'].iloc[0])
    print(f"read_id={rid}  read_idx={read_idx}  ll={ll:.1f}  "
          f"scale={scale_v:.3f}  shift={shift:.3f}  n_events={n_events}")

    sub = align[align.read_idx == read_idx].sort_values('event_idx')
    if sub.empty:
        raise SystemExit(f"no alignment rows for read_idx={read_idx}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(sub.event_idx, sub.pos_idx, s=3, c='steelblue', alpha=0.6,
               label=f'aligned events (n={len(sub)})')
    ax.plot(sub.event_idx, sub.pos_idx, lw=0.4, c='steelblue', alpha=0.3)
    ax.scatter([k_seed], [j_start], s=80, marker='*', c='green',
               edgecolors='k', label=f'k_seed/j_start ({k_seed},{j_start})',
               zorder=5)
    ax.scatter([best_end], [j_end], s=80, marker='*', c='red',
               edgecolors='k', label=f'best_end/j_end ({best_end},{j_end})',
               zorder=5)

    # Diagonal reference (slope 1, anchored at entry corner)
    di = best_end - k_start
    if di > 0:
        ref_x = np.linspace(k_start, best_end, 50)
        ref_y = j_start + (j_end - j_start) * (ref_x - k_start) / di
        ax.plot(ref_x, ref_y, '--', c='gray', lw=0.8, alpha=0.6,
                label='straight (k_start,j_start)→(best_end,j_end)')

    ax.set_xlabel('event_idx (3'+chr(8242)+'→5'+chr(8242)+' time)')
    ax.set_ylabel('pos_idx (3'+chr(8242)+'→5'+chr(8242)+' kmer axis)')
    pre = (args.title_prefix.rstrip(' /') + '  ') if args.title_prefix else ''
    ax.set_title(
        f"{pre}alignment path | read {rid} | n_events={n_events}, "
        f"shift={shift:.2f}", fontsize=9)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)

    out: Optional[Path] = args.out
    if out is None:
        out = align_dir / f'alignment_path_{rid[:12]}.png'
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main() or 0)
