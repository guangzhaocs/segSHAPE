"""Batch eval of every RNAfold .out under a root directory.

Walks ``--root-dir``, finds every ``*.out`` (or files matching ``--pattern``),
calls :func:`segshape.evaluate.fold_out.eval_one` on each, parses some
structural columns from the path (cell, method, smoothing) for downstream
analysis, and writes one CSV.

Companion to ``segshape evaluate fold-out`` (single file). Sibling
``.dat`` files (same basename, different ext) are auto-picked for the
struct-AUC scoring side-channel.

Typical usage::

    segshape evaluate fold-out-all \\
        --root-dir  datasets/miR17-92/treated/3_alignment \\
        --struct-gt datasets/miR17-92/reference/structure_gt.txt \\
        --out-csv   datasets/miR17-92/res_all.csv

Path is parsed under the assumption it follows::

    <root>/<sweep_cell>/mod_rate/<method_dir>/<dat_basename>.out

so the row's ``cell`` / ``method`` / ``dat_basename`` columns are
``rna002_de50_dk15_bc0.0_sp50_shift_only`` / ``default_if-1D_c0.0050`` /
``reactivity_smooth0_norm-zscore``. Files outside that layout still work
— they just get the full relative path in ``variant`` and blank
``cell``/``method``/``dat_basename``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from segshape.evaluate.fold_out import eval_one, read_struct_gt


def _parse_path(out_path: Path, root_dir: Path) -> dict:
    """Best-effort extract (cell, method, dat_basename) from
    ``<root>/<cell>/mod_rate/<method>/<basename>.out``."""
    try:
        rel = out_path.relative_to(root_dir)
    except ValueError:
        rel = out_path
    parts = rel.parts
    cell = method = ""
    dat_basename = out_path.stem
    if len(parts) >= 4 and parts[-3] == "mod_rate":
        cell = parts[-4]
        method = parts[-2]
    return {
        "variant":      str(rel.with_suffix("")),
        "cell":         cell,
        "method":       method,
        "dat_basename": dat_basename,
    }


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g_in = p.add_argument_group("input")
    g_in.add_argument("--root-dir", required=True,
                      help="walk this directory recursively for RNAfold .out files")
    g_in.add_argument("--struct-gt", required=True,
                      help="dot-bracket ground-truth file shared across every "
                           ".out in --root-dir (same ref_len = ref_len of every "
                           ".out's predicted structure)")
    g_in.add_argument("--pattern", default="*.out",
                      help="glob pattern under --root-dir (default '*.out')")
    g_in.add_argument("--edge-mask", type=int, default=0,
                      help="positions excluded from each end for struct-AUC "
                           "(default 0; pair-set MCC ignores this)")

    g_o = p.add_argument_group("output")
    g_o.add_argument("--out-csv", required=True,
                     help="single CSV aggregating one row per .out file. "
                          "Overwrites if exists.")
    g_o.add_argument("--quiet", action="store_true",
                     help="suppress per-file progress lines (errors still printed)")
    return p


def run(args: argparse.Namespace) -> int:
    root = Path(args.root_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"--root-dir not a directory: {root}")
    if not os.path.isfile(args.struct_gt):
        raise SystemExit(f"--struct-gt not found: {args.struct_gt}")

    # Read GT once (shared across all rows).
    gt = read_struct_gt(args.struct_gt)
    out_files = sorted(root.rglob(args.pattern))
    if not out_files:
        raise SystemExit(f"no files matching {args.pattern!r} under {root}")

    print(f"root      : {root}")
    print(f"struct-gt : {args.struct_gt}  ({len(gt)} chars, {gt.count('(')} pairs)")
    print(f"pattern   : {args.pattern}  → {len(out_files)} file(s)")
    print(f"out-csv   : {args.out_csv}")
    print()

    rows = []
    n_ok = n_fail = 0
    for f in out_files:
        try:
            res = eval_one(str(f), gt,           # gt as literal dotbracket
                           shape_dat=None,
                           edge_mask=args.edge_mask)
        except SystemExit as e:
            n_fail += 1
            print(f"  FAIL  {f.relative_to(root)}  →  {e}")
            continue
        path_meta = _parse_path(f, root)
        # drop the private pretty-print extras
        row = {**path_meta,
               **{k: v for k, v in res.items() if not k.startswith("_")},
               "struct_gt": os.path.abspath(args.struct_gt)}
        rows.append(row)
        n_ok += 1
        if not args.quiet:
            sm = res
            print(f"  [{n_ok:>3}] {row['cell']:<48} {row['method']:<28} "
                  f"sm={row['dat_basename'].split('_smooth')[1].split('_')[0] if '_smooth' in row['dat_basename'] else '?':<2} "
                  f"  cen-MCC={sm['centroid_mcc']}  AUC={sm['struct_auc']}")

    if not rows:
        raise SystemExit("no successful evaluations — nothing to write")

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print()
    print(f"wrote {out_csv}  ({n_ok} rows, {n_fail} failed)")
    return 0 if n_fail == 0 else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="segshape evaluate fold-out-all")
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main() or 0)
