"""Score a pre-computed RNAfold .out against a dot-bracket ground truth.

Companion to ``scripts/7_fold_batch.sh``: that script runs RNAfold on every
``reactivity_*.dat`` and saves the raw stdout as ``<basename>.out``. This
module extracts MFE + centroid from each .out and computes per-structure
precision / recall / F1 / MCC vs ``--struct-gt``, plus per-position
struct-AUC vs ``--shape-dat`` (1 = unpaired in GT, score = .dat value).

**No offset, no reference fasta.** Pair-set metrics are length-only:
the predicted dot-bracket and the GT dot-bracket are aligned 1-to-1
position-by-position, and the inputs only need to satisfy
``len(predicted) == len(GT)``. ``ref_len`` is derived from the
predicted dot-bracket itself.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# Parse RNAfold .out  (`-p -d2 --noLP`)
# ---------------------------------------------------------------------------
#
# Typical layout (after optional WARNING lines):
#   >contig
#   <sequence>
#   <MFE_dotbracket>          ( -X.XX kcal/mol)
#   <ensemble_brackets>       [-Y.YY]
#   <CENTROID_dotbracket>     {-Z.ZZ d=N.NN}        <-- always 3rd dot-bracket
#   frequency of mfe ...
#
# `--noLP` means MFE / centroid use only `.` `(` `)`, but the ensemble line
# uses `., (, ), [, ], {, }, |` — so we filter to lines whose first token is
# *purely* dot-bracket.

def parse_rnafold_out(path: str) -> List[Tuple[str, Optional[float]]]:
    """Return list of (dotbracket, energy) — RNAfold's structure rows in
    output order. The function looks at every line whose first token is
    a pure ``.()``-only string; sequence / header / warning / "frequency
    of mfe ..." lines are ignored. Energy parsed from any signed float
    inside trailing ``(...)``, ``[...]``, or ``{...}``."""
    import re
    energy_rx = re.compile(r"[\(\[\{]\s*(-?\d+\.\d+)")
    db_records: list = []
    with open(path) as fh:
        for line in fh:
            s = line.rstrip("\n")
            toks = s.split()
            if not toks:
                continue
            first = toks[0]
            if first and all(c in ".()" for c in first):
                m = energy_rx.search(s)
                e = float(m.group(1)) if m else None
                db_records.append((first, e))
    if len(db_records) < 1:
        raise SystemExit(f"no dot-bracket lines in {path}")
    return db_records


def _select_mfe_centroid(records: List[Tuple[str, Optional[float]]]
                          ) -> Tuple[Tuple[str, Optional[float]],
                                     Tuple[str, Optional[float]]]:
    """RNAfold under ``-p -d2`` emits 4 dot-bracket-only lines:
      0: MFE             (filtered: only . ( ) — passes here)
      1: ensemble        (uses . ( ) [ ] { } | — filtered out elsewhere)
      2: centroid        (only . ( ) — passes here)
      3: MEA             (only . ( ) — passes here)

    With our parser (which filters to *pure* `.()` lines), the surviving
    list is ``[MFE, centroid, MEA]``. Pick MFE = records[0],
    centroid = records[1] (by RNAfold's documented ordering)."""
    if not records:
        raise SystemExit("no parseable dot-bracket records")
    mfe = records[0]
    centroid = records[1] if len(records) >= 2 else records[0]
    return mfe, centroid


# ---------------------------------------------------------------------------
# Struct GT loader (same logic as evaluate/pipeline.py)
# ---------------------------------------------------------------------------

def read_struct_gt(path: str) -> str:
    """Plain text or fasta-like dot-bracket. Concatenates non-> lines.
    Strict: every char must be in ``.()``."""
    parts = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">") or line.startswith("#"):
                continue
            parts.append(line.strip())
    s = "".join(parts)
    if not s or not all(c in ".()" for c in s):
        raise SystemExit(
            f"struct-gt {path}: not a clean dot-bracket string "
            f"(got {len(s)} chars, sample={s[:40]!r})")
    return s


# ---------------------------------------------------------------------------
# Pair-set metrics  (precision / recall / F1 / MCC)
# ---------------------------------------------------------------------------

def find_bracket_pairs(s: str):
    """Stack-based pair extraction — returns list of (i, j) 1-indexed."""
    stack, pairs = [], []
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            pairs.append((stack.pop() + 1, i + 1))
    return pairs


def calc_struct_metrics(y_true: str, y_pred: str) -> dict:
    if len(y_true) != len(y_pred):
        raise SystemExit(
            f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)} "
            f"— RNAfold output and struct-GT must come from the same "
            f"reference contig (no offset sweep here).")
    n = len(y_true)
    # All possible position pairs: C(n, 2)
    comb_n = math.factorial(n) // (math.factorial(2) * math.factorial(n - 2))
    yt = set(find_bracket_pairs(y_true))
    yp = set(find_bracket_pairs(y_pred))
    tp = len(yt & yp)
    fn = len(yt - yp)
    fp = len(yp - yt)
    tn = comb_n - tp - fn - fp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) \
                  if (precision + recall) > 0 else 0
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = num / den if den > 0 else 0
    return {
        "precision": precision * 100,
        "recall":    recall * 100,
        "f1":        f1 * 100,
        "mcc":       mcc * 100,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "n_true_pairs": len(yt),
        "n_pred_pairs": len(yp),
    }


# ---------------------------------------------------------------------------
# Per-position struct AUC (no offset)
# ---------------------------------------------------------------------------

def read_shape_dat(path: str, ref_len: int) -> np.ndarray:
    """Read a 1-indexed `.dat` (pos\\tvalue, -999=NA). Returns array of
    length ref_len (0-padded NaN)."""
    arr = np.full(ref_len, np.nan, dtype=np.float64)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            toks = line.split()
            if len(toks) < 2: continue
            try:
                p = int(toks[0]); v = float(toks[1])
            except ValueError:
                continue
            if v == -999: continue
            if 1 <= p <= ref_len:
                arr[p - 1] = v
    return arr


def compute_struct_auc(struct: str, scores: np.ndarray,
                       edge_mask: int = 0) -> float:
    """Per-position AUC: positive class = unpaired (`.`), score = scores[i].
    Higher score → more "reactive" → expected unpaired → label=1.

    Kept for backward compatibility with scripts that import this name.
    For full metric set including PR-AUC / Mann-Whitney / threshold-based
    precision/recall/F1/accuracy, use ``compute_struct_metrics`` below."""
    n = len(struct)
    if len(scores) != n:
        raise SystemExit(
            f"AUC length mismatch: struct={n} scores={len(scores)}")
    labels = np.array([1 if c == "." else 0 for c in struct], dtype=np.int8)
    edge_ok = (np.arange(n) >= edge_mask) & (np.arange(n) < n - edge_mask)
    mask = edge_ok & np.isfinite(scores)
    if mask.sum() < 50 or len(np.unique(labels[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(labels[mask], scores[mask]))


def read_raw_mod_rate(csv_path: str, ref_len: int,
                       offset: int = 2) -> np.ndarray:
    """Read raw ``mod_rate`` column from ``mod_rate.csv`` and project onto
    1-based reference positions (length ``ref_len``).

    The CSV uses the alignment-internal ``pos_idx`` (kmer axis,
    reverse-oriented). The mapping ``fa_pos = ref_len − pos_idx − offset``
    matches what ``calling._write_reactivity_dat`` and the rest of the
    pipeline use. Default ``offset=2`` (RNA002 5-mer centre / RNA004
    9-mer centre with edge_pad=2 — see docs/attention.md §2).

    Returns float64 array of length ``ref_len``; positions with no
    alignment coverage stay as NaN."""
    df = pd.read_csv(csv_path, usecols=["pos_idx", "mod_rate"])
    arr = np.full(ref_len, np.nan, dtype=np.float64)
    pos = df["pos_idx"].to_numpy(dtype=np.int64)
    val = df["mod_rate"].to_numpy(dtype=np.float64)
    fa = ref_len - pos - offset
    sel = (fa >= 1) & (fa <= ref_len) & np.isfinite(val)
    arr[fa[sel] - 1] = val[sel]
    return arr


def compute_struct_metrics(struct: str, scores: np.ndarray,
                            edge_mask: int = 0) -> dict:
    """Comprehensive per-position metrics for ``scores`` (raw mod_rate
    recommended) against ``struct`` (dot-bracket GT).

    Convention: ``label=1`` if struct[i] == '.' (unpaired), ``label=0``
    if struct[i] in '()' (paired). Higher ``scores[i]`` means stronger
    "reactive / unpaired" prediction. All metrics are rank-based or
    distributional (no threshold), so **invariant under any monotonic
    transform** of scores — z-score, shape_28, log all give identical
    values; only smoothing changes ranks.

    Returns a flat dict suitable for direct DataFrame conversion. All
    counts are ints; ratios are kept as floats (no string formatting)."""
    n = len(struct)
    if len(scores) != n:
        raise SystemExit(
            f"length mismatch: struct={n} scores={len(scores)}")
    labels = np.array([1 if c == "." else 0 for c in struct], dtype=np.int8)
    edge_ok = (np.arange(n) >= edge_mask) & (np.arange(n) < n - edge_mask)
    mask = edge_ok & np.isfinite(scores)

    nan_dict = dict(struct_n=int(mask.sum()),
                    struct_n_unpair=0, struct_n_pair=0,
                    struct_auc=float("nan"),
                    struct_pr_auc=float("nan"),
                    struct_mw_u=float("nan"),
                    struct_mw_p=float("nan"),
                    struct_mean_unpair=float("nan"),
                    struct_mean_pair=float("nan"),
                    struct_mean_diff=float("nan"),
                    struct_median_diff=float("nan"))
    if mask.sum() < 50 or len(np.unique(labels[mask])) < 2:
        return nan_dict

    y = labels[mask]
    s = scores[mask]
    n_unpair = int(y.sum())
    n_pair = int(len(y) - n_unpair)

    auc    = float(roc_auc_score(y, s))
    pr_auc = float(average_precision_score(y, s))
    u_stat, u_p = mannwhitneyu(s[y == 1], s[y == 0], alternative="greater")
    u_stat, u_p = float(u_stat), float(u_p)

    s_unpair = s[y == 1]; s_pair = s[y == 0]

    return dict(
        struct_n=int(mask.sum()),
        struct_n_unpair=n_unpair,
        struct_n_pair=n_pair,
        struct_auc=auc,
        struct_pr_auc=pr_auc,
        struct_mw_u=u_stat,
        struct_mw_p=u_p,
        struct_mean_unpair=float(s_unpair.mean()),
        struct_mean_pair=float(s_pair.mean()),
        struct_mean_diff=float(s_unpair.mean() - s_pair.mean()),
        struct_median_diff=float(np.median(s_unpair) - np.median(s_pair)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g_in = p.add_argument_group("input")
    g_in.add_argument("--rnafold-out", required=True,
                      help="raw RNAfold stdout (e.g. "
                           "reactivity_smooth0_norm-zscore.out from "
                           "scripts/7_fold_batch.sh)")
    g_in.add_argument("--struct-gt", required=True,
                      help="dot-bracket ground-truth (1-based, same ref_len "
                           "as RNAfold output sequence)")
    g_in.add_argument("--shape-dat", default=None,
                      help="1-indexed .dat used as the SHAPE input to "
                           "RNAfold (defaults to .out path with .dat "
                           "extension); used for struct-AUC scoring")
    g_in.add_argument("--edge-mask", type=int, default=0,
                      help="positions excluded from each end for struct-AUC "
                           "(default 0; pair-set MCC ignores this)")

    g_o = p.add_argument_group("output")
    g_o.add_argument("--variant-name", default=None,
                     help="row tag for TSV output. Default: parent dir name "
                          "of --rnafold-out")
    g_o.add_argument("--out-tsv", default=None,
                     help="append a row to this TSV. Default: print pretty "
                          "to stdout, no TSV.")
    g_o.add_argument("--quiet", action="store_true",
                     help="suppress pretty-print to stdout")
    return p


def eval_one(rnafold_out: str, struct_gt: str,
             shape_dat: Optional[str] = None,
             edge_mask: int = 0) -> dict:
    """Compute metrics for one (.out, struct_gt) pair. Returns a dict
    suitable for direct DataFrame conversion. ``struct_gt`` may be either
    a file path or a literal dot-bracket string (auto-detected: literals
    use only ``.()``)."""
    records = parse_rnafold_out(rnafold_out)
    mfe, centroid = _select_mfe_centroid(records)
    mfe_db, mfe_e = mfe
    cen_db, cen_e = centroid
    ref_len = len(cen_db)

    if (struct_gt
            and not os.path.isfile(struct_gt)
            and all(c in ".()" for c in struct_gt)):
        gt = struct_gt                          # caller passed the string
    else:
        gt = read_struct_gt(struct_gt)

    sm_mfe = calc_struct_metrics(gt, mfe_db)
    sm_cen = calc_struct_metrics(gt, cen_db)

    # Per-position metrics on RAW mod_rate (rank-based; smoothing /
    # normalization choices in the .dat file would change ranks, so we
    # bypass the .dat and read mod_rate.csv directly).
    sm_pos = compute_struct_metrics(gt, np.full(ref_len, np.nan),
                                     edge_mask=edge_mask)
    mod_rate_csv = None
    out_dir = os.path.dirname(os.path.abspath(rnafold_out))
    cand_csv = os.path.join(out_dir, "mod_rate.csv")
    if os.path.isfile(cand_csv):
        mod_rate_csv = cand_csv
        scores = read_raw_mod_rate(mod_rate_csv, ref_len)
        sm_pos = compute_struct_metrics(gt, scores, edge_mask=edge_mask)

    return {
        "rnafold_out":        os.path.abspath(rnafold_out),
        "mod_rate_csv":       os.path.abspath(mod_rate_csv) if mod_rate_csv else "",
        "ref_len":            ref_len,
        "gt_pairs":           gt.count("("),
        "mfe_energy":         "" if mfe_e is None else f"{mfe_e:.4f}",
        "centroid_energy":    "" if cen_e is None else f"{cen_e:.4f}",
        "centroid_precision": f"{sm_cen['precision']:.4f}",
        "centroid_recall":    f"{sm_cen['recall']:.4f}",
        "centroid_f1":        f"{sm_cen['f1']:.4f}",
        "centroid_mcc":       f"{sm_cen['mcc']:.4f}",
        "centroid_tp":        sm_cen["tp"],
        "centroid_fn":        sm_cen["fn"],
        "centroid_fp":        sm_cen["fp"],
        "centroid_n_pred_pairs": sm_cen["n_pred_pairs"],
        "mfe_precision":      f"{sm_mfe['precision']:.4f}",
        "mfe_recall":         f"{sm_mfe['recall']:.4f}",
        "mfe_f1":             f"{sm_mfe['f1']:.4f}",
        "mfe_mcc":            f"{sm_mfe['mcc']:.4f}",
        "mfe_n_pred_pairs":   sm_mfe["n_pred_pairs"],
        # Per-position metrics on raw mod_rate
        "struct_n":              sm_pos["struct_n"],
        "struct_n_unpair":       sm_pos["struct_n_unpair"],
        "struct_n_pair":         sm_pos["struct_n_pair"],
        "struct_auc":            f"{sm_pos['struct_auc']:.6f}"        if np.isfinite(sm_pos["struct_auc"]) else "",
        "struct_pr_auc":         f"{sm_pos['struct_pr_auc']:.6f}"     if np.isfinite(sm_pos["struct_pr_auc"]) else "",
        "struct_mw_u":           f"{sm_pos['struct_mw_u']:.4e}"       if np.isfinite(sm_pos["struct_mw_u"]) else "",
        "struct_mw_p":           f"{sm_pos['struct_mw_p']:.4e}"       if np.isfinite(sm_pos["struct_mw_p"]) else "",
        "struct_mean_unpair":    f"{sm_pos['struct_mean_unpair']:.6f}"        if np.isfinite(sm_pos["struct_mean_unpair"]) else "",
        "struct_mean_pair":      f"{sm_pos['struct_mean_pair']:.6f}"          if np.isfinite(sm_pos["struct_mean_pair"]) else "",
        "struct_mean_diff":      f"{sm_pos['struct_mean_diff']:.6f}"          if np.isfinite(sm_pos["struct_mean_diff"]) else "",
        "struct_median_diff":    f"{sm_pos['struct_median_diff']:.6f}"        if np.isfinite(sm_pos["struct_median_diff"]) else "",
        "edge_mask":             edge_mask,
        # extra for pretty-print
        "_mfe_db": mfe_db, "_cen_db": cen_db,
        "_sm_mfe": sm_mfe, "_sm_cen": sm_cen, "_sm_pos": sm_pos,
        "_mfe_e":  mfe_e,  "_cen_e": cen_e,
    }


def run(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.rnafold_out):
        raise SystemExit(f"--rnafold-out not found: {args.rnafold_out}")
    if not os.path.isfile(args.struct_gt):
        raise SystemExit(f"--struct-gt not found: {args.struct_gt}")

    res = eval_one(args.rnafold_out, args.struct_gt,
                   shape_dat=args.shape_dat,
                   edge_mask=args.edge_mask)
    sm_cen = res["_sm_cen"]; sm_mfe = res["_sm_mfe"]; sm_pos = res["_sm_pos"]
    cen_db = res["_cen_db"]; mfe_db = res["_mfe_db"]
    cen_e  = res["_cen_e"];  mfe_e  = res["_mfe_e"]
    gt_pairs = res["gt_pairs"]; ref_len = res["ref_len"]

    if not args.quiet:
        print(f"  pred length  : {ref_len}")
        print(f"  GT length    : {ref_len}")
        print(f"  MFE energy   : {mfe_e if mfe_e is not None else 'n/a'} kcal/mol "
              f"({mfe_db.count('(')} pairs)")
        print(f"  centroid e   : {cen_e if cen_e is not None else 'n/a'} kcal/mol "
              f"({cen_db.count('(')} pairs)")
        print(f"  GT pairs     : {gt_pairs}")
        print()
        print("  ── pair-set metrics (centroid / MFE vs GT) ──")
        print(f"  {'metric':<14} {'centroid':>10} {'MFE':>10}")
        for k in ("precision", "recall", "f1", "mcc"):
            print(f"  {k:<12} {sm_cen[k]:>9.2f}% {sm_mfe[k]:>9.2f}%")
        print(f"  {'tp/fn/fp':<12} {sm_cen['tp']:>3}/{sm_cen['fn']:>3}/{sm_cen['fp']:>3} "
              f"   {sm_mfe['tp']:>3}/{sm_mfe['fn']:>3}/{sm_mfe['fp']:>3}")
        print(f"  {'n_pred_pairs':<12} {sm_cen['n_pred_pairs']:>10} {sm_mfe['n_pred_pairs']:>10}")
        print()
        print("  ── per-position metrics (raw mod_rate vs GT, unpair=1) ──")
        if not np.isfinite(sm_pos["struct_auc"]):
            print(f"  (skipped — no mod_rate.csv sibling, or n<50, or one-class GT)")
        else:
            print(f"  n_used / unpair / pair :  {sm_pos['struct_n']} / "
                  f"{sm_pos['struct_n_unpair']} / {sm_pos['struct_n_pair']}")
            print(f"  ROC AUC                :  {sm_pos['struct_auc']:.4f}")
            print(f"  PR  AUC                :  {sm_pos['struct_pr_auc']:.4f}")
            print(f"  Mann-Whitney U / p     :  {sm_pos['struct_mw_u']:.3e} / "
                  f"{sm_pos['struct_mw_p']:.3e}")
            print(f"  mean(unpair) - mean(pair) :  {sm_pos['struct_mean_diff']:+.4f}")
            print(f"  median(unpair) - median(pair) :  {sm_pos['struct_median_diff']:+.4f}")

    variant = args.variant_name or os.path.basename(
        os.path.dirname(os.path.abspath(args.rnafold_out)))
    # Strip the private "_*" pretty-print extras from eval_one().
    row = {"variant": variant,
           "struct_gt": os.path.abspath(args.struct_gt),
           **{k: v for k, v in res.items() if not k.startswith("_")}}

    if args.out_tsv:
        new_file = not os.path.exists(args.out_tsv)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_tsv)) or ".",
                    exist_ok=True)
        pd.DataFrame([row]).to_csv(args.out_tsv, sep="\t", index=False,
                                    mode=("w" if new_file else "a"),
                                    header=new_file)
        if not args.quiet:
            print(f"\n{'wrote' if new_file else 'appended to'} {args.out_tsv}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="segshape evaluate fold-out")
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main() or 0)
