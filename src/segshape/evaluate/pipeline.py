"""Structure-recovery evaluation: mod_rate.csv → RNAfold → MCC / AUC / Spearman.

Inputs:
  --mod-rate-csv   :  mod_rate.csv produced by `segshape mod-calling`
                     (cols: pos_idx, mod_rate; the z-score is recomputed
                     from mod_rate, or read from a reactivity_z column if
                     a legacy file still carries one)
  --ref-fa         :  fasta containing the contig
  --struct-gt      :  dot-bracket ground-truth (one structure line per fasta entry)
  --react-gt       :  .dat ground-truth reactivity (1-indexed pos\treact, -999=NA)
  --par-path       :  RNAfold --P thermodynamic params file
  --out-prefix     :  base path; produces {prefix}/{variant}_off{i}.{dat,centroid}
                      and {prefix}__{variant}.tsv

Per kmer-offset i (default 0..4):
  1. write reactivity_z → {prefix}/{variant}_off{i}.dat (fasta-indexed,
     missing positions = -999); fa = ref_len - pos_idx - offset (1-indexed)
  2. RNAfold -p -d2 --noLP --shape={dat} --shapeMethod=D
  3. parse centroid (last) + MFE (first) dot-bracket
  4. precision / recall / F1 / MCC vs --struct-gt
Plus per-offset Spearman vs --react-gt and AUC of mod_rate vs paired/unpaired
labels in --struct-gt. Best offset reported on each metric.

mod_rate computation lives in `segshape mod-calling`; this module is
purely the structure-recovery + GT-comparison stage.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _zscore(x: np.ndarray) -> np.ndarray:
    out = x.copy().astype(np.float64)
    m = ~np.isnan(out)
    if m.sum() < 2:
        return out
    mu, sd = out[m].mean(), out[m].std()
    if sd > 0:
        out[m] = (out[m] - mu) / sd
    return out


def write_reactivity_dat(rates_z: np.ndarray, ref_len: int, offset: int,
                          out_path: str) -> int:
    """Map pos_idx i → fasta pos (ref_len - i - offset, 1-indexed). Fill
    missing with -999. Input `rates_z` is already z-scored.
    Returns count of valid positions written.
    """
    L = len(rates_z)
    fa_react = np.full(ref_len, -999.0)
    n_valid = 0
    for i in range(L):
        if not np.isfinite(rates_z[i]):
            continue
        fa = ref_len - i - offset  # 1-indexed
        if 1 <= fa <= ref_len:
            fa_react[fa - 1] = rates_z[i]
            n_valid += 1
    with open(out_path, "w") as f:
        for i, v in enumerate(fa_react):
            f.write(f"{i+1}\t{v:.6f}\n" if v != -999.0 else f"{i+1}\t-999\n")
    return n_valid


def find_bracket_pairs(s: str):
    stack, pairs = [], []
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            pairs.append((stack.pop() + 1, i + 1))
    return pairs


def calc_struct_metrics(y_true: str, y_pred: str) -> dict:
    n = len(y_true)
    k = 2
    comb_n = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    yt_pairs = set(find_bracket_pairs(y_true))
    yp_pairs = set(find_bracket_pairs(y_pred))
    tp = len(yt_pairs & yp_pairs)
    fn = len(yt_pairs - yp_pairs)
    fp = len(yp_pairs - yt_pairs)
    tn = comb_n - tp - fn - fp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = num / den if den > 0 else 0
    return {
        "precision": precision * 100, "recall": recall * 100,
        "f1": f1 * 100, "mcc": mcc * 100,
        "tp": tp, "fn": fn, "fp": fp,
    }


def run_rnafold_with_shape(fasta_path: str, par_path: str,
                            shape_path: str, work_dir: str) -> Tuple[str, str]:
    """RNAfold -p -d2 --noLP --shape=DAT --shapeMethod=D.
    Return (mfe_struct, centroid_struct).
    """
    cmd = [
        "RNAfold", "-p", "-d2", "--noLP",
        "-P", par_path,
        f"--shape={shape_path}",
        "--shapeMethod=D",
        "--noPS",
    ]
    with open(fasta_path) as fa:
        result = subprocess.run(cmd, stdin=fa, capture_output=True, text=True,
                                cwd=work_dir, check=True)
    lines = result.stdout.split("\n")
    db_lines = []
    for line in lines:
        first_tok = line.split()[0] if line.split() else ""
        if first_tok and all(c in ".()" for c in first_tok):
            db_lines.append(first_tok)
    if len(db_lines) < 2:
        raise RuntimeError(f"Need MFE+centroid; got {len(db_lines)} dot-bracket "
                            f"lines. Output:\n{result.stdout}")
    return db_lines[0], db_lines[-1]


def correlate_react(rates_z: np.ndarray, react: pd.Series, ref_len: int,
                     edge_mask: int, offsets) -> Dict[int, float]:
    L = len(rates_z)
    out = {}
    for off in offsets:
        gt_pos = ref_len - np.arange(L) - off
        gt_v = np.array([react.get(p, np.nan) for p in gt_pos])
        edge_ok = (np.arange(L) >= edge_mask) & (np.arange(L) < L - edge_mask)
        mask = edge_ok & np.isfinite(gt_v) & np.isfinite(rates_z)
        if mask.sum() < 50:
            out[off] = np.nan
            continue
        r, _ = spearmanr(rates_z[mask], gt_v[mask])
        out[off] = float(r)
    return out


def auc_struct(rates: np.ndarray, struct: str, ref_len: int,
                edge_mask: int, offsets) -> Dict[int, float]:
    L = len(rates)
    out = {}
    for off in offsets:
        labels = np.zeros(L, dtype=np.int8) - 1
        for i in range(L):
            fa = ref_len - i - off
            if 1 <= fa <= len(struct):
                ch = struct[fa - 1]
                labels[i] = 1 if ch == "." else 0
        edge_ok = (np.arange(L) >= edge_mask) & (np.arange(L) < L - edge_mask)
        mask = edge_ok & (labels >= 0) & np.isfinite(rates)
        if mask.sum() < 50 or len(np.unique(labels[mask])) < 2:
            out[off] = np.nan
            continue
        try:
            from sklearn.metrics import roc_auc_score
            out[off] = float(roc_auc_score(labels[mask], rates[mask]))
        except Exception:
            out[off] = np.nan
    return out


def _read_fasta_seq(fa_path: str, contig: Optional[str] = None) -> str:
    """Read a single contig from a fasta file. If `contig` is None, return
    the first record. Strips whitespace; preserves uppercase as-is."""
    seq, cur_name, capture = [], None, False
    with open(fa_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                name = line[1:].split()[0] if len(line) > 1 else ""
                if contig is None:
                    if cur_name is not None:
                        break
                    cur_name, capture = name, True
                else:
                    capture = (name == contig)
                    if capture:
                        cur_name = name
            elif capture:
                seq.append(line.strip())
    if not seq:
        raise SystemExit(f"contig '{contig or '(first)'}' not found in {fa_path}")
    return "".join(seq)


def _read_struct_gt(path: str, contig: Optional[str] = None) -> str:
    """Read dot-bracket structure GT. Accepts:
      - plain text: one or more lines containing '.()' chars (concatenated)
      - fasta-like: header lines starting with '>' skipped (matches legacy
        miR17-92_GT.txt format)
    """
    parts = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            parts.append(line.strip())
    s = "".join(parts)
    if not s or not all(c in ".()" for c in s):
        raise SystemExit(f"struct-gt {path}: not a clean dot-bracket string "
                          f"(got {len(s)} chars, sample={s[:40]!r})")
    return s


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--mod-rate-csv", required=True,
                   help="mod_rate.csv from `segshape mod-calling` (cols: "
                        "pos_idx, mod_rate; z-score recomputed from mod_rate)")
    p.add_argument("--ref-fa", required=True,
                   help="fasta with the contig (used for ref_len + RNAfold input)")
    p.add_argument("--contig", default=None,
                   help="contig name; default = first record in --ref-fa")
    p.add_argument("--struct-gt",
                   help="dot-bracket ground-truth file (optional; needed for "
                        "MCC/F1/AUC)")
    p.add_argument("--react-gt",
                   help="reactivity .dat ground-truth (optional; needed for "
                        "Spearman; 1-indexed pos\\treact, -999=NA)")
    p.add_argument("--par-path",
                   help="RNAfold thermodynamic params file (e.g. "
                        "rna_andronescu2007.par); required iff --struct-gt is set")
    p.add_argument("--out-prefix", required=True,
                   help="output base; per-offset .dat / .centroid go under "
                        "{prefix}/, summary TSV at {prefix}__{variant}.tsv")
    p.add_argument("--variant-name", default="default",
                   help="tag for the summary TSV row + per-offset filenames")
    p.add_argument("--offsets", default="0,1,2,3,4",
                   help="comma-separated kmer-offsets to test (default 5-mer)")
    p.add_argument("--edge-mask", type=int, default=20,
                   help="positions on each end excluded from Spearman / AUC")
    p.add_argument("--save-dat-dir", default=None,
                   help="if given, also copy each .dat + .centroid here")
    return p


def run(args: argparse.Namespace) -> int:
    df_mr = pd.read_csv(args.mod_rate_csv)
    if "pos_idx" not in df_mr.columns or "mod_rate" not in df_mr.columns:
        raise SystemExit(f"--mod-rate-csv {args.mod_rate_csv}: missing "
                          f"required columns (pos_idx, mod_rate)")
    df_mr = df_mr.sort_values("pos_idx").reset_index(drop=True)
    L = int(df_mr["pos_idx"].max()) + 1
    rates = np.full(L, np.nan)
    rates[df_mr["pos_idx"].astype(int).to_numpy()] = df_mr["mod_rate"].to_numpy()
    if "reactivity_z" in df_mr.columns and df_mr["reactivity_z"].notna().any():
        rates_z = np.full(L, np.nan)
        rates_z[df_mr["pos_idx"].astype(int).to_numpy()] = df_mr["reactivity_z"].to_numpy()
    else:
        rates_z = _zscore(rates)
    n_eval = int(np.isfinite(rates).sum())
    print(f"variant={args.variant_name}  L={L}  n_eval={n_eval}", flush=True)

    ref_seq = _read_fasta_seq(args.ref_fa, args.contig)
    ref_len = len(ref_seq)
    print(f"  ref_len={ref_len} (contig='{args.contig or '(first)'}')",
          flush=True)

    offsets = [int(x) for x in args.offsets.split(",")]

    react_gt = None
    if args.react_gt and os.path.exists(args.react_gt):
        rg = pd.read_csv(args.react_gt, sep="\t", header=None,
                         names=["pos", "react"])
        rg.loc[rg["react"] == -999, "react"] = np.nan
        react_gt = rg.set_index("pos")["react"]

    if args.struct_gt:
        struct_gt = _read_struct_gt(args.struct_gt, args.contig)
        if len(struct_gt) != ref_len:
            print(f"  WARNING: struct-gt len {len(struct_gt)} != ref_len "
                  f"{ref_len}; AUC mapping may be off-by-one", flush=True)
    else:
        struct_gt = None

    react_corr = (correlate_react(rates_z, react_gt, ref_len, args.edge_mask,
                                    offsets)
                  if react_gt is not None else
                  {i: float("nan") for i in offsets})
    struct_aucs = (auc_struct(rates, struct_gt, ref_len, args.edge_mask, offsets)
                   if struct_gt is not None else
                   {i: float("nan") for i in offsets})
    print(f"  react Spearman: "
          f"{[round(react_corr[i], 3) for i in offsets]}", flush=True)
    print(f"  struct AUC    : "
          f"{[round(struct_aucs[i], 3) for i in offsets]}", flush=True)

    out_root = args.out_prefix
    os.makedirs(out_root, exist_ok=True)
    if args.save_dat_dir:
        os.makedirs(args.save_dat_dir, exist_ok=True)

    struct_metrics_per_off: Dict[int, dict] = {}
    mfe_metrics_per_off: Dict[int, dict] = {}
    can_fold = (struct_gt is not None and args.par_path
                and os.path.exists(args.par_path))
    if struct_gt is not None and not can_fold:
        print("  WARNING: --struct-gt provided but --par-path missing/invalid; "
              "skipping RNAfold + MCC", flush=True)

    for off in offsets:
        shape_path = os.path.join(out_root,
                                   f"{args.variant_name}_off{off}.dat")
        n_valid = write_reactivity_dat(rates_z, ref_len, off, shape_path)
        if not can_fold:
            struct_metrics_per_off[off] = {
                "precision": np.nan, "recall": np.nan,
                "f1": np.nan, "mcc": np.nan,
            }
            mfe_metrics_per_off[off] = struct_metrics_per_off[off]
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                mfe_pred, centroid_pred = run_rnafold_with_shape(
                    args.ref_fa, args.par_path, shape_path, tmpdir)
            except Exception as e:
                print(f"  off={off}: RNAfold FAILED: {e}", flush=True)
                struct_metrics_per_off[off] = {
                    "precision": np.nan, "recall": np.nan,
                    "f1": np.nan, "mcc": np.nan,
                }
                mfe_metrics_per_off[off] = struct_metrics_per_off[off]
                continue
            sm = calc_struct_metrics(struct_gt, centroid_pred)
            sm_mfe = calc_struct_metrics(struct_gt, mfe_pred)
            struct_metrics_per_off[off] = sm
            mfe_metrics_per_off[off] = sm_mfe
            print(f"  off={off}: n_valid={n_valid}  "
                  f"centroid: P={sm['precision']:.2f} R={sm['recall']:.2f} "
                  f"F1={sm['f1']:.2f} MCC={sm['mcc']:.2f}  "
                  f"mfe: MCC={sm_mfe['mcc']:.2f}", flush=True)
            cent_path = os.path.join(out_root,
                                      f"{args.variant_name}_off{off}.centroid")
            with open(cent_path, "w") as f:
                f.write(centroid_pred + "\n")
            if args.save_dat_dir:
                shutil.copy(shape_path, args.save_dat_dir)
                shutil.copy(cent_path, args.save_dat_dir)

    def _best(d, default_key=offsets[0]):
        finite = {k: v for k, v in d.items()
                  if isinstance(v, (int, float)) and np.isfinite(v)}
        if not finite:
            return default_key
        return max(finite, key=finite.get)

    best_react_off = _best(react_corr)
    best_struct_off = _best(struct_aucs)
    best_mcc_off = _best({k: v["mcc"] for k, v in struct_metrics_per_off.items()})

    row = {
        "variant": args.variant_name,
        "n_eval": n_eval,
        "L": L,
        "ref_len": ref_len,
        **{f"react_off{i}": react_corr[i] for i in offsets},
        **{f"auc_off{i}": struct_aucs[i] for i in offsets},
        **{f"mcc_off{i}": struct_metrics_per_off[i]["mcc"] for i in offsets},
        **{f"f1_off{i}": struct_metrics_per_off[i]["f1"] for i in offsets},
        **{f"mfe_mcc_off{i}": mfe_metrics_per_off[i]["mcc"] for i in offsets},
        "best_react": react_corr[best_react_off],
        "best_react_off": best_react_off,
        "best_auc": struct_aucs[best_struct_off],
        "best_auc_off": best_struct_off,
        "best_mcc": struct_metrics_per_off[best_mcc_off]["mcc"],
        "best_mcc_off": best_mcc_off,
        "best_f1": struct_metrics_per_off[best_mcc_off]["f1"],
        "best_precision": struct_metrics_per_off[best_mcc_off]["precision"],
        "best_recall": struct_metrics_per_off[best_mcc_off]["recall"],
    }
    out_tsv = f"{out_root}__{args.variant_name}.tsv"
    pd.DataFrame([row]).to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {out_tsv}", flush=True)
    print(f"Best: react={react_corr[best_react_off]:.3f}@off{best_react_off}  "
          f"auc={struct_aucs[best_struct_off]:.3f}@off{best_struct_off}  "
          f"MCC={struct_metrics_per_off[best_mcc_off]['mcc']:.2f}@off{best_mcc_off}",
          flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="segshape evaluate pipeline")
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
