"""SHAPE-constrained secondary structure prediction via ViennaRNA RNAfold.

Step 7 of the segshape pipeline. Consumes ``mod_rate.csv`` produced by
``segshape mod-calling`` (columns: ``pos_idx, mod_rate``) and emits the
predicted dot-bracket structure(s) for the given reference contig.

Pipeline:
  1. Load mod_rate.csv + reference fasta.
  2. Map pos_idx → 1-indexed reference base position. With our universal
     anchor offset (``(k-1)//2 - edge_pad == 2`` for both RNA002 5-mer and
     RNA004 9-mer) this is just ``ref_pos = ref_len - pos_idx - 2``.
  3. Write a 1-indexed ``.shape`` file (``-999`` for missing positions).
     The reactivity fed to RNAfold is always the per-position z-score
     computed on the fly from the ``mod_rate`` column.
  4. Run ``RNAfold -p -d2 --noLP --shape=DAT --shapeMethod=D``
     (Deigan pseudo-energy SHAPE constraint).
  5. Parse MFE + centroid dot-bracket lines and the MFE/ensemble energies.
  6. Write ``<variant>.shape`` (the input dat), ``<variant>.bracket``
     (sequence + structures), ``<variant>.summary.tsv`` (counts + energies).

Ground-truth comparison (MCC / AUC / Spearman vs --struct-gt / --react-gt)
is in ``segshape evaluate pipeline``; this module is prediction-only.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from segshape.data import rnafold_par_path


def _zscore(x: np.ndarray) -> np.ndarray:
    """Per-position z-score on the finite slots only; NaN slots stay NaN.
    Matches ``segshape mod-calling --normalize zscore`` (smooth window 0)
    so a recompute from ``mod_rate`` reproduces the dropped reactivity_z
    column."""
    out = x.copy().astype(np.float64)
    m = ~np.isnan(out)
    if m.sum() < 2:
        return out
    mu, sd = out[m].mean(), out[m].std()
    if sd > 0:
        out[m] = (out[m] - mu) / sd
    return out


# ---------------------------------------------------------------------------
# Reference fasta
# ---------------------------------------------------------------------------

def read_fasta_seq(fa_path: str, contig: Optional[str] = None) -> Tuple[str, str]:
    """Read a single contig from a fasta. ``contig=None`` → first record.

    Returns (name, sequence). Sequence is uppercase; ``U`` left as-is so
    RNAfold sees a real RNA string (RNAfold accepts both T and U)."""
    records: List[Tuple[str, list]] = []
    with open(fa_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                name = line[1:].split()[0] if len(line) > 1 else ""
                records.append((name, []))
            elif records:
                records[-1][1].append(line.strip())
    if not records:
        raise SystemExit(f"no fasta records in {fa_path}")
    if contig is None:
        name, parts = records[0]
    else:
        match = [r for r in records if r[0] == contig]
        if not match:
            have = ", ".join(r[0] for r in records[:6])
            raise SystemExit(
                f"contig '{contig}' not in {fa_path}; have: {have}"
                f"{' …' if len(records) > 6 else ''}")
        name, parts = match[0]
    seq = "".join(parts).upper()
    return name, seq


# ---------------------------------------------------------------------------
# pos_idx → reference position mapping
# ---------------------------------------------------------------------------

def pos_idx_to_ref_pos(pos_idx: int, ref_len: int, anchor_off: int = 2) -> int:
    """Map a kmer-axis ``pos_idx`` to its 1-indexed 5'→3' reference position.

    ``anchor_off = (k-1)//2 - edge_pad`` and equals 2 for both bundled
    chemistries (RNA002 5-mer / RNA004 9-mer). See
    ``src/segshape/align/anchored.py`` top docstring for the derivation.
    """
    return ref_len - pos_idx - anchor_off


# ---------------------------------------------------------------------------
# .shape (.dat) writer
# ---------------------------------------------------------------------------

def write_shape_dat(values: np.ndarray, ref_len: int, anchor_off: int,
                    out_path: str, na_value: float = -999.0) -> int:
    """Write a 1-indexed ``.shape`` file for ViennaRNA ``--shape=DAT``.

    ``values[i]`` is the per-pos_idx reactivity (NaN allowed).
    Returns the number of valid (non-NaN) positions written.

    ViennaRNA's ``--shape=DAT --shapeMethod=D`` (Deigan) expects:
        ``<position>\\t<reactivity>``  per line, 1-indexed, sorted ascending,
    with missing positions either omitted or set to a sentinel like -999.
    We write one line per ref position so the file's row count matches
    ref_len (matches the legacy `eval_full_pipeline.py` convention)."""
    L = len(values)
    fa_react = np.full(ref_len, na_value, dtype=np.float64)
    n_valid = 0
    for i in range(L):
        v = values[i]
        if not np.isfinite(v):
            continue
        fa = ref_len - i - anchor_off                # 1-indexed 5'→3' pos
        if 1 <= fa <= ref_len:
            fa_react[fa - 1] = float(v)
            n_valid += 1
    with open(out_path, "w") as f:
        for i, v in enumerate(fa_react):
            if v == na_value:
                f.write(f"{i + 1}\t{int(na_value)}\n")
            else:
                f.write(f"{i + 1}\t{v:.6f}\n")
    return n_valid


# ---------------------------------------------------------------------------
# RNAfold invocation + output parsing
# ---------------------------------------------------------------------------

# Energy parsing: lines like "....((((.... ( -12.34)", "((..)) [-15.7]",
# or centroid/MEA "((..)) { -1.30 d=2.30}" / "((..)) { -1.40 MEA=2.30}".
# RNAfold -p uses (), [], and {} as energy delimiters across the four lines,
# so the opener class must include all three.
_ENERGY_RX = re.compile(r"[\(\[\{]\s*(-?\d+\.\d+)")


def parse_rnafold_output(stdout: str) -> List[Tuple[str, Optional[float]]]:
    """Parse ``RNAfold -p`` stdout into ordered (dot_bracket, energy) tuples.

    ``-p`` typically emits MFE, ensemble (ufp/free energy), centroid, MEA.
    We scan every line and keep ones whose first token is a pure dot-bracket
    string. Energy (if present) is parsed from the trailing ``( -X.XX )`` or
    ``[ -X.XX ]`` group; ensemble-only lines with no dot-bracket prefix are
    skipped."""
    out: List[Tuple[str, Optional[float]]] = []
    for line in stdout.splitlines():
        toks = line.split()
        if not toks:
            continue
        first = toks[0]
        if any(c not in ".()" for c in first):
            continue
        m = _ENERGY_RX.search(line)
        energy = float(m.group(1)) if m else None
        out.append((first, energy))
    return out


def run_rnafold(seq: str, shape_path: Optional[str] = None,
                par_path: Optional[str] = None,
                shape_method: str = "D",
                contig_name: str = "seq",
                work_dir: Optional[str] = None,
                verbose: bool = False
                ) -> List[Tuple[str, Optional[float]]]:
    """Run ``RNAfold -p -d2 --noLP`` (with optional --shape / -P) on ``seq``.

    Returns the parsed ``[(dot_bracket, energy), …]`` list. Typically:
        index 0 → MFE structure
        index 1 → ensemble representative (e.g. MEA / centroid; format
                  varies by ViennaRNA version)
        index 2 → centroid
    The caller picks by index.

    Requires the ``RNAfold`` binary on PATH (ViennaRNA package).
    """
    cmd = ["RNAfold", "-p", "-d2", "--noLP", "--noPS"]
    if par_path:
        cmd += ["-P", par_path]
    if shape_path:
        cmd += [f"--shape={shape_path}", f"--shapeMethod={shape_method}"]

    fa = f">{contig_name}\n{seq}\n"
    if verbose:
        sys.stderr.write(f"RNAfold cmd: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, input=fa, capture_output=True, text=True,
                                cwd=work_dir, check=True)
    except FileNotFoundError as e:
        raise SystemExit(
            "RNAfold not found on PATH. Install ViennaRNA "
            "(`conda install -c bioconda viennarna`) or pass its bin path."
        ) from e
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"RNAfold failed (exit {e.returncode}).\n"
            f"stdout:\n{e.stdout}\nstderr:\n{e.stderr}") from e

    parsed = parse_rnafold_output(result.stdout)
    if len(parsed) < 2:
        raise SystemExit(
            f"RNAfold output has fewer than 2 dot-bracket lines:\n"
            f"{result.stdout}")
    return parsed


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_bracket_file(out_path: str, contig: str, seq: str,
                       parsed: List[Tuple[str, Optional[float]]]) -> None:
    """Write a human-readable summary of RNAfold output:
        >contig
        <seq>
        <mfe_dotbracket>            (-X.XX kcal/mol)    [MFE]
        <alt_dotbracket>            (-Y.YY kcal/mol)    [alt]

    ``parsed[-1]`` is whichever structure RNAfold prints last under
    ``-p -d2`` — typically the MEA structure (4 lines: MFE, ensemble,
    centroid, MEA). Labelled ``alt`` rather than ``centroid`` to avoid
    misnaming. Matches ``code/eval_full_pipeline.py`` legacy behaviour
    (which also took the last line) so MCC/F1 baselines stay comparable.
    """
    mfe_db, mfe_e = parsed[0]
    alt_db, alt_e = parsed[-1]
    with open(out_path, "w") as f:
        f.write(f">{contig}\n{seq}\n")
        f.write(f"{mfe_db}    "
                f"({mfe_e:.2f} kcal/mol)" if mfe_e is not None else mfe_db)
        f.write("    [MFE]\n")
        f.write(f"{alt_db}    "
                f"({alt_e:.2f} kcal/mol)" if alt_e is not None else alt_db)
        f.write("    [alt]\n")


def write_summary_tsv(out_path: str, contig: str, ref_len: int,
                      n_valid_shape: int, anchor_off: int,
                      parsed: List[Tuple[str, Optional[float]]]) -> None:
    """One-line summary TSV with counts + energies. ``alt_*`` columns track
    ``parsed[-1]`` (last RNAfold structure line — usually MEA under ``-p``)."""
    mfe_db, mfe_e = parsed[0]
    alt_db, alt_e = parsed[-1]
    pd.DataFrame([{
        "contig":        contig,
        "ref_len":       ref_len,
        "n_valid_shape": n_valid_shape,
        "anchor_off":    anchor_off,
        "mfe_energy":    "" if mfe_e is None else f"{mfe_e:.4f}",
        "alt_energy":    "" if alt_e is None else f"{alt_e:.4f}",
        "mfe_n_pairs":   mfe_db.count("("),
        "alt_n_pairs":   alt_db.count("("),
    }]).to_csv(out_path, sep="\t", index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g_in = p.add_argument_group("input")
    g_in.add_argument("--mod-rate-csv", required=True,
                      help="mod_rate.csv from `segshape mod-calling` "
                           "(cols: pos_idx, mod_rate)")
    g_in.add_argument("--ref-fa", required=True,
                      help="fasta with the reference contig")
    g_in.add_argument("--contig", default=None,
                      help="contig name (default: first record in fasta)")

    g_m = p.add_argument_group("method")
    g_m.add_argument("--anchor-off", type=int, default=2,
                     help="pos_idx → ref_pos offset; "
                          "(k-1)//2 - edge_pad. Default 2 (RNA002 + RNA004 with "
                          "bundled tables).")
    g_m.add_argument("--shape-method", default="D",
                     choices=["D", "W", "Z", "C"],
                     help="ViennaRNA --shapeMethod (default D = Deigan "
                          "pseudo-energy)")
    g_m.add_argument("--par-path", default="andronescu2007",
                     help="RNAfold thermodynamic params (-P). Accepts: "
                          "(a) bundled name 'andronescu2007' (default; "
                          "Andronescu et al. 2007 — matches the legacy "
                          "code/eval_full_pipeline.py baseline); "
                          "(b) 'turner2004' or 'none' to use ViennaRNA's "
                          "built-in Turner2004; (c) absolute path to a .par "
                          "file.")

    g_o = p.add_argument_group("output")
    g_o.add_argument("--out-dir", default=None,
                     help="write outputs here. Default: same dir as "
                          "--mod-rate-csv")
    g_o.add_argument("--variant-name", default=None,
                     help="basename used for output files. Default: stem of "
                          "--mod-rate-csv")
    g_o.add_argument("--keep-shape-dat", action="store_true",
                     help="also retain the intermediate .shape file written "
                          "for RNAfold (default: kept already; flag is a no-op "
                          "but documents intent for callers)")
    g_o.add_argument("--verbose", action="store_true")
    return p


def run(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.mod_rate_csv):
        raise SystemExit(f"mod_rate.csv not found: {args.mod_rate_csv}")
    if not os.path.isfile(args.ref_fa):
        raise SystemExit(f"ref-fa not found: {args.ref_fa}")

    contig, seq = read_fasta_seq(args.ref_fa, args.contig)
    ref_len = len(seq)

    par_path = args.par_path
    if par_path is None or par_path.lower() in ("none", "turner2004", "default"):
        par_path = None
        par_label = "ViennaRNA built-in (Turner2004)"
    elif os.path.isfile(par_path):
        par_label = par_path
    else:
        par_path = rnafold_par_path(par_path)
        if not os.path.isfile(par_path):
            raise SystemExit(f"--par-path: bundled file not found ({par_path})")
        par_label = par_path

    print(f"contig={contig}  ref_len={ref_len}  "
          f"anchor_off={args.anchor_off}", flush=True)
    print(f"par={par_label}", flush=True)

    df = pd.read_csv(args.mod_rate_csv)
    for col in ("pos_idx", "mod_rate"):
        if col not in df.columns:
            raise SystemExit(
                f"--mod-rate-csv missing column {col!r}; "
                f"have: {list(df.columns)}")
    df = df.sort_values("pos_idx")
    L = int(df["pos_idx"].max()) + 1
    pos = df["pos_idx"].astype(int).to_numpy()
    in_range = (pos >= 0) & (pos < L)

    # The reactivity fed to RNAfold is always the per-position z-score
    # computed on the fly from the raw mod_rate column (mod_rate.csv no
    # longer stores a reactivity_z column).
    raw = np.full(L, np.nan, dtype=np.float64)
    raw_vals = pd.to_numeric(df["mod_rate"], errors="coerce").to_numpy()
    raw[pos[in_range]] = raw_vals[in_range]
    values = _zscore(raw)
    print(f"  L (max pos_idx + 1) = {L}, "
          f"finite values = {int(np.isfinite(values).sum())} "
          f"(per-position z-score of mod_rate)", flush=True)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.mod_rate_csv))
    os.makedirs(out_dir, exist_ok=True)
    variant = args.variant_name or os.path.splitext(
        os.path.basename(args.mod_rate_csv))[0]
    shape_path = os.path.join(out_dir, f"{variant}.shape")
    bracket_path = os.path.join(out_dir, f"{variant}.bracket")
    summary_path = os.path.join(out_dir, f"{variant}.summary.tsv")

    n_valid = write_shape_dat(values, ref_len, args.anchor_off, shape_path)
    print(f"wrote {shape_path}  ({n_valid} non-NA positions / {ref_len})",
          flush=True)
    if n_valid == 0:
        raise SystemExit(
            "no valid SHAPE constraints (all NaN). Check --anchor-off, or "
            "rerun mod-calling with adjusted thresholds.")

    with tempfile.TemporaryDirectory() as work_dir:
        parsed = run_rnafold(seq, shape_path=shape_path,
                             par_path=par_path,
                             shape_method=args.shape_method,
                             contig_name=contig,
                             work_dir=work_dir, verbose=args.verbose)
    print(f"  RNAfold returned {len(parsed)} structure line(s)", flush=True)
    mfe_db, mfe_e = parsed[0]
    alt_db, alt_e = parsed[-1]
    print(f"  MFE : {mfe_e if mfe_e is not None else 'n/a'} kcal/mol  "
          f"({mfe_db.count('(')} pairs)", flush=True)
    print(f"  alt : {alt_e if alt_e is not None else 'n/a'} kcal/mol  "
          f"({alt_db.count('(')} pairs)  [parsed[-1]; usually MEA under -p]",
          flush=True)

    write_bracket_file(bracket_path, contig, seq, parsed)
    write_summary_tsv(summary_path, contig, ref_len, n_valid,
                      args.anchor_off, parsed)
    print(f"wrote {bracket_path}\nwrote {summary_path}", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="segshape fold")
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main() or 0)
