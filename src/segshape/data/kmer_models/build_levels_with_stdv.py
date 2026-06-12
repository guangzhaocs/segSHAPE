"""Generate `*_levels_v1_with_stdv_5_to_3.txt` for RNA002 (5-mer) and RNA004
(9-mer) by combining the z-score levels (mean only) with per-kmer std taken
from the paired pA tables.

Why: the bundled `*_levels_v1_5_to_3.txt` files only ship per-kmer **mean**
levels (z-score units). The anchored DP cost
`-0.5 * ((obs - mu) / sigma)^2 - log(sigma)` needs per-kmer sigma too;
hardcoding sigma=1.0 (current `DEFAULT_NORM_SIGMA` in
`align/anchored.py`) over-estimates the true z-score-domain sigma by ~6x
(empirical typical sigma_z ≈ 0.16), flattening the DP landscape and
losing per-kmer discriminative weight.

Method: derive the affine transform (a, b) that maps the paired pA-domain
mean column to the z-score-domain level column via least-squares
regression on the inner-join of kmer keys. Then
    sigma_z_k = |a| * sigma_pA_k
which is exact for any linear z-score normalisation (subtract a constant,
divide by a constant). The fit also serves as a sanity check that the two
files truly trace the same underlying ONT v1 k-mer model.

Sources (all 5'->3' canonical keys; see README.md for provenance):
  RNA002 mean column : ont_rna002_5mer_levels_v1_5_to_3.txt
  RNA002 stdv source : ont_rna002_template_median69pA_5_to_3.model
  RNA004 mean column : ont_rna004_9mer_levels_v1_5_to_3.txt
  RNA004 stdv source : f5c_rna004_9mer_template_5_to_3.csv

Output format (whitespace-separated, no header, comment-prefixed provenance):
  # kmer  level_mean  level_stdv
  AAAAA   0.9087  0.0263
  ...
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_levels(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep=r"\s+", comment="#", header=None, names=["kmer", "level_mean"]
    )


def load_model(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#")
    return df[["kmer", "level_mean", "level_stdv"]].rename(
        columns={"level_mean": "mean_pA", "level_stdv": "stdv_pA"}
    )


def load_f5c_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    return df.rename(
        columns={"model_kmer": "kmer", "model_mean": "mean_pA",
                 "model_stdv": "stdv_pA"}
    )


def fit_affine(mean_pA: np.ndarray, level_z: np.ndarray
               ) -> tuple[float, float, float]:
    """Return (a, b, r2) so that  level_z ≈ a * mean_pA + b."""
    A = np.vstack([mean_pA, np.ones_like(mean_pA)]).T
    (a, b), *_ = np.linalg.lstsq(A, level_z, rcond=None)
    pred = a * mean_pA + b
    ss_res = float(((level_z - pred) ** 2).sum())
    ss_tot = float(((level_z - level_z.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), r2


def build(levels_path: Path, std_src_path: Path, std_src_loader,
          out_path: Path, label: str) -> None:
    lev = load_levels(levels_path)
    src = std_src_loader(std_src_path)
    df = lev.merge(src, on="kmer", how="inner")
    if len(df) != len(lev):
        raise SystemExit(
            f"[{label}] join lost rows: levels={len(lev)} src={len(src)} "
            f"merged={len(df)}")
    a, b, r2 = fit_affine(df["mean_pA"].to_numpy(), df["level_mean"].to_numpy())
    if r2 < 0.99:
        raise SystemExit(
            f"[{label}] affine fit r2={r2:.6f} < 0.99 — the pA table and the "
            "levels table likely don't trace the same underlying ONT model "
            "(or one is in the wrong direction). Refusing to emit σ_z.")
    df["level_stdv"] = np.abs(a) * df["stdv_pA"]

    print(f"[{label}]  N={len(df):,}  fit: level_z = {a:+.6f} * pA + {b:+.6f}"
          f"   r2={r2:.6f}")
    print(f"           sigma_z  min/median/max = "
          f"{df['level_stdv'].min():.4f} / {df['level_stdv'].median():.4f} / "
          f"{df['level_stdv'].max():.4f}")
    print(f"           sigma_pA min/median/max = "
          f"{df['stdv_pA'].min():.4f} / {df['stdv_pA'].median():.4f} / "
          f"{df['stdv_pA'].max():.4f}")

    out_df = df[["kmer", "level_mean", "level_stdv"]].copy()
    with open(out_path, "w") as f:
        f.write(f"# source mean : {levels_path.name}\n")
        f.write(f"# source stdv : {std_src_path.name}\n")
        f.write(f"# transform   : level_z = {a:+.8f} * mean_pA + {b:+.8f}  "
                f"(r2={r2:.8f})\n")
        f.write(f"# stdv rule   : sigma_z = |a| * sigma_pA "
                f"= {abs(a):.8f} * sigma_pA\n")
        f.write("# kmer\tlevel_mean\tlevel_stdv\n")
        for r in out_df.itertuples(index=False):
            f.write(f"{r.kmer}\t{r.level_mean:.6f}\t{r.level_stdv:.6f}\n")
    print(f"           wrote {out_path}  ({out_path.stat().st_size:,} bytes)")


def main() -> None:
    build(
        levels_path=HERE / "ont_rna002_5mer_levels_v1_5_to_3.txt",
        std_src_path=HERE / "ont_rna002_template_median69pA_5_to_3.model",
        std_src_loader=load_model,
        out_path=HERE / "ont_rna002_5mer_levels_v1_with_stdv_5_to_3.txt",
        label="RNA002 5-mer",
    )
    build(
        levels_path=HERE / "ont_rna004_9mer_levels_v1_5_to_3.txt",
        std_src_path=HERE / "f5c_rna004_9mer_template_5_to_3.csv",
        std_src_loader=load_f5c_csv,
        out_path=HERE / "ont_rna004_9mer_levels_v1_with_stdv_5_to_3.txt",
        label="RNA004 9-mer",
    )


if __name__ == "__main__":
    main()
