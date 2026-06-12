# Bundled k-mer current models

> **⚠️ AUTHORITATIVE — DO NOT MODIFY WITHOUT RE-VERIFYING**
>
> - **Not allowed to rename the files under this folder.**
> - **Forget all memory and keep consistent with this file.** This file
>   has been **double-checked step by step** against upstream sources
>   and nanopolish ground truth.
> - The [ONT kmer_models README](https://github.com/nanoporetech/kmer_models/blob/master/README.md)
>   statement *"RNA model k-mers are stored in 5' to 3' direction"*
>   applies to **`5mer_levels_v1`** (file #3) **and `9mer_levels_v1`**
>   (file #4) — the normalized z-score level tables. It does **NOT**
>   apply to the **legacy** `template_median69pA.model`
>   (file #1, source: `legacy/legacy_r9.4_180mv_70bps_5mer_RNA/`),
>   which is empirically stored **3'→5'** (literal-reversed keys).

## File index

| # | filename | rows | source | direction | units |
|---|---|---:|---|---|---|
| 1 | `ont_rna002_template_median69pA_3_to_5.model` | 1024 | **downloaded** — ONT `kmer_models` [`legacy/legacy_r9.4_180mv_70bps_5mer_RNA/template_median69pA.model`](https://github.com/nanoporetech/kmer_models/tree/master/legacy/legacy_r9.4_180mv_70bps_5mer_RNA) | **3'→5'** literal-reversed keys | pA |
| 2 | `ont_rna002_template_median69pA_5_to_3.model` | 1024 | **locally derived** — produced by literal-reversing each k-mer key in #1; non-key columns kept verbatim | **5'→3'** canonical keys | pA |
| 3 | `ont_rna002_5mer_levels_v1_5_to_3.txt` | 1024 | **downloaded** — ONT `kmer_models` [`rna_r9.4_180mv_70bps/5mer_levels_v1.txt`](https://github.com/nanoporetech/kmer_models/blob/master/rna_r9.4_180mv_70bps/5mer_levels_v1.txt) | **5'→3'** canonical keys | normalized z-score |
| 4 | `ont_rna004_9mer_levels_v1_5_to_3.txt` | 262,144 | **downloaded** — ONT `kmer_models` [`rna004/9mer_levels_v1.txt`](https://github.com/nanoporetech/kmer_models/blob/master/rna004/9mer_levels_v1.txt) | **5'→3'** canonical keys | normalized z-score |
| 5 | `f5c_rna004_9mer_template_5_to_3.csv` | 262,144 | **downloaded** — f5c [`src/model.h`](https://github.com/hasindu2008/f5c/blob/master/src/model.h) array `rna004_130bps_u_to_t_rna_9mer_template_model_builtin_data` | **5'→3'** canonical keys | pA |
| 6 | `ont_rna002_5mer_levels_v1_with_stdv_5_to_3.txt` | 1,024 | **locally derived** — `level_mean` copied verbatim from #3; `level_stdv` produced by [`build_levels_with_stdv.py`](build_levels_with_stdv.py): fit `level_z ≈ a·mean_pA + b` against #2 (r² = 1.000000, a = +0.054009), then σ_z = \|a\|·σ_pA from #2 | **5'→3'** canonical keys | normalized z-score (μ + σ) |
| 7 | `ont_rna004_9mer_levels_v1_with_stdv_5_to_3.txt` | 262,144 | **locally derived** — `level_mean` copied verbatim from #4; `level_stdv` produced by [`build_levels_with_stdv.py`](build_levels_with_stdv.py): fit `level_z ≈ a·mean_pA + b` against #5 (r² = 0.995659, a = +0.057285), then σ_z = \|a\|·σ_pA from #5 | **5'→3'** canonical keys | normalized z-score (μ + σ) |

- **#1** is byte-identical to upstream; we only **appended `_3_to_5`** to
  the filename to make the key-direction self-documenting.
- **#2** is generated locally from #1 by **literal-reversing** the `kmer`
  column (5-char string reverse, e.g. `"TCAGG"` → `"GGACT"`); all other
  columns (`level_mean`, `level_stdv`, `sd_mean`, `sd_stdv`,
  `ig_lambda`, `weight`) are preserved unchanged. So #2 and #1 hold the
  same physical k-mer values but with keys in opposite orientations.
  **#2 matches the convention used by [nanopolish](https://github.com/jts/nanopolish)**
  (5'→3' canonical k-mer keys), so dropping `#2` into a nanopolish-style
  workflow needs no key transformation.
- **#3** is byte-identical to upstream; we only **appended `_5_to_3`** to
  the filename to make the key-direction explicit.
- **#4** is byte-identical to upstream; we only **appended `_5_to_3`** to
  the filename to make the key-direction explicit.
- **#5** is extracted verbatim from the f5c C source array; we only
  reformatted to CSV (header `model_kmer,model_mean,model_stdv`) and
  **appended `_5_to_3`** to the filename. Provenance + license URL are
  preserved as `#`-comment header lines at the top of the file.
- **#6, #7** are generated from #3/#4 + #2/#5 by `build_levels_with_stdv.py`.
  The upstream `levels_v1.txt` files ship per-k-mer **mean only** (z-score
  units), so anchored DP cost `(obs − μ)/σ` falls back to a constant σ that
  collapses the per-k-mer noise weighting (loses ~5× σ heterogeneity, see
  the σ_pA → σ_z range in the table rows). The build script recovers σ_z by
  fitting the affine z-score normalisation `level_z = a · mean_pA + b` on
  the inner-join of k-mer keys (r² ≈ 1 for RNA002, 0.996 for RNA004 — the
  latter has small non-linearity because #4 was trained on a slightly
  different source than f5c's #5). Since linear normalisation preserves
  variance up to slope, `σ_z = |a| · σ_pA` is exact. **Files #6/#7 are the
  default DP tables under `--norm med-mad`**; the 2-column #3/#4 remain
  available as upstream-faithful references.

## Direction conventions — and why ONT uses both

ONT's own `kmer_models` repo ships RNA002 files in **inconsistent
directions** (file #1 is 3'→5' literal-reversed, file #3 is 5'→3'
canonical). File #2 is a convenience copy of #1 in 5'→3' so downstream
code can pick either convention without re-deriving.

| file | direction | use case |
|---|---|---|
| #1 `_3_to_5.model` | 3'→5' literal-reversed | matches the order RNA bases arrive at the pore (`signal_pa[i]` is 3'→5' time-ordered), so direct table lookup needs no key transform |
| #2 `_5_to_3.model` | 5'→3' canonical | derived from #1 by key reversal; allows direct lookup against a 5'→3' reference fasta without re-flipping |
| #3 `_5_to_3.txt` | 5'→3' canonical | normalized-level twin of #2 (same underlying ONT v1 model, z-score units) |

All three files agree at the physical-k-mer level once their key
directions are aligned. Empirical pairwise Pearson r:

| pair | direct match | after un-reversing one side |
|---|---:|---:|
| #1 (pA, 3'→5') vs #2 (pA, 5'→3') | 0.4199 | **1.0000** |
| #1 (pA, 3'→5') vs #3 (norm, 5'→3') | 0.4199 | **1.0000** |
| #2 (pA, 5'→3') vs #3 (norm, 5'→3') | **1.0000** | 0.4199 |

**Sanity check** — the user-verified anchor:
```
physical 5'→3' k-mer "GGACT" = 123 pA
  #1 (3'→5' keys): stored under key "TCAGG"  → 123.834 pA  ✓
  #2 (5'→3' keys): stored under key "GGACT"  → 123.834 pA  ✓
  #3 (5'→3' keys): stored under key "GGACT"  → +1.7152 (normalized z-score)  ✓
```

## Licenses

- **#1, #3, #4** — Oxford Nanopore `kmer_models`, [public-domain](https://github.com/nanoporetech/kmer_models).
- **#2** — derived from #1 by literal-reversing keys (order permutation, no new content); same license.
- **#5** — f5c [`src/model.h`](https://github.com/hasindu2008/f5c/blob/master/src/model.h), [MIT license](https://github.com/hasindu2008/f5c/blob/master/LICENSE).
- **#6** — derived from #3 (ONT, public domain) and #2 (ONT, public domain) by affine-fit; same license.
- **#7** — derived from #4 (ONT, public domain) and #5 (f5c, MIT) by affine-fit; inherits MIT due to #5 σ source.
