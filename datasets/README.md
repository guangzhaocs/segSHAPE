# segSHAPE benchmark datasets

Per-dataset reproduction artifacts for the segSHAPE manuscript. See
[reproduction.ipynb](reproduction.ipynb) for the full reproduction report (run it to
recompute the metrics). One folder per dataset.

| dataset | chemistry | contig | ref_len | contamination (if-1D) |
|---|---|---|---:|---:|
| miR17-92 | RNA002 | miR17-92 | 951 | 0.005 |
| tetra | RNA002 | tetra | 421 | 0.005 |
| bsub_16S | RNA002 | bacillus_subtilis_16S | 1552 | 0.005 |
| smPC_002pool_wt | RNA002 | TETRA (in a 16-RNA pool) | 421 | 0.005 |
| smPC_004pool_wt | RNA004 | TETRA (in a 16-RNA pool) | 421 | 0.02 |

The two `smPC_*pool_wt` datasets share a multi-record `reference.fa` (16 RNAs); their
contig-specific files are prefixed with the contig name (e.g. `TETRA_…`) so other contigs
can be added alongside later. The other three datasets are single-contig (no prefix).

## Files (per dataset)

Direction (3'→5' or 5'→3') and indexing base (0- or 1-based) are given per column where
they apply.

| file | contents | format & coordinates |
|---|---|---|
| `reference.fa` | reference sequence(s) | FASTA, **5'→3'** (single contig, or the full pool for `smPC_*`) |
| `structure_gt.txt` | ground-truth secondary structure | `>gt` + dot-bracket; **5'→3', 1-based** (position *i* = reference base *i*), length = ref_len |
| `pos_kmer_table.csv` | per-position k-mer model (fit on the control sample) | `pos` (**3'→5', 0-based**), `reference_pos` (**5'→3', 1-based**), `kmer` (string **5'→3' canonical**), `mean`, `stdv`, `n_obs` |
| `mod_rate_if_contam<c>.csv` | per-position **raw** modification rate, **if-1D** (contamination `<c>`) | `pos_idx` (**3'→5', 0-based**), `reference_pos` (**5'→3', 1-based**), `mod_rate` (`nan` = no coverage) |
| `mod_rate_gmm_q<q>.csv` | same, **gmm-1D** (quantile `<q>`) | same columns |
| `reactivity_norm_if_contam<c>.dat` | normalized reactivity (RNAfold SHAPE input), **if-1D** | two cols: `reference_pos` (**5'→3', 1-based**) ⟶ value (z-score; `-999` = dead-zone / no data) |
| `reactivity_norm_gmm_q<q>.dat` | same, **gmm-1D** | same format |
| `rnafold_if.out` | `RNAfold -p -d2 --noLP --noPS --shapeMethod=D` output, **if-1D** | header / sequence / MFE / ensemble / centroid structures; all **5'→3', 1-based** |
| `rnafold_gmm.out` | same, **gmm-1D** | as above |

`if-1D` is the manuscript's mod-calling method (contamination `c`); `gmm-1D` is an
alternative, fully-deterministic method (quantile `q` = 0.005 RNA002 / 0.02 RNA004). Both
ship the full chain (`mod_rate` + `reactivity` + `rnafold_*.out`). GMM is not reported in
the manuscript for the two `smPC_*pool_wt` datasets.

## Coordinate conventions & conversion

There are **two coordinate systems**. Mixing them up is the most common source of error, so
both are written explicitly in every table.

**1. `pos_idx` — internal index, 3'→5', 0-based.** `pos_idx = 0` is the **3'-most** position;
the range is `0 … ref_len-5` (`ref_len-4` positions, the same for RNA002 and RNA004).
ONT sequences RNA 3'→5' (the molecule passes through the pore 3' end first), and the
alignment DP indexes positions in that signal-time order. Used by `pos_kmer_table.csv`
(`pos` column) and `mod_rate_if_contam<c>.csv` (`pos_idx` column).

**2. `reference_pos` — 1-based, 5'→3'.** The standard FASTA / SHAPE / RNAfold / `.ct`
position: `1` is the **5'** base, `ref_len` is the **3'** base. Used directly by
`structure_gt.txt`, `reactivity_norm_if_contam<c>.dat`, and `rnafold.out`, and added as a
column to `pos_kmer_table.csv` and `mod_rate_if_contam<c>.csv` for convenience.

**Conversion** (the k-mer **center** base maps to `reference_pos`):

```
reference_pos = ref_len - 2 - pos_idx
pos_idx       = ref_len - 2 - reference_pos
```

Example (tetra, ref_len = 421): `pos_idx 0 → reference_pos 419` (near 3' end),
`pos_idx 416 → reference_pos 3` (near 5' end).

**Dead zone.** The k-mer center covers reference positions `[3, ref_len-2]` only;
positions `{1, 2, ref_len-1, ref_len}` are never assigned a k-mer center (a 4-nt SHAPE
dead zone) and appear as `-999` in the `.dat`. (RNA004 uses a 9-mer model with edge padding
so its SHAPE-callable range matches RNA002's.)

### k-mer string direction ≠ index direction

In `pos_kmer_table.csv` the **`kmer` string is 5'→3' canonical** (e.g. for tetra,
`pos = 0 → kmer = ACTCG`, which is `reference.fa[417..421]` read 5'→3'), matching the
nanopolish / dorado `model_kmer` convention — **even though the `pos` index runs 3'→5'**.
The two are independent: the index direction follows the alignment DP geometry, while the
string spelling follows the k-mer-model key convention. Use `reference_pos` (not `pos`) to
join any of these files to the reference, structure, or reactivity.

## Quick join recipe

To line everything up in 5'→3' reference coordinates: take `reference_pos` from
`pos_kmer_table.csv` / `mod_rate_if_contam<c>.csv`, and index `reference.fa` /
`structure_gt.txt` / `reactivity_norm_if_contam<c>.dat` at the same 1-based position.
