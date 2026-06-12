# segSHAPE tutorial

End-to-end walkthrough of the segSHAPE pipeline, from raw nanopore signal to
SHAPE-constrained RNA secondary structure. **Every command below uses the
built-in defaults** — those defaults are the parameter set used in the
segSHAPE paper, so a clean run reproduces the published results without any
extra tuning flags.

The pipeline has **7 steps**. Five map onto a `segshape` sub-command; step 2
(basecalling) and step 7 (folding) wrap external tools (Dorado / ViennaRNA):

| step | what | command | external tool |
|---|---|---|---|
| 1 | fast5 → pod5, then build a read-id index | `pod5 convert` + `segshape pod5index` | `pod5` |
| 2 | basecalling with move table | `dorado basecaller` | Dorado |
| 3 | extract per-read alignment + signal interval | `segshape dorado-extract` | — |
| 4 | segmentation: pod5 → peaks → subevents | `segshape segment` | — |
| 5 | anchored event-alignment + per-read scaling | `segshape event-align` | — |
| 6 | per-position modification / reactivity calling | `segshape mod-calling` | — |
| 7 | SHAPE-constrained structure prediction | `segshape fold` | ViennaRNA |

Two auxiliary commands work on the outputs of any step:

- `segshape evaluate` — per-read LL filter, per-position summary, and
  structure / reactivity scoring against a ground truth.
- `segshape plot` — diagnostic plots (segmentation QC, alignment path,
  signal trajectory, Dorado move table).

Run `segshape --help` or `segshape <command> --help` at any time to see every
flag for a sub-command.

> **Running on a cluster?** Every step is a plain `segshape` command, so it
> drops straight into a SLURM `sbatch` script or any job scheduler — the CLI
> is fully usable on a single machine, no special wrappers required.

## Install

Requires Python ≥ 3.10 and the `RNAfold` binary (ViennaRNA) on `PATH` for
step 7.

```bash
pip install -e .            # from a source checkout
conda install -c bioconda viennarna     # provides RNAfold for step 7
```

Dorado (step 2) and the `pod5` CLI (step 1) are installed separately from
Oxford Nanopore. The bundled ONT k-mer tables used by step 5 ship inside the
package — nothing extra to download.

## Directory layout

`segshape` expects a fixed `datasets/<DATASET>/<SAMPLE>/` tree. Each dataset
has two samples — `control` (SHAPE-reagent-free, e.g. DMSO) and `treated`
(SHAPE-reagent-exposed, e.g. NAI-N3 / 1M7) — and a shared `reference/`. After
a full run the tree looks like:

```
datasets/
└── <DATASET>/                      # e.g. riboswitch_wt
    ├── reference/
    │   └── <DATASET>.fa            # transcript FASTA, shared by both samples
    ├── control/
    │   ├── 1_raw_signal/
    │   │   ├── pod5/               # *.pod5            (step 1)
    │   │   └── pod5.index          # read_id → pod5    (step 1)
    │   ├── 2_base_called/
    │   │   └── dorado-<VER>/
    │   │       └── dorado.sorted.bam[.bai]            (step 2)
    │   └── 3_alignment/
    │       ├── dorado.extract_mv.csv                  (step 3)
    │       ├── subevents.norm.parquet                 (step 4)
    │       └── <sweep_cell>/        # alignment.csv, scale.csv,
    │           └── ...              # pos_kmer_table.csv (step 5)
    └── treated/                     # same structure as control/
        └── ...
```

The `<root>` you pass as `--root-dir` is the directory that *contains*
`datasets/`. Throughout this tutorial we use the worked example
`riboswitch_wt` (an RNA004 NAI-N3 riboswitch); substitute your own dataset
name and reference FASTA as needed.

---

## Step 1 — fast5 → pod5 + index

Dorado consumes **pod5**. If your data is distributed as multi-fast5, convert
it one-to-one, then build a `read_id → pod5 filename` index so later steps can
resolve a read to its source file in O(1).

```bash
# 1a — convert (skip if your data is already pod5)
pod5 convert fast5 \
    datasets/riboswitch_wt/control/1_raw_signal/multi_fast5/*.fast5 \
    --output datasets/riboswitch_wt/control/1_raw_signal/pod5/ --one-to-one ...

# 1b — index the pod5 folder (writes a sibling pod5.index)
segshape pod5index datasets/riboswitch_wt/control/1_raw_signal/pod5
```

The index is a small Parquet file (`read_id`, `filename`). If the same
`read_id` appears in more than one pod5 file, add `--verify-dups`: it
compares a per-read fingerprint across files and aborts if two different
reads truly collide (otherwise the duplicate is a benign chunk overlap, safe
to dedup downstream).

```bash
segshape pod5index <pod5_dir> --force --verify-dups
```

`--force` overwrites an existing index; `-o PATH` writes elsewhere.

---

## Step 2 — basecalling with the move table (Dorado)

Basecall with inline minimap2 alignment to the reference. Three options are
essential for the downstream steps:

- `--emit-moves` — exports the move table (step 3 uses it to find the signal
  interval covering only the aligned bases).
- `--reference` — inline alignment so reads carry reference coordinates.
- `--estimate-poly-a` — polyA length, used by step 5 as a transcript-start
  seed.

```bash
dorado basecaller <model> \
    datasets/riboswitch_wt/control/1_raw_signal/pod5/ \
    --reference datasets/riboswitch_wt/reference/ref_wt.fa \
    --emit-moves --estimate-poly-a \
    > dorado.bam
samtools sort dorado.bam -o \
    datasets/riboswitch_wt/control/2_base_called/dorado-<VER>/dorado.sorted.bam
samtools index .../dorado.sorted.bam
```

Use the chemistry-matched model: `rna002_70bps_hac@v3` (RNA002, Dorado 0.9.6)
or `rna004_130bps_sup@v5.3.0` (RNA004, Dorado 1.4.0). Keeping the Dorado
version in the output directory name lets you re-basecall without overwriting.

Quick sanity check that basecalling produced mappable reads:

```bash
samtools quickcheck dorado.sorted.bam && \
samtools view -c -F 2308 dorado.sorted.bam     # primary aligned count
```

---

## Step 3 — extract per-read CSV (`dorado-extract`)

Parse the BAM into one row per primary mapped read, deriving from the move
table the signal-sample interval that covers the aligned bases.

```bash
segshape dorado-extract \
    --root-dir <root> --dataset riboswitch_wt --sample control
# → datasets/riboswitch_wt/control/3_alignment/dorado.extract_mv.csv
```

`--bam PATH` / `--out PATH` override path resolution directly.

**Default filters** (each reported separately in the run summary):

| flag | default | meaning |
|---|---|---|
| `-F 2324` | on | drop unmapped / reverse / secondary / supplementary |
| `--min-mapq` | 20 | drop low mapping quality |
| `--drop-split-reads` | on | drop split-read children (`pi:Z` tag) |
| `--min-ref-coverage` | 0.8 | keep reads covering ≥ 80 % of the reference |
| `--drop-dup-reads` | on | keep first occurrence of each `read_id` |

Pass `--min-mapq 0`, `--min-ref-coverage 0.0`, `--no-drop-split-reads`, etc.
to relax any filter; `-F 2308` keeps reverse-strand reads.

---

## Step 4 — segmentation (`segment`)

For each whitelisted read, clip the signal to the basecaller-consumed
interval, detect level transitions with `find_peaks`, and emit one subevent
per inter-peak segment (`mean_pa`, `std_pa` per subevent).

```bash
segshape segment \
    --root-dir <root> --dataset riboswitch_wt --sample control
# → datasets/riboswitch_wt/control/3_alignment/subevents.norm.parquet
```

The default applies **median/MAD per-read normalization** (`--norm med-mad`),
which makes the signal chemistry-agnostic and pairs with the bundled ONT
normalized k-mer tables used in step 5. The legacy raw-pA mode
(`--norm none`, writing `subevents.parquet`) is kept for debugging.

Key defaults (all at the tuned production values — see the table from
`segshape segment --help` for the rest):

| flag | default | meaning |
|---|---|---|
| `--peak-distance` | 10 | min sample distance between detected peaks |
| `--smooth` | 3 | box-car smoothing of \|slope\| before find_peaks |
| `--trim` | 0.1 | trim 10 % off each end of a subevent before computing mean/std |
| `--resplit-std` | norm-aware (0.15 σ for med-mad, 3.0 pA for none) | re-split a subevent whose raw std exceeds this; pass `0` to disable |
| `-j` | 1 | pod5-file-level parallelism |

`--resplit-std` defaults to the value matching the active `--norm`; pass an
explicit number (or `0`) to override. Use `-j N` to parallelize across
pod5 files.

> **QC plot:** `segshape plot segment --root-dir <root> --dataset riboswitch_wt
> --sample control` overlays the subevents on the raw signal so you can
> eyeball peak placement and level fitting.

Run step 4 (and steps 1–3) for **both** `control` and `treated`.

---

## Step 5 — anchored event-alignment (`event-align`)

Map each read's subevents onto reference k-mer positions with an
anchored Viterbi DP, fit a per-read shift, and (for `control`) refine the
k-mer model with an EM pass that produces `pos_kmer_table.csv`. The
`treated` sample then aligns against control's frozen table.

```bash
# control first — produces pos_kmer_table.csv
segshape event-align \
    --root-dir <root> --dataset riboswitch_wt --sample control \
    --reference-file datasets/riboswitch_wt/reference/ref_wt.fa --contig TETRA

# treated second — reuses control's pos_kmer_table.csv automatically
segshape event-align \
    --root-dir <root> --dataset riboswitch_wt --sample treated \
    --reference-file datasets/riboswitch_wt/reference/ref_wt.fa --contig TETRA
```

Outputs land in a sweep-cell-named subdir under `3_alignment/`:

```
3_alignment/rna004_de50_dk15_bc0.0_sp50_shift_only/
├── alignment.csv          # read_idx, event_idx, pos_idx
├── scale.csv              # per-read calibration + DP corners + LL
└── pos_kmer_table.csv     # control only — EM-refined k-mer μ/σ
```

The chemistry (`--rna {002,004}`) and the signal domain are auto-detected; the
k-mer table and the per-read shift bounds are selected accordingly. The DP
geometry defaults (`--delta-event 50`, `--delta-kmer 15`, `--boundary-cost 0`,
`--skip-penalty 50`, `--sigma-multiplier 1.5`, `--length-weight capped`,
`--fit-mode shift_only`) are the production values; you normally do not need
to touch them.

> **Always run `control` before `treated`** for the same dataset — `treated`
> loads control's `pos_kmer_table.csv` and aborts if it is missing.

---

## Step 6 — modification / reactivity calling (`mod-calling`)

Compare control vs treated event distributions per reference position and emit
a reactivity profile. The default method `if-1D` fits an IsolationForest on the
control events and scores the treated events as an outlier rate.

```bash
segshape mod-calling \
    --root-dir <root> --dataset riboswitch_wt \
    --sweep-cell rna004_de50_dk15_bc0.0_sp50_shift_only \
    --ref-fa datasets/riboswitch_wt/reference/ref_wt.fa
```

This writes a per-run folder under the treated sample's `mod_rate/`:

```
mod_rate/default_if-1D_c0.0050/
├── mod_rate.csv                       # pos_idx, mod_rate (raw rate only)
└── reactivity_smooth0_norm-zscore.dat # 1-based z-scored reactivity, RNAfold-ready (needs --ref-fa)
```

Production defaults:

| flag | default | meaning |
|---|---|---|
| `--method` | `if-1D` | IsolationForest outlier-rate test |
| `--contamination` | 0.005 | IF control-tail fraction (matches `--nu` / `--gmm-quantile`) |
| `--smooth-window` | 0 | no smoothing (raw per-position z-score); set 5 for the nanoSHAPE convention |
| `--normalize` | zscore | per-position normalization written to the `.dat` reactivity |

Other methods are available for comparison (`ks`, `wass`, `dmed`,
`ocsvm-1D/2D`, `gmm-1D/2D`, `if-2D`, `xpore`) — pass e.g.
`--method ks,wass` or `--method all`. `--smooth-window` / `--normalize` are
post-processing knobs encoded in the `.dat` filename, so you can sweep them
without re-running the metric.

> `mod_rate.csv` holds only the raw `mod_rate` (independent of
> smooth/normalize), so it is stable across post-processing sweeps. The
> z-scored reactivity lives in the `.dat`, which is only written when
> `--ref-fa` is given (it carries the `pos_idx → 1-based reference position`
> transform RNAfold needs). `segshape fold` / `segshape evaluate` recompute
> the z-score from `mod_rate` when they need it.

---

## Step 7 — SHAPE-constrained structure prediction (`fold`)

Feed the reactivity profile to ViennaRNA `RNAfold` as a Deigan
pseudo-energy constraint (`-p -d2 --noLP --shape=DAT --shapeMethod=D`).

```bash
segshape fold \
    --mod-rate-csv mod_rate/default_if-1D_c0.0050/mod_rate.csv \
    --ref-fa datasets/riboswitch_wt/reference/ref_wt.fa --contig TETRA
```

`fold` reads the `mod_rate` column and always feeds RNAfold the
**per-position z-score computed on the fly** from it (no smoothing). If you
want a different post-processing (e.g. smoothing, or the SHAPE 2–8 %
normalization), produce the `.dat` in step 6 with the desired
`--smooth-window` / `--normalize` and pass that `.dat` to RNAfold directly.

Outputs alongside the input:

- `<variant>.shape` — the 1-indexed `--shape=DAT` file (`-999` = missing)
- `<variant>.bracket` — sequence + MFE + centroid dot-bracket structures
- `<variant>.summary.tsv` — energies and pair counts

Requires `RNAfold` on `PATH` (`conda install -c bioconda viennarna`).

---

## Evaluation

Score a predicted structure against a dot-bracket ground truth (base-pair
precision / recall / F1 / MCC for both the MFE and centroid structures):

```bash
segshape evaluate fold-out \
    --rnafold-out <run>/reactivity_smooth0_norm-zscore.out \
    --shape-dat   <run>/reactivity_smooth0_norm-zscore.dat \
    --struct-gt   datasets/riboswitch_wt/reference/structure_gt.txt
```

`segshape evaluate filter-ll` (per-read log-likelihood filtering) and
`segshape evaluate build-summary` (per-position aggregation) are the other
evaluation entry points. See `segshape evaluate --help`.

---

## Quick reference — full run for one dataset

```bash
ROOT=/path/to/data            # directory containing datasets/
DS=riboswitch_wt
REF=datasets/$DS/reference/ref_wt.fa
CONTIG=TETRA
CELL=rna004_de50_dk15_bc0.0_sp50_shift_only

for S in control treated; do
    segshape pod5index datasets/$DS/$S/1_raw_signal/pod5 --force
    # (basecall with Dorado → 2_base_called/dorado-<VER>/dorado.sorted.bam)
    segshape dorado-extract --root-dir $ROOT --dataset $DS --sample $S
    segshape segment        --root-dir $ROOT --dataset $DS --sample $S
done

# control before treated (treated reuses control's k-mer table)
segshape event-align --root-dir $ROOT --dataset $DS --sample control \
    --reference-file $REF --contig $CONTIG
segshape event-align --root-dir $ROOT --dataset $DS --sample treated \
    --reference-file $REF --contig $CONTIG

segshape mod-calling --root-dir $ROOT --dataset $DS --sweep-cell $CELL --ref-fa $REF
segshape fold --mod-rate-csv \
    datasets/$DS/treated/3_alignment/$CELL/mod_rate/default_if-1D_c0.0050/mod_rate.csv \
    --ref-fa $REF --contig $CONTIG
```
