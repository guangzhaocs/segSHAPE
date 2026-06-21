# segSHAPE: RNA secondary structure prediction from nanopore direct RNA sequencing

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.06.15.732177-green)](https://www.biorxiv.org/content/10.64898/2026.06.15.732177)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/guangzhaocs/segSHAPE/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/guangzhaocs/segSHAPE?include_prereleases)](https://github.com/guangzhaocs/segSHAPE/releases)
[![PyPI](https://img.shields.io/pypi/v/segshape)](https://pypi.org/project/segshape/)
[![Downloads](https://static.pepy.tech/personalized-badge/segshape?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/segshape)

End-to-end pipeline: raw nanopore direct-RNA signal → SHAPE reactivity → RNA
secondary structure, supporting both **RNA002** and **RNA004** chemistries.

segSHAPE turns raw Oxford Nanopore direct-RNA signal into a SHAPE-constrained
RNA secondary structure: it segments the signal into subevents, aligns them to
a reference with an anchored Viterbi DP, calls per-position modification rates
by comparing a SHAPE-treated sample against an untreated control, and feeds the
resulting reactivity profile to ViennaRNA `RNAfold`.

> **Status**: v0.1.1. All 7 pipeline steps are implemented end-to-end
> (`pod5index → dorado-extract → segment → event-align → mod-calling → fold`,
> plus `evaluate` / `plot`). **Every command runs on built-in defaults that
> are the parameter set used in the segSHAPE paper**, so a clean run
> reproduces the published results without extra tuning flags.

## Install

Requires Python ≥ 3.10. The folding step (7) needs the ViennaRNA `RNAfold`
binary on `PATH`.

```bash
# 1. Create an isolated environment (Python ≥ 3.10)
conda create -n segshape python=3.10
conda activate segshape

# 2. Install segSHAPE from PyPI
pip install segshape

# 3. Provide RNAfold for the folding step (7); tested with ViennaRNA 2.7.0 and 2.7.2
conda install -c bioconda viennarna

# 4. Test the install
segshape --help
```

To install from source instead (for development):

```bash
git clone https://github.com/guangzhaocs/segSHAPE.git
cd segSHAPE
pip install -e .
```

The bundled ONT k-mer tables and RNAfold parameter file ship inside the
package — nothing extra to download. Basecalling (step 2) uses Dorado and the
`pod5` CLI, installed separately from Oxford Nanopore.

## Tutorial

End-to-end walkthrough — directory layout, per-step commands, and expected
outputs — in [docs/tutorial.md](https://github.com/guangzhaocs/segSHAPE/blob/main/docs/tutorial.md). Diagnostic plotting
(`segshape plot`), with a tiny ready-to-run example fixture, is in
[docs/plot.md](https://github.com/guangzhaocs/segSHAPE/blob/main/docs/plot.md).

## Public modules (CLI surface)

`segshape <module> ...` exposes **8 user-facing modules**:

| module | what |
|---|---|
| `pod5index` | scan a folder of `*.pod5` and write `pod5.index` (read_id → filename) |
| `dorado-extract` | extract per-read alignment coordinates + mv-derived signal intervals from Dorado BAM |
| `segment` | pod5 → `find_peaks` → `subevents.parquet` (per-read) |
| `event-align` | anchored Viterbi DP event→position alignment + per-read shift calibration |
| `mod-calling` | per-position modification-rate calling |
| `fold` | SHAPE-constrained secondary structure prediction (ViennaRNA RNAfold) |
| `evaluate` | per-read LL filter, structure scoring (precision/recall/F1/MCC) |
| `plot` | diagnostic plots (alignment path, dorado-mv, segment QC) |

Run `segshape <cmd> --help` (and `segshape <cmd> <step> --help`) to see every
parameter. Sub-command parsers register lazily, so `segshape --help` does not
import numba / scipy / sklearn.

## Pipeline overview

Raw signal → RNA secondary structure in **7 steps**. Most map 1:1 onto a
segshape module; step 2 wraps Dorado and step 7 wraps ViennaRNA `RNAfold`:

| step | what | command |
|---|---|---|
| 1 | fast5 → pod5 + read_id index | `pod5 convert` + `segshape pod5index` |
| 2 | basecalling with move table | `dorado basecaller` |
| 3 | dorado extract → per-read CSV | `segshape dorado-extract` |
| 4 | segmentation: pod5 → peaks → subevents | `segshape segment` |
| 5 | anchored event-alignment + per-read shift calibration | `segshape event-align` |
| 6 | per-position modification-rate calling | `segshape mod-calling` |
| 7 | SHAPE-constrained structure prediction | `segshape fold` |

Steps 3–5 outputs all live under `datasets/<DATASET>/<SAMPLE>/3_alignment/`.

## Datasets & reproduction

The benchmark datasets and a runnable reproduction report (regenerating the paper's
precision / recall / F1 / MCC) are in
[`datasets/`](https://github.com/guangzhaocs/segSHAPE/tree/main/datasets) — see
[`datasets/reproduction.ipynb`](https://github.com/guangzhaocs/segSHAPE/blob/main/datasets/reproduction.ipynb).

## Quick start

```bash
ROOT=/path/to/data            # directory containing datasets/
DS=riboswitch_wt
REF=datasets/$DS/reference/ref_wt.fa
CONTIG=TETRA
CELL=rna004_de50_dk15_bc0.0_sp50_shift_only

for S in control treated; do
    segshape pod5index    datasets/$DS/$S/1_raw_signal/pod5 --force
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

See [docs/tutorial.md](https://github.com/guangzhaocs/segSHAPE/blob/main/docs/tutorial.md) for each step's inputs, outputs, and
options.

## Library use

Every `cli.py` is thin glue: each leaf module exports `add_arguments(parser)`
and `run(args)`, so the same code is reachable from the CLI and from Python.

```python
from segshape.align import anchored

anchored.run(args)        # args: argparse.Namespace (or anything with the
                          # same attributes)
```

## Repo layout

```
segSHAPE/
├── pyproject.toml
├── src/segshape/                # the importable package
│   ├── cli.py                   # `segshape` console-script entry
│   ├── io/                      # steps 1+3 — pod5 index, dorado-extract, readers
│   ├── segment/                 # step 4 — pod5 → peaks → subevents.parquet
│   ├── align/                   # step 5 — anchored alignment
│   ├── reactivity/              # step 6 — modification calling
│   ├── fold/                    # step 7 — SHAPE-constrained RNAfold
│   ├── evaluate/                # MCC / reactivity eval, LL filter, summary
│   ├── plot/                    # diagnostic plots
│   └── data/                    # bundled ONT k-mer tables + RNAfold params
├── tests/                       # smoke / unit tests
└── docs/tutorial.md             # end-to-end walkthrough
```

## Tests

```bash
pip install -e .[dev]
pytest -q
```

Smoke tests verify package import, bundled-data presence, and that every leaf
CLI sub-command responds to `--help` cleanly, alongside unit tests for the
segmentation, calibration, and reactivity logic.

## License

Apache-2.0 — see [LICENSE](https://github.com/guangzhaocs/segSHAPE/blob/main/LICENSE).
