#!/usr/bin/env python
"""Build the tiny ``tests/data/plot_example`` fixture used by ``docs/plot.md``.

This carves a handful of reads out of a full segSHAPE run and lays them out in
the standard ``datasets/<DATASET>/<SAMPLE>/`` tree, small enough to commit, so
the ``segshape plot`` sub-commands can be demonstrated end-to-end without
shipping a multi-GB pod5 set.

It auto-picks reads that survived every step (so all plots work on the same
reads): present in the pod5 index, a primary record in the BAM, a row in
``dorado.extract_mv.csv`` and ``subevents.norm.parquet``, and ``PASS`` in the
sweep cell's ``scale.csv`` — choosing reads that share one source pod5 file so
only that file is read.

What it copies, per chosen read:
  - 1_raw_signal/pod5/<one .pod5>      via ``pod5 filter``  (only the picks)
  - 1_raw_signal/pod5.index            via ``segshape pod5index``
  - 2_base_called/dorado-*/dorado.sorted.bam[.bai]  via ``samtools view -N``
  - 3_alignment/dorado.extract_mv.csv  header/comments + the picks' rows
  - 3_alignment/subevents.norm.parquet rows for the picks
  - 3_alignment/<CELL>/{alignment,scale,pos_kmer_table}.csv  rows for the picks
                                       (original read_idx preserved)

Re-run after changing SRC / CELL to regenerate. Requires the ``pod5`` and
``samtools`` CLIs and a ``segshape`` install on PATH.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

# ---- what to carve out -----------------------------------------------------
# Source full run: a ``<DATASET>/<SAMPLE>`` dir of a completed segSHAPE run.
# Override with PLOT_EXAMPLE_SRC=/path/to/datasets/<DATASET>/<SAMPLE>.
DEFAULT_SRC = ("/scratch/cs/infantbiome/chengg1/segSHAPE/"
               "datasets/miR17-92/control")
SRC = Path(os.environ.get("PLOT_EXAMPLE_SRC", DEFAULT_SRC))
DATASET = SRC.parent.name          # e.g. "miR17-92"
SAMPLE = SRC.name                  # e.g. "control"
CELL = os.environ.get("PLOT_EXAMPLE_CELL",
                      "rna002_norm_de50_dk15_bc0.0_sp50_lwnone_sm1.0_k50.0")
N_READS = 3

# ---- where the fixture lands ----------------------------------------------
DST_ROOT = Path(__file__).resolve().parent / "data" / "plot_example"
DST = DST_ROOT / "datasets" / DATASET / SAMPLE


def sh(*cmd: str) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pick_reads() -> tuple[list[str], Path]:
    """N PASS reads sharing one source pod5 file; returns (reads, pod5_src)."""
    scale = pd.read_csv(SRC / "3_alignment" / CELL / "scale.csv")
    passr = set(scale[scale.qc_tag == "PASS"].read_id)
    ext = set(pd.read_csv(SRC / "3_alignment" / "dorado.extract_mv.csv",
                          comment="#").read_id)
    idx = pd.read_parquet(SRC / "1_raw_signal" / "pod5.index")
    sub = pd.read_parquet(SRC / "3_alignment" / "subevents.norm.parquet",
                          columns=["read_id"])
    common = passr & ext & set(idx.read_id) & set(sub.read_id)
    idx_c = idx[idx.read_id.isin(common)]
    by = idx_c.groupby("filename")["read_id"].apply(list)
    by = by[by.map(len) >= N_READS]
    if by.empty:
        raise SystemExit(f"no pod5 file has >= {N_READS} eligible reads")
    fname = sorted(by.index)[0]                       # deterministic pick
    reads = sorted(by[fname])[:N_READS]
    return reads, SRC / "1_raw_signal" / "pod5" / fname


def main() -> None:
    print(f"source: {SRC}  cell: {CELL}")
    reads, pod5_src = pick_reads()
    print(f"picked {len(reads)} reads from {pod5_src.name}:")
    for r in reads:
        print("  ", r)

    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    pod5_dir = DST / "1_raw_signal" / "pod5"
    bam_glob = sorted(glob.glob(str(SRC / "2_base_called" / "dorado-*"
                                    / "dorado.sorted.bam")))
    if len(bam_glob) != 1:
        raise SystemExit(f"expected one dorado-*/dorado.sorted.bam, got {bam_glob}")
    bam_src = Path(bam_glob[0])
    bam_dir = DST / "2_base_called" / bam_src.parent.name
    aln_dir = DST / "3_alignment"
    cell_dir = aln_dir / CELL
    for d in (pod5_dir, bam_dir, cell_dir):
        d.mkdir(parents=True, exist_ok=True)

    ids_file = DST_ROOT / "read_ids.txt"
    ids_file.write_text("\n".join(reads) + "\n")

    # 1) pod5 -> a small single-file pod5 with only the picks, then index it.
    print("[1] pod5 filter + index")
    sh("pod5", "filter", str(pod5_src),
       "--output", str(pod5_dir / "plot_example.pod5"),
       "--ids", str(ids_file), "--missing-ok", "--force-overwrite")
    sh("segshape", "pod5index", str(pod5_dir), "--force")

    # 2) BAM -> subset to the picks, then index.
    print("[2] samtools subset BAM")
    out_bam = bam_dir / "dorado.sorted.bam"
    sh("samtools", "view", "-b", "-N", str(ids_file),
       "-o", str(out_bam), str(bam_src))
    sh("samtools", "index", str(out_bam))

    # 3) dorado.extract_mv.csv -> header/comment lines + the picks' rows.
    print("[3] dorado.extract_mv.csv")
    src_csv = SRC / "3_alignment" / "dorado.extract_mv.csv"
    comments = [ln for ln in src_csv.read_text().splitlines()
                if ln.startswith("#")]
    ext = pd.read_csv(src_csv, comment="#")
    ext = ext[ext.read_id.isin(reads)]
    with (aln_dir / "dorado.extract_mv.csv").open("w") as fh:
        fh.write("\n".join(comments) + "\n")
        ext.to_csv(fh, index=False)

    # 4) subevents.norm.parquet -> rows for the picks (preserve schema metadata).
    print("[4] subevents.norm.parquet")
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(SRC / "3_alignment" / "subevents.norm.parquet")
    sub = table.to_pandas()
    sub = sub[sub.read_id.isin(reads)].reset_index(drop=True)
    out = pa.Table.from_pandas(sub, preserve_index=False)
    out = out.replace_schema_metadata(table.schema.metadata)  # keep trim/norm
    pq.write_table(out, aln_dir / "subevents.norm.parquet")

    # 5) sweep cell alignment.csv + scale.csv -> the picks (read_idx preserved).
    print(f"[5] {CELL}/{{alignment,scale}}.csv")
    scale = pd.read_csv(SRC / "3_alignment" / CELL / "scale.csv")
    keep_idx = set(scale[scale.read_id.isin(reads)].read_idx)
    scale[scale.read_id.isin(reads)].to_csv(cell_dir / "scale.csv", index=False)
    aln = pd.read_csv(SRC / "3_alignment" / CELL / "alignment.csv")
    aln[aln.read_idx.isin(keep_idx)].to_csv(cell_dir / "alignment.csv",
                                            index=False)
    shutil.copy(SRC / "3_alignment" / CELL / "pos_kmer_table.csv",
                cell_dir / "pos_kmer_table.csv")

    print("\nDone ->", DST_ROOT)


if __name__ == "__main__":
    main()
