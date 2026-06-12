"""Build and load a per-folder pod5 read_id index.

The index is a Parquet file written next to the folder it indexes:
    /path/to/pod5_test/   →   /path/to/pod5_test.index

Each row maps a ``read_id`` to the basename of the pod5 file that contains it.
Lookup is then ``DataFrame.set_index('read_id').loc[id, 'filename']`` —
the actual signal access goes through ``pod5.DatasetReader`` on the folder,
which already handles cross-file batched selection efficiently.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import pod5


# Per-read fingerprint used by --verify-dups. Compares the biological /
# physical capture identity of a read; deliberately excludes per-file
# storage details (e.g. run_info_index, byte offsets) that legitimately
# differ when the same read is copied into two pod5 files by chunk-overlap.
# `run_info_index` is replaced by the resolved RunInfo content fields.
_FINGERPRINT_READ_FIELDS = (
    "num_samples", "start_sample", "median_before",
)
_FINGERPRINT_RUNINFO_FIELDS = (
    "acquisition_id", "acquisition_start_time",
    "flow_cell_id", "protocol_run_id", "sample_id", "sample_rate",
)


INDEX_SCHEMA_VERSION = "1"


def _iter_pod5_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.pod5"))
    if not files:
        raise FileNotFoundError(f"no *.pod5 files under {folder}")
    return files


def _collect_rows(pod5_files: Iterable[Path]) -> tuple[list[str], list[str]]:
    read_ids: list[str] = []
    filenames: list[str] = []
    try:
        from tqdm import tqdm
        it = tqdm(list(pod5_files), desc="indexing pod5", unit="file")
    except ImportError:
        it = pod5_files
    for p in it:
        with pod5.Reader(p) as r:
            ids = [str(x) for x in r.read_ids]
        read_ids.extend(ids)
        filenames.extend([p.name] * len(ids))
    return read_ids, filenames


def _warn_duplicates(read_ids: list[str], filenames: list[str]) -> int:
    counts = Counter(read_ids)
    dups = [rid for rid, c in counts.items() if c > 1]
    if not dups:
        return 0
    n_dup = sum(counts[rid] - 1 for rid in dups)
    sample = dups[:5]
    by_id: dict[str, list[str]] = {rid: [] for rid in sample}
    for rid, fn in zip(read_ids, filenames):
        if rid in by_id:
            by_id[rid].append(fn)
    detail = "; ".join(f"{rid} -> {by_id[rid]}" for rid in sample)
    warnings.warn(
        f"{len(dups)} read_id(s) appear in >1 pod5 file "
        f"({n_dup} duplicate row(s) total). All occurrences are kept. "
        f"Examples: {detail}",
        stacklevel=2,
    )
    return len(dups)


def _read_fingerprint(row: "pod5.ReadRecord") -> tuple:
    base = tuple(getattr(row, f) for f in _FINGERPRINT_READ_FIELDS)
    cal = row.calibration
    base += (cal.scale, cal.offset)
    ri = row.run_info
    base += tuple(getattr(ri, f) for f in _FINGERPRINT_RUNINFO_FIELDS)
    return base


def _verify_duplicates(folder: Path, df: pd.DataFrame) -> int:
    """For every read_id appearing in more than one pod5 file, fetch its
    record from each file and compare a biological-identity fingerprint
    (num_samples, start_sample, median_before, calibration, RunInfo content).

    Returns the number of dup ids verified to be byte-redundant copies.
    Raises ``ValueError`` listing the conflicts on any mismatch.
    """
    dup_df = df[df.duplicated("read_id", keep=False)]
    if dup_df.empty:
        return 0

    id_to_files = dup_df.groupby("read_id")["filename"].agg(list).to_dict()
    file_to_ids: dict[str, set[str]] = defaultdict(set)
    for rid, files in id_to_files.items():
        for f in files:
            file_to_ids[f].add(rid)

    print(
        f"verifying {len(id_to_files):,} duplicate read_id(s) across "
        f"{len(file_to_ids)} pod5 file(s) ...",
        file=sys.stderr,
    )

    fps: dict[tuple[str, str], tuple] = {}
    for fn, rids in file_to_ids.items():
        with pod5.Reader(folder / fn) as r:
            for row in r.reads(selection=list(rids)):
                fps[(str(row.read_id), fn)] = _read_fingerprint(row)

    conflicts = []
    for rid, files in id_to_files.items():
        per_file = [fps[(rid, f)] for f in files]
        if any(fp != per_file[0] for fp in per_file[1:]):
            conflicts.append((rid, files, per_file))

    if conflicts:
        lines = [
            f"--verify-dups: {len(conflicts)} duplicate read_id(s) have "
            f"INCONSISTENT metadata across pod5 files (likely a real id "
            f"collision, not chunk overlap):"
        ]
        for rid, files, per_file in conflicts[:5]:
            lines.append(f"  {rid}: {files}")
            for f, fp in zip(files, per_file):
                lines.append(f"    {f}: {fp}")
        if len(conflicts) > 5:
            lines.append(f"  ... and {len(conflicts) - 5} more")
        raise ValueError("\n".join(lines))

    print(
        f"--verify-dups: all {len(id_to_files):,} duplicate read_id(s) are "
        f"byte-redundant copies (identical fingerprint).",
        file=sys.stderr,
    )
    return len(id_to_files)


def index_path_for(folder: Path) -> Path:
    """Convention: ``/x/y/foo/`` → ``/x/y/foo.index`` (sibling of folder)."""
    folder = folder.resolve()
    return folder.parent / f"{folder.name}.index"


def build_index(
    folder: str | Path,
    output: str | Path | None = None,
    force: bool = False,
    verify_dups: bool = False,
) -> Path:
    """Scan ``folder`` for *.pod5 and write a Parquet index next to it.

    With ``verify_dups=True``, any duplicated read_id is cross-checked across
    its source pod5 files: their biological fingerprints (samples, start,
    calibration, run_info content) must match, otherwise the build raises
    ``ValueError``. Without verify_dups, duplicates are warned but kept.
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    out = Path(output).resolve() if output else index_path_for(folder)
    if out.exists() and not force:
        raise FileExistsError(
            f"{out} already exists; pass force=True / --force to overwrite")

    pod5_files = _iter_pod5_files(folder)
    print(f"indexing {len(pod5_files)} pod5 file(s) under {folder}",
          file=sys.stderr)

    read_ids, filenames = _collect_rows(pod5_files)
    n_dup_ids = _warn_duplicates(read_ids, filenames)

    df = pd.DataFrame({"read_id": read_ids, "filename": filenames})
    if verify_dups and n_dup_ids:
        _verify_duplicates(folder, df)
    df.attrs["schema_version"] = INDEX_SCHEMA_VERSION
    df.attrs["folder"] = str(folder)
    df.attrs["n_pod5_files"] = len(pod5_files)
    df.attrs["n_reads"] = len(df)
    df.attrs["n_duplicate_ids"] = n_dup_ids
    df.attrs["dups_verified"] = bool(verify_dups and n_dup_ids)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"wrote {out}  ({len(df):,} reads, "
        f"{df['read_id'].nunique():,} unique, {n_dup_ids} dup id(s))",
        file=sys.stderr,
    )
    return out


def load_index(index_path: str | Path) -> pd.DataFrame:
    """Load an index produced by :func:`build_index`. Columns: read_id, filename."""
    return pd.read_parquet(index_path)
