#!/usr/bin/env python3
"""Extract per-read alignment + signal-mapping summary from a dorado BAM
produced with `--emit-moves --estimate-poly-a`.

References:
  https://software-docs.nanoporetech.com/dorado/latest/basecaller/sam_spec/
  https://github.com/nanoporetech/dorado/issues/1428

Tested on:
  RNA002: Dorado v0.9.6
  RNA004: Dorado v0.1XXX

One row per primary mapped read. Output columns:
  read_id, flag, is_reverse, is_child, mapq, ref_name, ref_len,
  seq_len,
  align_start, align_end,           # 0-based half-open in basecalled seq
  ref_start,   ref_end,             # 0-based half-open in reference
  called_start, called_end,         # = (ts, ns); signal interval consumed by basecaller
                                    # NOTE: includes polyA + soft-clipped bases
  mv_trans_start, mv_trans_end,     # signal interval covering ONLY the aligned bases,
                                    # derived by walking the mv table (see notes)
  ts, ns,                           # raw spec values (kept for convenience)
  stride, n_moves,                  # mv stride and count of '1' moves
  polya_tail,                       # poly-A length in BASES, -1 if absent
  polya_start, polya_end,           # primary polyA range in signal samples
  polya_sec_start, polya_sec_end,   # secondary polyA (plasmids); -1 if none

Spec / direction notes
----------------------
Signal direction
  RNA is sequenced 3' -> 5' (motor protein feeds the 3' end first), so
  raw signal time order is 3' -> 5'. Basecalled SEQ in the BAM
  (record.query_sequence) is 5' -> 3' as usual.

mv tag
  mv:B:c, [stride, m_0, m_1, ...] in SIGNAL order (3' -> 5' for RNA).
  m_j == 1 means a new base is emitted at signal sample (ts + j*stride).
  The j-th '1' in mv corresponds to base index (seq_len - 1 - j) in the
  5' -> 3' SEQ. The mv body covers polyA + mRNA body together: polyA bases
  appear FIRST in mv body (signal 3' -> 5'), at the END of SEQ (5' -> 3').

ts / ns
  ts:i = number of samples trimmed from the START of the signal.
  ns:i = upper index (exclusive) of the basecaller-consumed interval; SEQ
         and mv table both correspond to signal[ts : ns]. ns reflects rear
         trimming (if any).
  A tail of up to stride-1 samples that doesn't fill a full stride block
  is dropped:  len(mv body) == (ns - ts) // stride.
  called_start = ts, called_end = ns (Python-style half-open). This is the
  basecaller input window and INCLUDES the polyA tail; the mRNA-body signal
  window must be derived downstream from polya_end (when pa:B:i is present)
  or by walking mv past the first polya_tail '1' moves.

Poly-A (latest spec, RNA002 + RNA004)
  pt:i   = poly(A/T) tail length in BASES.
  pa:B:i = 5-int array of signal-sample positions:
              [anchor, start, end, sec_start, sec_end]
           sec_* = -1 unless this is a plasmid with a secondary polyA.
           We discard arr[0] (anchor) -- it is the search seed and not
           useful downstream; only start/end are kept.
  Old RNA002 dorado wrote pa:i:length as a scalar; we detect that and
  fall back to using it as polya_tail (logged as legacy form).

mv-derived aligned signal interval (mv_trans_start, mv_trans_end)
  Walk the mv body to translate SEQ alignment coordinates into a signal
  interval that excludes polyA, 5'/3' soft clip and adapter remnants.
  The j-th '1' in mv body (0-indexed) corresponds to base index
  seq_len - 1 - j in 5'->3' SEQ. With aligned region SEQ[align_start:align_end):
      k_first = seq_len - align_end + 1   # 1-based '1'-rank, 3'-most aligned base
      k_last  = seq_len - align_start     # 1-based '1'-rank, 5'-most aligned base
      j_first = position of k_first-th '1'
      mv_trans_start = ts + j_first * stride
      mv_trans_end   = ts + j_{k_last+1} * stride       (k_last < seq_len)
                     = ts + mv_body_len * stride         (k_last == seq_len)
  This is the interval segSHAPE downstream wants for per-read signal
  windowing. -1 if mv is unavailable or the read failed sanity checks.

Child reads
  pi:Z = parent read id, set on records that come from a split read.
         is_child = 1 when present (this record is only a segment of the
         parent's signal); = 0 for unsplit reads.

Reverse strand
  Reverse-strand reads (FLAG & 16) are filtered upstream by
  `samtools view -F 2324` and skipped here too. For reverse reads pysam
  reports SEQ as its reverse-complement, so q_start/q_end would not be
  in the basecall frame and the mv table would need separate RC handling.
  Pass --include-reverse to keep them; the caller is responsible.

Uniqueness of read_id
  This script is a faithful BAM dump: one CSV row per primary-mapped
  forward record (after the filters above). It does NOT guarantee unique
  read_id. Some BAMs contain identical-coords primary forward duplicates
  (observed: tetra/treated, ~0.8% of non-child rows -- exact duplicate
  records on flag/ref_start/ref_end/mapq, root cause unknown but possibly
  from a re-mapping or batch concat in the upstream pipeline). A WARNING
  is printed at the end if any duplicates were written. Downstream
  consumers (e.g. anchored_alignment.load_extract_mv) must dedupe.

Usage
  segshape dorado-extract --bam /path/to/dorado.sorted.bam
  segshape dorado-extract --bam BAM --out CSV
  segshape dorado-extract --root-dir DIR --dataset tpp_bound --sample control

Add --strict to abort on the first sanity-check violation; default just
counts and reports at the end.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter

import pysam


DEFAULT_BAM = "dorado.sorted.bam"
DEFAULT_CSV = "dorado.extract_mv.csv"


def get_tag(read, name, default=None):
    try:
        return read.get_tag(name)
    except KeyError:
        return default


def resolve_paths(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve (bam_path, out_path).

    Path conventions in workflow mode:
        BAM in : <root>/datasets/<DATASET>/<SAMPLE>/2_base_called/dorado-*/<DEFAULT_BAM>
        CSV out: <root>/datasets/<DATASET>/<SAMPLE>/3_alignment/<DEFAULT_CSV>

    The BAM is glob-matched on ``dorado-*`` to support multiple dorado
    versions (RNA002 → 0.9.6, RNA004 → 1.4.0). Multiple matches → fail
    fast and ask the user to disambiguate via ``--bam``.
    """
    if args.bam:
        bam_path = args.bam
        out_path = args.out or os.path.join(
            os.path.dirname(bam_path) or ".", DEFAULT_CSV)
    else:
        if not (args.root_dir and args.dataset and args.sample):
            raise SystemExit(
                "ERROR: either --bam (with optional --out) "
                "or all of --root-dir, --dataset, --sample must be given")
        sample_dir = os.path.join(
            args.root_dir, "datasets", args.dataset, args.sample)
        candidates = sorted(glob.glob(
            os.path.join(sample_dir, "2_base_called", "dorado-*", DEFAULT_BAM)))
        if not candidates:
            raise SystemExit(
                f"ERROR: no {DEFAULT_BAM} under "
                f"{sample_dir}/2_base_called/dorado-*/. "
                f"Pass --bam explicitly.")
        if len(candidates) > 1:
            raise SystemExit(
                f"ERROR: multiple {DEFAULT_BAM} found under "
                f"{sample_dir}/2_base_called/dorado-*/: {candidates}. "
                f"Pass --bam explicitly to disambiguate.")
        bam_path = candidates[0]
        out_path = args.out or os.path.join(
            sample_dir, "3_alignment", DEFAULT_CSV)

    if not os.path.isfile(bam_path):
        raise SystemExit(f"ERROR: BAM not found: {bam_path}")
    return bam_path, out_path


COLS = [
    "read_id", "flag", "is_reverse", "is_child", "mapq", "ref_name", "ref_len",
    "seq_len",
    "align_start", "align_end",
    "ref_start",   "ref_end",
    "called_start", "called_end",
    "mv_trans_start", "mv_trans_end",
    "ts", "ns",
    "stride", "n_moves",
    "polya_tail",
    "polya_start", "polya_end",
    "polya_sec_start", "polya_sec_end",
]


def _dorado_version_from_bam(bam: "pysam.AlignmentFile") -> str:
    """Return the VN (e.g. '1.4.0+ba44a013') of the first @PG with PN=dorado,
    or 'unknown' if no such line exists."""
    pg_list = bam.header.to_dict().get("PG", [])
    for pg in pg_list:
        if pg.get("PN") == "dorado":
            return pg.get("VN", "unknown")
    return "unknown"


def extract_mv(
    bam_path: str,
    out_path: str,
    *,
    filter_flag: int = 2324,
    keep_contigs: list[str] | None = None,
    min_mapq: int = 20,
    drop_dup_reads: bool = True,
    drop_split_reads: bool = True,
    min_ref_coverage: float = 0.8,
    strict: bool = False,
) -> int:
    """Extract per-read alignment + signal-mapping summary from a dorado BAM
    to ``out_path`` (CSV). Returns 0 on success.

    Filtering (in order of application):
      filter_flag         samtools-style ``-F`` mask. Default 2324 = 4+16+256+2048
                          (unmapped | reverse | secondary | supplementary).
      keep_contigs        whitelist of reference contig names to keep
                          (case-sensitive). None / empty list = keep all.
      min_mapq            skip records with ``read.mapping_quality < min_mapq``.
                          Default 20; pass 0 to disable.
      drop_split_reads    skip records carrying ``pi:Z`` (split-read children).
                          Default True.
      min_ref_coverage    skip records where (ref_end - ref_start) / ref_len < t.
                          Default 0.8; pass 0.0 to disable.
      drop_dup_reads      keep only the first occurrence of each read_id.
                          Default True.
    """
    keep_set: set[str] = set(keep_contigs) if keep_contigs else set()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"BAM in : {bam_path}", file=sys.stderr)
    print(f"CSV out: {out_path}", file=sys.stderr)
    print(
        f"filters: -F {filter_flag} "
        f"keep_contigs={sorted(keep_set) if keep_set else 'all'} "
        f"min_mapq={min_mapq} "
        f"drop_split_reads={drop_split_reads} "
        f"min_ref_coverage={min_ref_coverage} "
        f"drop_dup_reads={drop_dup_reads}",
        file=sys.stderr,
    )

    n_total = 0
    n_kept = 0
    n_unmapped = 0
    n_secondary = 0
    n_supplementary = 0
    n_reverse = 0
    n_other_flag = 0
    n_dropped_other_contig = 0
    n_dropped_low_mapq = 0
    n_dropped_split = 0
    n_dropped_low_cov = 0
    n_dropped_dup = 0
    n_no_mv = 0
    n_child = 0
    # Per-read_id counts among KEPT (written) records. Dorado/mm2 output is
    # not guaranteed to be unique on read_id even after the secondary/
    # supplementary/reverse filters above (observed: 0.78% of tetra/treated
    # primary forward records appear twice with identical fields). We do NOT
    # dedupe here -- the CSV stays a faithful BAM dump -- but we report a
    # warning so downstream consumers (e.g. anchored_alignment.load_extract_mv)
    # know to apply their own dedup.
    kept_id_counts: Counter = Counter()

    # Per-read sanity-check counters. Each is incremented per offending read.
    checks = {
        "no_mv_tag":           0,  # mv tag absent on a primary mapped read
        "no_ts_tag":           0,  # ts tag absent
        "no_ns_tag":           0,  # ns tag absent
        "stride_nonpos":       0,  # mv[0] <= 0
        "mv_nonbinary":        0,  # mv body has values not in {0, 1}
        "n_moves_ne_seqlen":   0,  # popcount(mv body) != seq_len
        "mv_span_ne_ns":       0,  # 0 <= (ns - ts) - mv_body_len*stride < stride fails
        "ts_negative":         0,  # ts < 0
        "ns_nonpos":           0,  # ns <= 0
        "seq_len_zero":        0,  # query_length == 0 (no SEQ stored)
        "qrange_bad":          0,  # 0 <= qstart < qend <= seq_len fails
        "rrange_bad":          0,  # rstart < rend fails
        "polya_tail_negative": 0,  # pt < -1
        "polya_array_bad_len": 0,  # pa:B:i length != 5
        "polya_array_legacy":  0,  # pa returned as a scalar (legacy RNA002 form)
        "polya_primary_bad":   0,  # 0 <= polya_start <= polya_end fails
        "mv_trans_bad":        0,  # 0 <= mv_trans_start < mv_trans_end <= ns fails
    }

    def violate(read, key, msg):
        checks[key] += 1
        if strict:
            raise AssertionError(f"[{key}] read={read.query_name}: {msg}")

    # Track which poly-A encoding each read carried so the first run on a
    # new BAM tells us what dorado actually wrote.
    polya_form_seen = {
        "pa_array":         0,  # spec form: pa:B:i array
        "pa_scalar_legacy": 0,  # legacy: pa:i scalar length
        "pt_only":          0,  # only pt, no pa at all
        "neither":          0,  # no polyA tags
    }

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam, \
         open(out_path, "w", newline="") as fh:
        # Provenance header — `#` lines are skipped by pd.read_csv(comment='#').
        # Two lines: software versions + filter parameters used for this run.
        from segshape import __version__ as _segshape_ver
        dorado_ver = _dorado_version_from_bam(bam)
        fh.write(
            f"# segshape_version={_segshape_ver} "
            f"dorado_version={dorado_ver} "
            f"bam={bam_path}\n"
        )
        fh.write(
            f"# filter_flag={filter_flag} "
            f"min_mapq={min_mapq} "
            f"min_ref_coverage={min_ref_coverage} "
            f"drop_split_reads={drop_split_reads} "
            f"drop_dup_reads={drop_dup_reads} "
            f"keep_contigs={','.join(sorted(keep_set)) if keep_set else 'all'}\n"
        )
        w = csv.writer(fh)
        w.writerow(COLS)

        ref_lengths = dict(zip(bam.references, bam.lengths))

        # Validate --keep-contig whitelist against BAM header up front.
        # An empty intersection would silently filter everything; warn so
        # the user can fix the typo instead of staring at a 0-row CSV.
        if keep_set:
            valid_contigs = set(bam.references)
            unknown = sorted(keep_set - valid_contigs)
            if unknown:
                print(
                    f"WARNING: --keep-contig values not in BAM header: {unknown}. "
                    f"Available contigs: {sorted(valid_contigs)}",
                    file=sys.stderr,
                )

        for read in bam:
            n_total += 1
            if read.flag & filter_flag:
                # Per-bit breakdown for diagnostics. A record may match
                # multiple bits; account it under the first one we see.
                if read.is_unmapped:
                    n_unmapped += 1
                elif read.is_secondary:
                    n_secondary += 1
                elif read.is_supplementary:
                    n_supplementary += 1
                elif read.is_reverse:
                    n_reverse += 1
                else:
                    n_other_flag += 1
                continue

            # Reverse-strand reads survive only when the user lifts the 0x10
            # bit from --filter-flag (default 2324 includes it). The mv-walking
            # math below assumes mv body is in 3'→5' signal time AND that
            # `read.query_sequence` is 5'→3' SEQ; pysam reports the reverse-
            # complement for reverse reads, breaking the
            # `seq_len - 1 - j` ↔ "5'→3' SEQ base" mapping. Reject explicitly.
            if read.is_reverse:
                raise SystemExit(
                    f"ERROR: reverse-strand read passed --filter-flag "
                    f"({filter_flag}) — mv-walking does not support reverse "
                    f"reads. Either keep the 0x10 bit in --filter-flag (the "
                    f"default 2324 does this) or pre-process the BAM to "
                    f"reverse-complement only the SEQ field. "
                    f"Offending read: {read.query_name}")

            if keep_set and read.reference_name not in keep_set:
                n_dropped_other_contig += 1
                continue

            if min_mapq > 0 and read.mapping_quality < min_mapq:
                n_dropped_low_mapq += 1
                continue

            mv     = get_tag(read, "mv")
            ts_raw = get_tag(read, "ts")
            ns_raw = get_tag(read, "ns")
            pt_raw = get_tag(read, "pt")
            pa_raw = get_tag(read, "pa")
            pi_raw = get_tag(read, "pi")  # parent read id (split read child)

            is_child = 1 if pi_raw is not None else 0
            if is_child:
                n_child += 1
            if drop_split_reads and is_child:
                n_dropped_split += 1
                continue

            seq_len = read.query_length or 0
            if seq_len == 0:
                violate(read, "seq_len_zero", "query_length is 0")

            # --- mv tag -----------------------------------------------------
            if mv is None:
                violate(read, "no_mv_tag", "mv tag missing")
                n_no_mv += 1
                stride = -1
                n_moves = -1
                mv_body_len = 0
            else:
                stride = int(mv[0])
                mv_body = [int(x) for x in mv[1:]]
                mv_body_len = len(mv_body)
                n_moves = sum(1 for x in mv_body if x == 1)

                if stride <= 0:
                    violate(read, "stride_nonpos", f"stride={stride}")
                if any(x not in (0, 1) for x in mv_body):
                    violate(read, "mv_nonbinary", "mv body has non-{0,1} values")
                if seq_len and n_moves != seq_len:
                    violate(read, "n_moves_ne_seqlen",
                            f"#moves={n_moves} seq_len={seq_len}")

            # --- ts / ns ----------------------------------------------------
            if ts_raw is None:
                violate(read, "no_ts_tag", "ts tag missing")
                ts = 0
            else:
                ts = int(ts_raw)
                if ts < 0:
                    violate(read, "ts_negative", f"ts={ts}")

            if ns_raw is None:
                violate(read, "no_ns_tag", "ns tag missing")
                ns = 0
            else:
                ns = int(ns_raw)
                if ns <= 0:
                    violate(read, "ns_nonpos", f"ns={ns}")

            # mv body covers signal[ts:ns]; tail of up to stride-1 samples
            # that doesn't fill a full stride block is dropped, so
            #     0 <= (ns - ts) - mv_body_len*stride < stride.
            if mv is not None and stride > 0 \
                    and ts_raw is not None and ns_raw is not None and ns > 0:
                residual = (ns - ts) - mv_body_len * stride
                if not (0 <= residual < stride):
                    violate(read, "mv_span_ne_ns",
                            f"residual={residual} stride={stride} "
                            f"mv_body*stride={mv_body_len * stride} ns-ts={ns - ts}")

            # --- alignment ranges -------------------------------------------
            qs = read.query_alignment_start
            qe = read.query_alignment_end
            if not (qs is not None and qe is not None and 0 <= qs < qe <= seq_len):
                violate(read, "qrange_bad",
                        f"qstart={qs} qend={qe} seq_len={seq_len}")

            rs = read.reference_start
            re_ = read.reference_end
            if not (rs is not None and re_ is not None and rs < re_):
                violate(read, "rrange_bad", f"rstart={rs} rend={re_}")

            ref_len = ref_lengths.get(read.reference_name, -1)
            if min_ref_coverage > 0.0 and rs is not None and re_ is not None \
                    and ref_len > 0:
                if (re_ - rs) / ref_len < min_ref_coverage:
                    n_dropped_low_cov += 1
                    continue

            # --- poly-A -----------------------------------------------------
            polya_tail = -1
            polya_start = polya_end = -1
            polya_sec_start = polya_sec_end = -1

            if pt_raw is not None:
                polya_tail = int(pt_raw)
                if polya_tail < -1:
                    violate(read, "polya_tail_negative", f"pt={polya_tail}")

            if pa_raw is None:
                polya_form_seen["pt_only" if pt_raw is not None else "neither"] += 1
            elif isinstance(pa_raw, int):
                # Legacy RNA002: pa:i:length as a scalar.
                polya_form_seen["pa_scalar_legacy"] += 1
                violate(read, "polya_array_legacy",
                        f"pa is scalar (legacy form), value={pa_raw}")
                if polya_tail == -1:
                    polya_tail = int(pa_raw)
                    if polya_tail < -1:
                        violate(read, "polya_tail_negative",
                                f"pa(legacy)={polya_tail}")
            else:
                arr = list(pa_raw)
                polya_form_seen["pa_array"] += 1
                if len(arr) != 5:
                    violate(read, "polya_array_bad_len",
                            f"pa:B:i length={len(arr)} expected 5")
                # arr[0] = anchor (search seed); discarded.
                if len(arr) >= 3:
                    polya_start = int(arr[1])
                    polya_end   = int(arr[2])
                    if not (0 <= polya_start <= polya_end):
                        violate(read, "polya_primary_bad",
                                f"start={polya_start} end={polya_end}")
                if len(arr) >= 5:
                    polya_sec_start = int(arr[3])
                    polya_sec_end   = int(arr[4])

            # --- mv-derived signal range for the ALIGNED bases ---------------
            # SEQ is 5'->3' with positions 0..seq_len-1; aligned region is
            # SEQ[align_start : align_end). The j-th '1' in mv body (0-indexed,
            # signal time order = 3'->5' for RNA) maps to base seq_len-1-j.
            # Convert SEQ indices to 1-based '1'-rank in mv:
            #   k_first = seq_len - align_end + 1   (3'-most aligned base)
            #   k_last  = seq_len - align_start     (5'-most aligned base)
            # Strict half-open signal interval covering the full last base:
            #   mv_trans_start = ts + j_first * stride
            #   mv_trans_end   = ts + j_{k_last+1} * stride            (k_last < seq_len)
            #                  = ts + mv_body_len * stride             (k_last == seq_len)
            mv_trans_start = -1
            mv_trans_end = -1
            if (mv is not None and seq_len > 0 and stride > 0
                    and qs is not None and qe is not None
                    and 0 <= qs < qe <= seq_len
                    and n_moves == seq_len):
                ones_positions = [j for j, m in enumerate(mv_body) if m == 1]
                k_first = seq_len - qe + 1   # 1-indexed
                k_last  = seq_len - qs       # 1-indexed
                j_first = ones_positions[k_first - 1]
                mv_trans_start = ts + j_first * stride
                if k_last < seq_len:
                    j_next = ones_positions[k_last]  # 0-indexed = k_last gives (k_last+1)-th '1'
                    mv_trans_end = ts + j_next * stride
                else:
                    mv_trans_end = ts + mv_body_len * stride
                if not (ts <= mv_trans_start < mv_trans_end <= ns):
                    violate(read, "mv_trans_bad",
                            f"mv_trans=[{mv_trans_start}, {mv_trans_end}) ts={ts} ns={ns}")

            if drop_dup_reads and kept_id_counts.get(read.query_name, 0) > 0:
                n_dropped_dup += 1
                continue

            kept_id_counts[read.query_name] += 1
            w.writerow([
                read.query_name,
                read.flag,
                int(read.is_reverse),
                is_child,
                read.mapping_quality,
                read.reference_name,
                ref_len,
                seq_len,
                read.query_alignment_start,
                read.query_alignment_end,
                read.reference_start,
                read.reference_end,
                ts,        # called_start
                ns,        # called_end
                mv_trans_start,
                mv_trans_end,
                ts,
                ns,
                stride,
                n_moves,
                polya_tail,
                polya_start,
                polya_end,
                polya_sec_start,
                polya_sec_end,
            ])
            n_kept += 1

    print(f"records={n_total} kept={n_kept}", file=sys.stderr)
    print(
        f"  filter_flag={filter_flag}: "
        f"unmapped={n_unmapped} secondary={n_secondary} "
        f"supplementary={n_supplementary} reverse={n_reverse} "
        f"other={n_other_flag}",
        file=sys.stderr,
    )
    print(
        f"  dropped_other_contig={n_dropped_other_contig} "
        f"dropped_low_mapq={n_dropped_low_mapq} "
        f"dropped_split={n_dropped_split} "
        f"dropped_low_cov={n_dropped_low_cov} "
        f"dropped_dup={n_dropped_dup} "
        f"no_mv={n_no_mv} child_reads={n_child}",
        file=sys.stderr,
    )

    # Warn (do not filter) if the same read_id was written more than once.
    n_unique_ids = len(kept_id_counts)
    n_dup_ids = sum(1 for v in kept_id_counts.values() if v > 1)
    n_dup_rows = sum(v - 1 for v in kept_id_counts.values() if v > 1)
    if n_dup_ids:
        max_count = max(kept_id_counts.values())
        sample = [(rid, c) for rid, c in kept_id_counts.items() if c > 1][:5]
        print(f"WARNING: {n_dup_ids} read_id(s) written multiple times "
              f"({n_dup_rows} duplicate rows, max occurrences={max_count}, "
              f"unique={n_unique_ids}/{n_kept} = "
              f"{100*n_unique_ids/max(1,n_kept):.2f}%). "
              f"CSV is a faithful BAM dump; downstream consumers must dedupe "
              f"by read_id.", file=sys.stderr)
        print(f"  example duplicate read_ids: {sample}", file=sys.stderr)

    bad = {k: v for k, v in checks.items() if v}
    if bad:
        print("sanity-check violations (per-read counts):", file=sys.stderr)
        for k, v in sorted(bad.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v}", file=sys.stderr)
    else:
        print("sanity checks: all reads passed", file=sys.stderr)

    print(f"poly-A tag forms seen: {polya_form_seen}", file=sys.stderr)
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bam",
        help="explicit BAM path (default: derived from --root-dir/--dataset/--sample)",
    )
    parser.add_argument(
        "--out",
        help=f"explicit CSV out path (default: <bam_dir>/{DEFAULT_CSV})",
    )
    parser.add_argument("--root-dir", help="project root for path derivation")
    parser.add_argument("--dataset", help="dataset name (e.g. tpp_bound)")
    parser.add_argument("--sample", help="sample name (e.g. control | treated)")
    parser.add_argument(
        "-F", "--filter-flag", type=int, default=2324, metavar="FLAG",
        help="samtools-style flag mask: skip records where (flag & FLAG) != 0. "
             "Default 2324 = 4|16|256|2048 (unmapped|reverse|secondary|"
             "supplementary). Pass 2308 to keep reverse-strand reads.",
    )
    parser.add_argument(
        "-q", "--min-mapq", type=int, default=20, metavar="MAPQ",
        help="skip records with mapping_quality < MAPQ "
             "(default: 20; pass 0 to disable).",
    )
    parser.add_argument(
        "--keep-contig",
        type=lambda s: [c.strip() for c in s.split(",") if c.strip()],
        default=[], metavar="LIST",
        help="comma-separated whitelist of reference contig names to keep "
             "(default: empty = keep all). Unknown contigs trigger a warning. "
             "Example: --keep-contig miR17-92,bacillus_subtilis_16S",
    )
    parser.add_argument(
        "--drop-dup-reads", action=argparse.BooleanOptionalAction, default=True,
        help="keep only the first occurrence of each read_id (default: on; "
             "handles dorado duplicate primary-forward records observed in "
             "some BAMs). Pass --no-drop-dup-reads to keep all duplicates.",
    )
    parser.add_argument(
        "--drop-split-reads", action=argparse.BooleanOptionalAction, default=True,
        help="skip records carrying the pi:Z parent-id tag (default: on; "
             "removes split-read children). Pass --no-drop-split-reads to "
             "keep them.",
    )
    parser.add_argument(
        "--min-ref-coverage", type=float, default=0.8, metavar="RATIO",
        help="keep only reads with (ref_end - ref_start) / ref_len >= RATIO "
             "(default: 0.8; pass 0.0 to disable).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="raise on the first sanity-check violation instead of just counting",
    )


def run(args: argparse.Namespace) -> int:
    bam_path, out_path = resolve_paths(args)
    return extract_mv(
        bam_path, out_path,
        filter_flag=args.filter_flag,
        keep_contigs=args.keep_contig,
        min_mapq=args.min_mapq,
        drop_dup_reads=args.drop_dup_reads,
        drop_split_reads=args.drop_split_reads,
        min_ref_coverage=args.min_ref_coverage,
        strict=args.strict,
    )
