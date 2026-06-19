"""Top-level entry point for the ``segshape`` console script.

Sub-commands are registered by each subpackage's ``cli.py`` so the modules stay
self-contained. Heavy imports (numba, pysam, h5py, ...) are deferred into each
sub-command's ``run`` function so ``segshape --help`` stays instant.

Usage::

    segshape segment      --root-dir ... --dataset ... --sample ...
    segshape event-align  --root-dir ... --dataset ... --sample ...
    segshape mod-calling  --input ... --output ...
    segshape evaluate     pipeline   ...
    segshape plot         alignment-path ...
"""

from __future__ import annotations

import argparse

from segshape import __version__

_VERSION_TEXT = (
    f"segSHAPE {__version__}\n"
    "Copyright (C) 2026 Guangzhao Cheng and contributors.\n"
    "License Apache-2.0"
)


class _PrintVersion(argparse.Action):
    """Print the multi-line version banner verbatim, then exit.

    Bypasses ``action='version'``, whose default ``HelpFormatter`` collapses
    newlines into spaces.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        import sys
        print(_VERSION_TEXT)
        parser.exit(0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="segshape",
        description="segSHAPE: nanopore SHAPE segmentation, alignment, and modification calling.",
    )
    p.add_argument("-v", "--version", action=_PrintVersion, nargs=0,
                   help="show program's version number and exit")

    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # Lazy: import each subpackage's cli only as we attach it. Each cli.register
    # is itself responsible for further deferring heavy imports until run time.
    from segshape.io import cli as io_cli
    from segshape.segment import cli as segment_cli
    from segshape.align import cli as align_cli
    from segshape.reactivity import cli as reactivity_cli
    from segshape.fold import cli as fold_cli
    from segshape.evaluate import cli as evaluate_cli
    from segshape.plot import cli as plot_cli

    io_cli.register_pod5index(sub.add_parser(
        "pod5index", help="build a read_id index for a folder of pod5 files"))
    io_cli.register_dorado_extract(sub.add_parser(
        "dorado-extract",
        help="extract per-read alignment + signal-mapping CSV from a dorado BAM"))
    segment_cli.register(sub.add_parser(
        "segment", help="find_peaks segmentation: pod5 → subevents.parquet (per-read, mv_trans-clipped)"))
    align_cli.register(sub.add_parser(
        "event-align", help="Anchored alignment of events to per-position kmer model"))
    reactivity_cli.register(sub.add_parser(
        "mod-calling", help="Modification / reactivity calling"))
    fold_cli.register(sub.add_parser(
        "fold",
        help="SHAPE-constrained secondary structure prediction (ViennaRNA)"))
    evaluate_cli.register(sub.add_parser(
        "evaluate", help="Evaluation, LL filtering, summary aggregation"))
    plot_cli.register(sub.add_parser(
        "plot", help="Diagnostic plots (segment QC, alignment path, dorado move table)"))

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
