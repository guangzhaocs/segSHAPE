"""CLI surface for diagnostic plotting helpers.

Sub-sub-commands::

    segshape plot alignment-path ...   # side-by-side C++ vs Py path windows
    segshape plot trajectory     ...   # 20-sample trajectory snapshot
    segshape plot dorado-mv      ...   # one read's signal + mv table overlay
    segshape plot segment        ...   # subevents.parquet overlaid on signal
"""

from __future__ import annotations

import argparse


def register(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="kind", required=True, metavar="<kind>")

    from segshape.plot import alignment_path, dorado_mv, segment_qc, trajectory

    sp_a = sub.add_parser("alignment-path",
                          help="detailed start/middle/end window of one read")
    alignment_path.add_arguments(sp_a)
    sp_a.set_defaults(func=alignment_path.run)

    sp_t = sub.add_parser("trajectory",
                          help="20-sample chronological trajectory for a few reads")
    trajectory.add_arguments(sp_t)
    sp_t.set_defaults(func=trajectory.run)

    sp_d = sub.add_parser("dorado-mv",
                          help="signal + mv table overlay for one read")
    dorado_mv.add_arguments(sp_d)
    sp_d.set_defaults(func=dorado_mv.run)

    sp_s = sub.add_parser("segment",
                          help="subevents.parquet overlaid on raw pod5 signal")
    segment_qc.add_arguments(sp_s)
    sp_s.set_defaults(func=segment_qc.run)
