"""CLI surface for the segment stage (leaf command).

The legacy ``peaks`` / ``borders`` / ``events`` sub-sub-commands collapsed
into a single per-read pipeline: pod5 → find_peaks → subevents.parquet,
filtered by the dorado-extract whitelist (mv_trans range). The legacy
modules ``segment.peaks`` and ``segment.borders`` are kept on disk for
reference but no longer wired into the CLI.
"""

from __future__ import annotations

import argparse


def register(parser: argparse.ArgumentParser) -> None:
    """Register as a leaf sub-command (``segshape segment ...``)."""
    from segshape.segment import events
    events.add_arguments(parser)
    parser.set_defaults(func=events.run)
