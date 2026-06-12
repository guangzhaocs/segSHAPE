"""CLI surface for the event-align stage.

Currently a single method (anchored): dorado-mv + minimap2 anchored alignment
of events to a per-position kmer model, with per-read scaling.

Usage::

    segshape event-align --root-dir ... --dataset ... --sample ... \\
                         --reference-file ... --contig ...
"""

from __future__ import annotations

import argparse


def register(parser: argparse.ArgumentParser) -> None:
    # Heavy imports stay local so `segshape --help` doesn't pull numba/numpy.
    from segshape.align import anchored

    anchored.add_arguments(parser)
    parser.set_defaults(func=anchored.run)
