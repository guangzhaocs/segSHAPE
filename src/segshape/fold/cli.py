"""CLI surface for `segshape fold`.

Thin glue: registers `rnafold.add_arguments` and dispatches to `rnafold.run`.
"""

from __future__ import annotations

import argparse


def register(p: argparse.ArgumentParser) -> None:
    from segshape.fold import rnafold
    rnafold.add_arguments(p)
    p.set_defaults(func=rnafold.run)
