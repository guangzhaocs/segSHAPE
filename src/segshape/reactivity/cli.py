"""CLI surface for `segshape mod-calling`.

Thin glue: registers `calling.add_arguments` and dispatches to `calling.run`.
"""

from __future__ import annotations

import argparse


def register(p: argparse.ArgumentParser) -> None:
    from segshape.reactivity import calling
    calling.add_arguments(p)
    p.set_defaults(func=calling.run)
