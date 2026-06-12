"""CLI surface for IO utilities (pod5 index, ...)."""

from __future__ import annotations

import argparse


def add_pod5index_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "folder",
        type=str,
        help="path to a folder of *.pod5 files",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="output index path (default: <folder>.index next to the folder)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="overwrite an existing index file",
    )
    parser.add_argument(
        "--verify-dups",
        action="store_true",
        help="when read_ids appear in >1 pod5 file, cross-check their "
             "metadata + run_info fingerprint and abort if any conflict",
    )


def run_pod5index(args: argparse.Namespace) -> int:
    """CLI wrapper. The underlying ``build_index`` raises typed exceptions
    so library users can branch on them; here we collapse the user-actionable
    ones (bad path, existing index, dup conflict) into a clean
    ``SystemExit`` so the CLI prints a single ``ERROR: ...`` line and exits 1
    instead of dumping a traceback into SLURM logs.
    """
    from segshape.io.pod5_index import build_index
    try:
        build_index(
            args.folder,
            output=args.output,
            force=args.force,
            verify_dups=args.verify_dups,
        )
    except (FileNotFoundError, NotADirectoryError, FileExistsError, ValueError) as e:
        # type(e).__name__ pinned in the message so SLURM log readers can
        # tell at a glance which guard fired (e.g. FileExistsError vs
        # ValueError-from-verify-dups), without needing the traceback.
        raise SystemExit(f"ERROR [{type(e).__name__}]: {e}") from None
    return 0


def register_pod5index(parser: argparse.ArgumentParser) -> None:
    """Register as a leaf sub-command (``segshape pod5index ...``)."""
    add_pod5index_arguments(parser)
    parser.set_defaults(func=run_pod5index)


def register_dorado_extract(parser: argparse.ArgumentParser) -> None:
    """Register as a leaf sub-command (``segshape dorado-extract ...``)."""
    from segshape.io import dorado_mv
    dorado_mv.add_arguments(parser)
    parser.set_defaults(func=dorado_mv.run)


# Backwards-compatible alias for any caller that imported `register`.
register = register_pod5index
