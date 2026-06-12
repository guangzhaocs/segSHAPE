"""CLI surface for evaluation / aggregation utilities.

Sub-sub-commands::

    segshape evaluate pipeline      ...   # full structure-recovery eval
    segshape evaluate fold-out      ...   # score a pre-computed RNAfold .out
    segshape evaluate fold-out-all  ...   # batch fold-out, aggregate to CSV
    segshape evaluate filter-ll     ...   # log-likelihood whitelist generation
    segshape evaluate summarize     ...   # ablation table aggregator
    segshape evaluate build-summary ...   # eval_full__*.tsv → CSV
"""

from __future__ import annotations

import argparse


def register(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="task", required=True, metavar="<task>")

    from segshape.evaluate import (pipeline, fold_out, fold_out_all,
                                   filter_ll, summarize, build_summary)

    sp_p = sub.add_parser("pipeline", help="end-to-end MCC / reactivity evaluation")
    pipeline.add_arguments(sp_p)
    sp_p.set_defaults(func=pipeline.run)

    sp_fo = sub.add_parser("fold-out",
                            help="score a pre-computed RNAfold .out vs struct GT")
    fold_out.add_arguments(sp_fo)
    sp_fo.set_defaults(func=fold_out.run)

    sp_foa = sub.add_parser("fold-out-all",
                             help="batch fold-out: walk a dir, aggregate CSV")
    fold_out_all.add_arguments(sp_foa)
    sp_foa.set_defaults(func=fold_out_all.run)

    sp_f = sub.add_parser("filter-ll", help="LL-based read whitelist generation")
    filter_ll.add_arguments(sp_f)
    sp_f.set_defaults(func=filter_ll.run)

    sp_s = sub.add_parser("summarize", help="aggregate Stage-2.1 ablation tables")
    summarize.add_arguments(sp_s)
    sp_s.set_defaults(func=summarize.run)

    sp_b = sub.add_parser("build-summary", help="eval_full__*.tsv → summary CSV")
    build_summary.add_arguments(sp_b)
    sp_b.set_defaults(func=build_summary.run)
