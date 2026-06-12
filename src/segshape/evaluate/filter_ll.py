"""Stage 2.1 — per-read alignment-quality filter via DP log-likelihood.

Reads scale.csv from a (control, treated) pair of alignment dirs, computes
mean_ll = ll / n_events, derives a control-only threshold via
    thr(K) = median(control mean_ll) - K * MAD(control mean_ll)
and writes per-K read_id whitelists for both control and treated. The
absolute threshold is shared between control and treated (avoids treating
modified reads as outliers, which would bias the IF training set).

Outputs:
  {out_prefix}_K{K}_ctrl.txt  : one read_id per line (control reads passing thr)
  {out_prefix}_K{K}_trt.txt   : one read_id per line (treated reads passing thr)
  {out_prefix}_stats.tsv      : per-K summary (n_pass, kept_frac, symmetry)

Notes:
  - Only reads with qc_tag == 'PASS' are considered (matches downstream usage).
  - The DP `ll` is raw V[N,L] (includes skip penalty); not comparable across
    different alignment configs (different sp), but valid within one config.
"""
import argparse
import os

import numpy as np
import pandas as pd


def collect_scale(align_dir: str) -> pd.DataFrame:
    parts = sorted(d for d in os.listdir(align_dir) if d.startswith('partition'))
    if not parts:
        raise RuntimeError(f'no partitions in {align_dir}')
    dfs = []
    for p in parts:
        path = f'{align_dir}/{p}/scale.csv'
        df = pd.read_csv(path)
        df['partition'] = p
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def add_arguments(ap):
    ap.add_argument('--ctrl-align-dir', required=True)
    ap.add_argument('--trt-align-dir', required=True)
    ap.add_argument('--out-prefix', required=True,
                    help='output path prefix; produces {prefix}_K{K}_{ctrl,trt}.txt and {prefix}_stats.tsv')
    ap.add_argument('--K-values', default='0,0.5,1,1.5,2',
                    help='comma-separated K values for thr = med - K*MAD')
    ap.add_argument('--qc-pass-only', action='store_true',
                    help='restrict to qc_tag == "PASS" reads.')
    return ap


def run(args):
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f'Loading control scale.csv from {args.ctrl_align_dir}')
    ctrl = collect_scale(args.ctrl_align_dir)
    print(f'Loading treated scale.csv from {args.trt_align_dir}')
    trt = collect_scale(args.trt_align_dir)

    if args.qc_pass_only:
        ctrl_full, trt_full = len(ctrl), len(trt)
        ctrl = ctrl[ctrl['qc_tag'] == 'PASS'].copy()
        trt = trt[trt['qc_tag'] == 'PASS'].copy()
        print(f'  control: {len(ctrl)}/{ctrl_full} PASS')
        print(f'  treated: {len(trt)}/{trt_full} PASS')
    else:
        print(f'  control: total={len(ctrl)} (qc_tag-agnostic; '
              f'{(ctrl["qc_tag"]=="PASS").sum()} PASS)')
        print(f'  treated: total={len(trt)} (qc_tag-agnostic; '
              f'{(trt["qc_tag"]=="PASS").sum()} PASS)')

    ctrl = ctrl[ctrl['n_events'] > 0].copy()
    trt = trt[trt['n_events'] > 0].copy()

    ctrl['mean_ll'] = ctrl['ll'] / ctrl['n_events']
    trt['mean_ll'] = trt['ll'] / trt['n_events']

    med = float(ctrl['mean_ll'].median())
    mad = float((ctrl['mean_ll'] - med).abs().median())
    print(f'\ncontrol mean_ll: median={med:.4f}  MAD={mad:.4f}'
          f'  q[5,25,75,95]='
          f'{np.percentile(ctrl["mean_ll"], [5,25,75,95]).round(4).tolist()}')
    print(f'treated mean_ll: median={trt["mean_ll"].median():.4f}'
          f'  q[5,25,75,95]='
          f'{np.percentile(trt["mean_ll"], [5,25,75,95]).round(4).tolist()}')

    Ks = [float(k) for k in args.K_values.split(',')]
    rows = []
    for K in Ks:
        thr = med - K * mad
        ctrl_pass = ctrl[ctrl['mean_ll'] >= thr]
        trt_pass = trt[trt['mean_ll'] >= thr]
        K_tag = (f'{K:g}').replace('.', 'p')  # 0.5 -> '0p5'
        ctrl_wl = f'{args.out_prefix}_K{K_tag}_ctrl.txt'
        trt_wl = f'{args.out_prefix}_K{K_tag}_trt.txt'
        ctrl_pass['read_id'].to_csv(ctrl_wl, index=False, header=False)
        trt_pass['read_id'].to_csv(trt_wl, index=False, header=False)
        ctrl_kept = len(ctrl_pass) / max(1, len(ctrl))
        trt_kept = len(trt_pass) / max(1, len(trt))
        sym = abs(ctrl_kept - trt_kept)
        rows.append({
            'K': K, 'threshold': thr,
            'ctrl_n_total': len(ctrl), 'ctrl_n_pass': len(ctrl_pass),
            'ctrl_kept_frac': ctrl_kept,
            'trt_n_total': len(trt), 'trt_n_pass': len(trt_pass),
            'trt_kept_frac': trt_kept,
            'symmetry_diff': sym,
            'ctrl_whitelist': ctrl_wl, 'trt_whitelist': trt_wl,
        })
        print(f'  K={K}: thr={thr:.4f}  '
              f'ctrl: {len(ctrl_pass)}/{len(ctrl)} ({ctrl_kept:.1%})  '
              f'trt:  {len(trt_pass)}/{len(trt)} ({trt_kept:.1%})  '
              f'|Δkept|={sym:.3f}')

    stats_path = f'{args.out_prefix}_stats.tsv'
    pd.DataFrame(rows).to_csv(stats_path, sep='\t', index=False,
                              float_format='%.4f')
    print(f'\nWrote {stats_path}')


def main(argv=None):
    import sys
    p = argparse.ArgumentParser()
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
