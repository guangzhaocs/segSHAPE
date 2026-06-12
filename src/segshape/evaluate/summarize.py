"""Aggregate Stage 2.1 v2 ablation eval TSVs (5 datasets) into a summary.

Reads:
  code/eval_full_out/eval_full__{ds}_s2_sp50_bc0_me0_mf3_k50.tsv         (baseline; original / no-filter)
  code/eval_full_out/eval_full__tpp_unbound_relaxed_s2_sp50_bc0_me0_mf3_k50.tsv  (tpp_unbound baseline uses relaxed)
  code/eval_full_out/eval_filter__{ds}_s2p1v2_K{K}_{mode}.tsv            (40 ablation cells)
  code/stage21_out/{ds}_winner_stats.tsv                                  (filter stats)
"""
import os
import pandas as pd

ROOT = '/scratch/cs/nanopore/chengg1/segSHAPE'
EVAL = f'{ROOT}/code/eval_full_out'
STAGE21 = f'{ROOT}/code/stage21_out'

DATASETS = ['miR17-92', 'tetra', 'bsub_16S', 'tpp_bound', 'tpp_unbound']
KS = ['inf', '2', '1p5', '1', '0p5']
MODES = ['joint', 'ctrl']

BASELINE_PATH = {
    'miR17-92':    f'{EVAL}/eval_full__miR17-92_s2_sp50_bc0_me0_mf3_k50.tsv',
    'tetra':       f'{EVAL}/eval_full__tetra_s2_sp50_bc0_me0_mf3_k50.tsv',
    'bsub_16S':    f'{EVAL}/eval_full__bsub_16S_s2_sp50_bc0_me0_mf3_k50.tsv',
    'tpp_bound':   f'{EVAL}/eval_full__tpp_bound_s2_sp50_bc0_me0_mf3_k50.tsv',
    # tpp_unbound: stage 2 winner uses relaxed segmentation
    'tpp_unbound': f'{EVAL}/eval_full__tpp_unbound_relaxed_s2_sp50_bc0_me0_mf3_k50.tsv',
}
TARGETS = {
    'miR17-92': 85, 'tetra': 72, 'bsub_16S': 60,
    'tpp_bound': 70, 'tpp_unbound': 70,
}


def load_tsv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep='\t')
    if df.empty:
        return None
    return df.iloc[0]


def fmt(v, prec=2):
    if v is None or pd.isna(v):
        return '—'
    return f'{v:.{prec}f}'


def get_filter_stats(ds):
    path = f'{STAGE21}/{ds}_winner_stats.tsv'
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep='\t')
    out = {}
    for _, r in df.iterrows():
        K_tag = (f'{r["K"]:g}').replace('.', 'p')
        out[K_tag] = {
            'thr': r['threshold'],
            'ctrl_kept': r['ctrl_kept_frac'],
            'trt_kept': r['trt_kept_frac'],
        }
    return out


def add_arguments(p):
    p.add_argument('--eval-dir', default=EVAL,
                   help='Directory with eval_full__*.tsv and eval_filter__*.tsv')
    p.add_argument('--stage21-dir', default=STAGE21,
                   help='Directory with {ds}_winner_stats.tsv')
    return p


def _run_summary():
    # Per-dataset detail
    for ds in DATASETS:
        target = TARGETS[ds]
        print(f'\n## {ds}  (target = {target}%)\n')
        stats = get_filter_stats(ds)
        rows = []
        baseline = load_tsv(BASELINE_PATH[ds])
        if baseline is not None:
            rows.append({
                'K': '∞', 'mode': 'baseline',
                'ctrl_kept': 1.0, 'trt_kept': 1.0,
                'n_eval': baseline['n_eval'],
                'mcc_off2': baseline['mcc_off2'],
                'best_mcc': baseline['best_mcc'],
                'best_mcc_off': baseline['best_mcc_off'],
                'react_off2': baseline.get('react_off2', None),
            })
        for K in KS:
            if K == 'inf':
                continue
            for mode in MODES:
                tsv = f'{EVAL}/eval_filter__{ds}_s2p1v2_K{K}_{mode}.tsv'
                row = load_tsv(tsv)
                if row is None:
                    continue
                s = stats.get(K, {})
                rows.append({
                    'K': K, 'mode': mode,
                    'ctrl_kept': s.get('ctrl_kept'),
                    'trt_kept': s.get('trt_kept') if mode == 'joint' else 1.0,
                    'n_eval': row['n_eval'],
                    'mcc_off2': row['mcc_off2'],
                    'best_mcc': row['best_mcc'],
                    'best_mcc_off': row['best_mcc_off'],
                    'react_off2': row.get('react_off2', None),
                })
        print('| K | mode | ctrl_kept | trt_kept | n_eval | mcc@off=2 | best_mcc@off | react@off=2 |')
        print('|---|------|----------:|---------:|-------:|----------:|--------------|------------:|')
        for r in rows:
            ck = f'{r["ctrl_kept"]:.0%}' if r['ctrl_kept'] is not None else '—'
            tk = f'{r["trt_kept"]:.0%}' if r['trt_kept'] is not None else '—'
            mark_off2 = '✓' if (r['mcc_off2'] is not None and not pd.isna(r['mcc_off2']) and r['mcc_off2'] >= target) else ''
            mark_best = '✓' if (r['best_mcc'] is not None and not pd.isna(r['best_mcc']) and r['best_mcc'] >= target) else ''
            print(f'| {r["K"]} | {r["mode"]} | {ck} | {tk} | {int(r["n_eval"])} | '
                  f'{fmt(r["mcc_off2"])} {mark_off2} | {fmt(r["best_mcc"])}@{int(r["best_mcc_off"])} {mark_best} | '
                  f'{fmt(r["react_off2"], 3)} |')

    # Universal config search (best K/mode that maximizes 5/5 pass rate at off=2)
    print('\n\n## Universal config search — mcc@off=2 fair frame, target pass rate\n')
    print('| K | mode | miR | tetra | bsub | tpp_b | tpp_u | pass/5 |')
    print('|---|------|----:|------:|-----:|------:|------:|-------:|')
    for K in ['2', '1p5', '1', '0p5']:
        for mode in MODES:
            cells = {}
            pass_count = 0
            for ds in DATASETS:
                row = load_tsv(f'{EVAL}/eval_filter__{ds}_s2p1v2_K{K}_{mode}.tsv')
                if row is not None and not pd.isna(row['mcc_off2']):
                    cells[ds] = row['mcc_off2']
                    if row['mcc_off2'] >= TARGETS[ds]:
                        pass_count += 1
                else:
                    cells[ds] = None
            print(f'| {K} | {mode} | '
                  f'{fmt(cells.get("miR17-92"))} | {fmt(cells.get("tetra"))} | '
                  f'{fmt(cells.get("bsub_16S"))} | {fmt(cells.get("tpp_bound"))} | '
                  f'{fmt(cells.get("tpp_unbound"))} | {pass_count}/5 |')

    # Same but for best_mcc
    print('\n## Universal config search — best_mcc, target pass rate\n')
    print('| K | mode | miR | tetra | bsub | tpp_b | tpp_u | pass/5 |')
    print('|---|------|----:|------:|-----:|------:|------:|-------:|')
    for K in ['2', '1p5', '1', '0p5']:
        for mode in MODES:
            cells = {}
            pass_count = 0
            for ds in DATASETS:
                row = load_tsv(f'{EVAL}/eval_filter__{ds}_s2p1v2_K{K}_{mode}.tsv')
                if row is not None and not pd.isna(row['best_mcc']):
                    cells[ds] = row['best_mcc']
                    if row['best_mcc'] >= TARGETS[ds]:
                        pass_count += 1
                else:
                    cells[ds] = None
            print(f'| {K} | {mode} | '
                  f'{fmt(cells.get("miR17-92"))} | {fmt(cells.get("tetra"))} | '
                  f'{fmt(cells.get("bsub_16S"))} | {fmt(cells.get("tpp_bound"))} | '
                  f'{fmt(cells.get("tpp_unbound"))} | {pass_count}/5 |')


def run(args):
    global EVAL, STAGE21, BASELINE_PATH
    EVAL = args.eval_dir
    STAGE21 = args.stage21_dir
    BASELINE_PATH = {
        ds: f'{EVAL}/eval_full__{ds}_s2_sp50_bc0_me0_mf3_k50.tsv' for ds in DATASETS
    }
    BASELINE_PATH['tpp_unbound'] = f'{EVAL}/eval_full__tpp_unbound_relaxed_s2_sp50_bc0_me0_mf3_k50.tsv'
    _run_summary()
    return 0


def main(argv=None):
    import argparse, sys
    p = argparse.ArgumentParser(description='Aggregate Stage 2.1 v2 ablation eval TSVs.')
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
