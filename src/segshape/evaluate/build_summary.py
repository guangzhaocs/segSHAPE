"""Aggregate all per-cell eval_full__*.tsv files into one summary CSV.

Output: code/all_results_summary.csv
Columns:
  dataset, variant, contam, off=*, react_off=*, auc_off=*, mcc_off=*,
  best_mcc, best_mcc_off, best_auc, best_auc_off, best_react, best_react_off, n_eval

Variant names follow several conventions:
  - eval_full__c_<tag>.tsv                       miR17-92 / new_legacy_match (legacy naming)
  - eval_full__<variant>_c<tag>.tsv              miR17-92 default
  - eval_full__<dataset>_<variant>_c<tag>.tsv    other 4 datasets
  - eval_full__skip<N>.tsv                       old skip-penalty sweep (single-eval)
"""
import glob
import os
import re
import sys
import pandas as pd
import numpy as np

CODE_DIR = '/scratch/cs/nanopore/chengg1/segSHAPE/code'
EVAL_DIR = f'{CODE_DIR}/eval_full_out'
OUT_PATH = f'{CODE_DIR}/all_results_summary.csv'

DATASETS = ['miR17-92', 'tetra', 'bsub_16S', 'tpp_bound', 'tpp_unbound']

CONTAM_TAG_TO_FLOAT = {
    'auto': float('nan'),
    'adapt': float('nan'),  # adaptive — actual value depends on n_c
    '0005': 0.0005, '001': 0.001, '002': 0.002, '003': 0.003,
    '005': 0.005, '007': 0.007, '01': 0.01, '05': 0.05,
    '1': 0.1, '15': 0.15, '2': 0.2,
}


def parse_filename(fname):
    """Extract (dataset, variant, contam_tag, contam_value) from eval_full__*.tsv name."""
    m = re.match(r'eval_full__(.+)\.tsv$', fname)
    if not m:
        return None
    body = m.group(1)

    # Special case: c_<tag>  (miR17-92 / new_legacy_match early sweep)
    m = re.match(r'^c_(.+)$', body)
    if m:
        return ('miR17-92', 'new_legacy_match', m.group(1),
                CONTAM_TAG_TO_FLOAT.get(m.group(1), float('nan')))

    # Special case: skip<N>  (old skip-penalty sweep, miR17-92, contam=auto)
    m = re.match(r'^skip(\d+)$', body)
    if m:
        return ('miR17-92', f'skip{m.group(1)}', 'auto', float('nan'))

    # Try <dataset>_<variant>_c<tag>
    for ds in DATASETS:
        if body.startswith(f'{ds}_'):
            rest = body[len(ds) + 1:]
            m = re.match(r'^(.+)_c([^_]+)$', rest)
            if m:
                return (ds, m.group(1), m.group(2),
                        CONTAM_TAG_TO_FLOAT.get(m.group(2), float('nan')))

    # Try <variant>_c<tag>  (miR17-92 default)
    m = re.match(r'^(.+)_c([^_]+)$', body)
    if m:
        return ('miR17-92', m.group(1), m.group(2),
                CONTAM_TAG_TO_FLOAT.get(m.group(2), float('nan')))

    return None


def parse_variant_params(variant):
    """Heuristically parse sp/K-end/bc from variant name."""
    sp = ke = bc = None
    if variant == 'new_legacy_match':
        sp, ke, bc = 100, 1.0, 0.0
    elif variant == 'new_default':
        sp, ke, bc = 10, 0.7, 0.5
    elif variant == 'new_kend_only':
        sp, ke, bc = 10, 0.7, 0.0
    elif variant == 'new_bc_only':
        sp, ke, bc = 10, 1.0, 0.5
    elif variant == 'kend50_default':
        sp, ke, bc = 10, 0.5, 0.5
    elif variant == 'kend70_bc2':
        sp, ke, bc = 10, 0.7, 2.0
    elif variant.startswith('skip'):
        # old code variants
        try:
            sp = int(variant[4:])
            ke, bc = 1.0, 0.0
        except ValueError:
            pass
    else:
        # parse spXX_kend<YY>_bc<ZZ> or sp<X>_kend<Y>_bc<Z>
        m = re.match(r'sp(\d+)_kend(\d+)_bc(\d+)$', variant)
        if m:
            sp = int(m.group(1))
            ke_raw = m.group(2)
            bc_raw = m.group(3)
            ke = int(ke_raw) / 10 if len(ke_raw) <= 2 else int(ke_raw) / 100
            bc = int(bc_raw) / 10 if len(bc_raw) <= 2 else int(bc_raw) / 100
        else:
            m = re.match(r'sp(\d+)_kend(\d+)_bc(\d+)0?$', variant)
            if m:
                sp = int(m.group(1)); ke = int(m.group(2))/10; bc = int(m.group(3))
        # special multi-digit names
        m = re.match(r'sp(\d+)_kend07_bc05$', variant);
        if m: sp, ke, bc = int(m.group(1)), 0.7, 0.5
        m = re.match(r'sp(\d+)_kend07_bc0$', variant)
        if m: sp, ke, bc = int(m.group(1)), 0.7, 0.0
        m = re.match(r'sp(\d+)_kend07_bc20$', variant)
        if m: sp, ke, bc = int(m.group(1)), 0.7, 2.0
        m = re.match(r'sp(\d+)_kend07_bc2$', variant)
        if m: sp, ke, bc = int(m.group(1)), 0.7, 2.0
        m = re.match(r'sp(\d+)_kend07_bc1$', variant)
        if m: sp, ke, bc = int(m.group(1)), 0.7, 1.0
        m = re.match(r'sp(\d+)_kend07_bc15$', variant)
        if m: sp, ke, bc = int(m.group(1)), 0.7, 1.5
        m = re.match(r'sp(\d+)_kend09_bc0$', variant)
        if m: sp, ke, bc = int(m.group(1)), 0.9, 0.0
        m = re.match(r'sp(\d+)_kend10_bc0$', variant)
        if m: sp, ke, bc = int(m.group(1)), 1.0, 0.0
        m = re.match(r'sp(\d+)_kend075_bc20$', variant)
        if m: sp, ke, bc = int(m.group(1)), 0.75, 2.0
        m = re.match(r'sp(\d+)_kend05_bc05$', variant)
        if m: sp, ke, bc = int(m.group(1)), 0.5, 0.5
        m = re.match(r'^K10_bc05_sp(\d+)$', variant)
        if m: sp, ke, bc = int(m.group(1)), 1.0, 0.5
    return sp, ke, bc


def add_arguments(p):
    p.add_argument('--eval-dir', default=EVAL_DIR,
                   help='Directory containing eval_full__*.tsv')
    p.add_argument('--out', default=OUT_PATH,
                   help='Output CSV path')
    return p


def run(args):
    files = sorted(glob.glob(f'{args.eval_dir}/eval_full__*.tsv'))
    print(f'Found {len(files)} TSVs', file=sys.stderr)

    rows = []
    skipped = []
    for f in files:
        name = os.path.basename(f)
        meta = parse_filename(name)
        if meta is None:
            skipped.append(name)
            continue
        dataset, variant, contam_tag, contam_val = meta
        sp, ke, bc = parse_variant_params(variant)
        try:
            df = pd.read_csv(f, sep='\t')
        except Exception as e:
            print(f'  failed to read {name}: {e}', file=sys.stderr)
            continue
        if len(df) == 0:
            skipped.append(name)
            continue
        d = df.iloc[0].to_dict()
        d['_filename'] = name
        d['dataset'] = dataset
        d['variant'] = variant
        d['contam_tag'] = contam_tag
        d['contam_value'] = contam_val
        d['param_sp'] = sp
        d['param_kend'] = ke
        d['param_bc'] = bc
        rows.append(d)

    if not rows:
        print('No rows!', file=sys.stderr)
        sys.exit(1)

    summary = pd.DataFrame(rows)

    # reorder columns: meta first
    meta_cols = ['_filename', 'dataset', 'variant', 'param_sp', 'param_kend',
                 'param_bc', 'contam_tag', 'contam_value', 'method', 'n_eval']
    other_cols = [c for c in summary.columns if c not in meta_cols]
    summary = summary[[c for c in meta_cols if c in summary.columns] + other_cols]

    summary.to_csv(args.out, index=False, float_format='%.4f', na_rep='nan')
    print(f'Wrote {len(summary)} rows × {len(summary.columns)} cols → {args.out}',
          file=sys.stderr)
    print(f'Skipped {len(skipped)} files (couldn\'t parse name)', file=sys.stderr)
    if skipped:
        print('Skipped:', file=sys.stderr)
        for s in skipped[:20]:
            print(f'  {s}', file=sys.stderr)
        if len(skipped) > 20:
            print(f'  ... and {len(skipped)-20} more', file=sys.stderr)

    # Print a quick coverage matrix
    print('\nDataset × variant counts:', file=sys.stderr)
    pivot = summary.pivot_table(index='variant', columns='dataset',
                                 values='_filename', aggfunc='count', fill_value=0)
    print(pivot, file=sys.stderr)
    return 0


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='Aggregate eval_full__*.tsv into summary CSV.')
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main() or 0)
