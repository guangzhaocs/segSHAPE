"""Show C++ vs Python alignment trajectory for a few specific reads.

For each read, sample 20 evenly-spaced chronological points; for each, show:
  signal_start | C++ kmer_pos | Python kmer_pos (after coord flip 946 - py_pos)

C++ events come from HHMM with signal_start relative to trans_st (signal[trans_st:]),
Python events from segSHAPE res_border with signal_start in original signal coords.
We map by `py_signal_start - trans_st == cpp_signal_start` (verified earlier).
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

L_KMER_DEFAULT = 947


def show(rid, cpp_df, py_align, py_scale, info, bd_lines, L_kmer, n_samples=20):
    sub_c = cpp_df[cpp_df.rid == rid].sort_values('start').reset_index(drop=True)
    if len(sub_c) < n_samples:
        return False
    ridx = py_scale[py_scale.read_id == rid].read_idx.iloc[0]
    sub_p = py_align[py_align.read_idx == ridx].sort_values('event_idx').reset_index(drop=True)
    bd = np.fromstring(bd_lines[ridx].strip(), sep=',', dtype=np.int64)[1:]
    trans_st = int(info[info.read_id == rid].trans_st.iloc[0])
    py_starts = np.array([bd[2 * e] for e in sub_p.event_idx.values])
    cpp_starts_orig = sub_c.start.values + trans_st

    print(f'\n========== read {rid}  (trans_st={trans_st}) ==========')
    print(f'  C++ events: {len(sub_c)}, kmer_pos range '
          f'[{sub_c.kmer_pos.min()}..{sub_c.kmer_pos.max()}], '
          f'unique={sub_c.kmer_pos.nunique()}')
    print(f'  Py  events: {len(sub_p)}, phys range '
          f'[{(L_kmer - 1) - sub_p.pos_idx.max()}..{(L_kmer - 1) - sub_p.pos_idx.min()}], '
          f'unique={sub_p.pos_idx.nunique()}')

    print(f'\n  {"chrono%":>7} {"sig_orig":>10} {"cpp_pos":>8} {"py_pos":>8} {"Δ":>5}  {"cpp_kmer":>8}')
    print(f'  {"-"*7} {"-"*10} {"-"*8} {"-"*8} {"-"*5}  {"-"*8}')
    idx_samples = np.linspace(0, len(sub_c) - 1, n_samples).astype(int)
    diffs = []
    for i in idx_samples:
        sc = cpp_starts_orig[i]
        cp = int(sub_c.kmer_pos.iloc[i])
        ks = sub_c.kmer_str.iloc[i]
        nn = np.argmin(np.abs(py_starts - sc))
        pp = (L_kmer - 1) - int(sub_p.pos_idx.iloc[nn])
        d = cp - pp if cp >= 0 else None
        if d is not None:
            diffs.append(d)
        d_str = f'{d:5d}' if d is not None else '  N/A'
        print(f'  {100 * i / (len(sub_c) - 1):6.0f}% {sc:>10d} {cp:>8d} {pp:>8d} {d_str}  {ks[-5:]:>8}')
    if diffs:
        diffs = np.array(diffs)
        print(f'\n  Δ stats over {len(diffs)} non-padding samples:  '
              f'mean={diffs.mean():.1f}  median={np.median(diffs):.0f}  '
              f'std={diffs.std():.1f}')
    return True


def add_arguments(p):
    p.add_argument('--seg-dir', required=True, type=Path)
    p.add_argument('--py-align-dir', required=True, type=Path)
    p.add_argument('--cpp-align-dir', required=True, type=Path)
    p.add_argument('--n-reads', type=int, default=3,
                   help='number of well-spaced reads to display')
    p.add_argument('--L-kmer', type=int, default=L_KMER_DEFAULT)
    return p


def run(args):
    print('Loading data...')
    cpp_df = pd.read_csv(args.cpp_align_dir / 'eventalign_v1.txt', sep='\t', header=None,
                         usecols=range(9),
                         names=['rid', 'contig', 'kmer_str', 'kmer_pos', 'mu', 'sigma',
                                'start', 'end', 'len'])
    py_align = pd.read_csv(args.py_align_dir / 'alignment.csv')
    py_scale = pd.read_csv(args.py_align_dir / 'scale.csv')
    info = pd.read_csv(args.seg_dir / 'info.csv', header=None,
                       names=['read_id', 'contig', 'polya_st', 'trans_st', 'qc_tag', 'signal_len'])
    bd_lines = open(args.seg_dir / 'res_border.csv').readlines()

    real = cpp_df[cpp_df.kmer_pos >= 0]
    cpp_nu = real.groupby('rid')['kmer_pos'].nunique()
    py_nu = py_align.groupby('read_idx')['pos_idx'].nunique()
    ridx_to_rid = py_scale.set_index('read_idx')['read_id']
    py_nu_by_rid = py_nu.rename(index=ridx_to_rid)
    both = cpp_nu.index.intersection(py_nu_by_rid.index)
    joint = pd.DataFrame({'cpp': cpp_nu.loc[both], 'py': py_nu_by_rid.loc[both]})
    joint['min'] = joint[['cpp', 'py']].min(axis=1)
    candidates = joint[(joint['cpp'] > 800) & (joint['py'] > 800)].index
    print(f'Candidates (both nunique>800): {len(candidates)}')

    n = max(1, args.n_reads)
    picks = list(candidates[::max(1, len(candidates) // n)])[:n]
    print(f'Picked: {picks}')

    for rid in picks:
        show(rid, cpp_df, py_align, py_scale, info, bd_lines, args.L_kmer)
    return 0


def main(argv=None):
    import sys
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
