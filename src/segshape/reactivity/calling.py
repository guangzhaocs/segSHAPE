"""Per-position modification rate calling from step-5 outputs.

Inputs (per side, control + treated):
    subevents.parquet          : read_id, event_idx, start_sample, end_sample,
                                  mean_pa, std_pa
    <sweep_cell>/alignment.csv : read_idx, event_idx, pos_idx, ref_center_base_pos
    <sweep_cell>/scale.csv     : read_idx, read_id, qc_tag, scale, shift, ll, ...

Pipeline:
    1. join alignment ↔ scale on read_idx → per-row read_id + scale + shift
    2. join with subevents.parquet on (read_id, event_idx) → per-row mean_pa + sample-weight
    3. apply per-read affine: cal_mean = scale * mean_pa + shift
    4. weighted-mean over events for each (read, pos)
    5. bucket per pos_idx → control / treated event distributions
    6. compute_metric: per-pos test (IF / KS / wasserstein / dmed / OCSVM / GMM / xpore)
    7. write a per-run output folder containing:
         mod_rate.csv                          — pos_idx, mod_rate (0-based);
                                                 RAW per-position rate only
                                                 (z-score / smooth / norm
                                                 live in the .dat below)
         reactivity_smooth<W>_norm-<NORM>.dat  — 1-based pos, NaN -> -999;
                                                 RNAfold / RNAstructure
                                                 SHAPE-constraint format.
                                                 Multiple smooth/norm
                                                 sweeps coexist as separate
                                                 .dat files in the same
                                                 folder.

Output folder lives under the *treated* sweep-cell directory:
    <trt_align_dir>/mod_rate/<variant>_<method><param_tag>/
e.g.
    <trt_align_dir>/mod_rate/default_if-1D_c0.0050/
    <trt_align_dir>/mod_rate/shift_only_xpore/

`--method` accepts a single method, a comma-separated list
(`ks,wass,if-2D`), or one of the keywords `all` / `1d` / `2d`. When
multiple methods are passed, each writes to its own folder (no
conflict) and per-pos buckets are loaded **once per feature_mode**
(at most twice: 'mean' and 'mean_std') and reused across methods.

Folder naming rule:
- `<variant>_<method>[<param_tag>]` — encodes only the inputs that
  change `mod_rate` (variant, method, the method's headline knob).
- smooth/norm tags go on the .dat filename, NOT the folder, since
  they are pure post-processing on top of `mod_rate`.

MCC / AUC / Spearman / RNAfold orchestration live in `evaluate.pipeline`.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def _read_fasta_seq(fa_path: str, contig: Optional[str] = None) -> str:
    """Read a single contig from a fasta file. If `contig` is None,
    return the first record. Strips whitespace; preserves case as-is.

    (Inlined here to avoid a backward import from segshape.evaluate;
    same logic as evaluate.pipeline._read_fasta_seq.)"""
    seq, cur_name, capture = [], None, False
    with open(fa_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                name = line[1:].split()[0] if len(line) > 1 else ""
                if contig is None:
                    if cur_name is not None:
                        break
                    cur_name, capture = name, True
                else:
                    capture = (name == contig)
                    if capture:
                        cur_name = name
            elif capture:
                seq.append(line.strip())
    if not seq:
        raise SystemExit(
            f"contig {contig or '(first)'!r} not found in {fa_path}")
    return "".join(seq)


def _write_reactivity_dat(rates_z: np.ndarray, ref_len: int, offset: int,
                            out_path: str) -> int:
    """Write a 1-based, RNAfold/RNAstructure SHAPE-constraint .dat file.

    Maps alignment-internal pos_idx i → fasta pos
    (ref_len - i - offset, 1-indexed) — same convention as
    evaluate.pipeline.write_reactivity_dat. Missing positions written
    as ``-999``. Returns count of valid positions written.

    Why this mapping (and not just pos_idx + 1):
      Our anchored alignment runs Viterbi traceback in reverse signal
      order, and ``pos_idx`` indexes the kmer-center event, not the
      reference base. So:
        - pos_idx runs in REVERSE relative to the reference
        - There is a kmer-center offset (default 2 for both RNA002
          5-mer and RNA004 9-mer; see docs/attention.md §2)

      The combination is captured by ``fa = ref_len - pos_idx - offset``.
      Without this transform the .dat is misaligned with the reference
      and RNAfold will apply SHAPE constraints to wrong positions."""
    L = len(rates_z)
    fa_react = np.full(ref_len, -999.0)
    n_valid = 0
    for i in range(L):
        if not np.isfinite(rates_z[i]):
            continue
        fa = ref_len - i - offset                            # 1-indexed
        if 1 <= fa <= ref_len:
            fa_react[fa - 1] = rates_z[i]
            n_valid += 1
    with open(out_path, "w") as f:
        for i, v in enumerate(fa_react):
            if v == -999.0:
                f.write(f"{i+1}\t-999\n")
            else:
                f.write(f"{i+1}\t{v:.6f}\n")
    return n_valid


def _load_whitelist(path: Optional[str]) -> Optional[set]:
    if not path:
        return None
    with open(path) as f:
        ids = {ln.strip() for ln in f if ln.strip()}
    print(f"  loaded {len(ids)} read_ids from {path}", flush=True)
    return ids


def _plan_a_scale_std(events: pd.DataFrame) -> pd.DataFrame:
    """Plan A per-read std calibration for mod-calling 2-D features.

    Adds a ``cal_std = std_pa / read_median(std_pa)`` column. Reads with
    median std_pa <= 1e-3 (degenerate) get a 1e-3 floor to avoid divide-
    by-zero blowing up cal_std.

    Why per-read median (not v_scale): see docs/attention.md §5
    "Don't do: use v_scale to normalise std_pa for mod-calling" — v_scale
    itself absorbs modification signal (Δv ≈ +0.4-0.6 in treated
    samples), so dividing by it would partially erase what we want to
    detect. Per-read median std_pa is robust to ~30-50% modified
    positions (median dominated by unmodified events) and decouples
    pore-noise variability from modification signal.

    Caller must drop NaN std_pa rows first if subevents.parquet contains
    NaN-std micro-events (events.py marks std=0 / mean<0 events as NaN
    while preserving event_idx tiling)."""
    read_med = (events.groupby("read_id")["std_pa"]
                       .transform("median")
                       .clip(lower=1e-3))
    events = events.copy()
    events["cal_std"] = (events["std_pa"].astype(np.float64)
                         / read_med.astype(np.float64))
    return events


def collect_per_pos_events(
    events_path: str,
    alignment_path: str,
    scale_path: str,
    apply_shift: bool = True,
    coverage_filter: float = 0.0,
    read_id_whitelist: Optional[set] = None,
    qc_pass_only: bool = True,
    feature_mode: str = "mean",
) -> Dict[int, np.ndarray]:
    """Per-position event aggregation from new step-5 outputs.

    feature_mode:
      'mean'     → returns dict[pos] -> 1-D ndarray (n,) of per-read
                   weighted-mean pA. Matches legacy 1-D mod-calling input.
      'mean_std' → returns dict[pos] -> 2-D ndarray (n, 2) of per-read
                   [weighted-mean cal_mean, weighted-mean cal_std], where
                   cal_std uses Plan A per-read normalization
                   (std_pa / read_median(std_pa)). Drops NaN-std events.

    Weighting: per-event weight = (end_sample - start_sample), so
    weighted-mean over the read's events for that pos preserves
    signal-sample weighting (matches legacy `collect_per_read_pos`).
    """
    if feature_mode not in ("mean", "mean_std"):
        raise ValueError(
            f"feature_mode must be 'mean' or 'mean_std'; got {feature_mode!r}")

    align = pd.read_csv(alignment_path,
                        usecols=["read_idx", "event_idx", "pos_idx"])
    scale = pd.read_csv(scale_path)

    if qc_pass_only and "qc_tag" in scale.columns:
        scale = scale[scale["qc_tag"] == "PASS"]

    if read_id_whitelist is not None:
        scale = scale[scale["read_id"].isin(read_id_whitelist)]

    keep_cols = ["read_idx", "read_id", "scale", "shift"]
    align = align.merge(scale[keep_cols], on="read_idx", how="inner")

    if coverage_filter > 0:
        L_obs = int(align["pos_idx"].max()) + 1
        cov = align.groupby("read_idx")["pos_idx"].nunique() / L_obs
        kept = cov[cov >= coverage_filter].index
        align = align[align["read_idx"].isin(kept)]

    cols = ["read_id", "event_idx", "start_sample", "end_sample", "mean_pa"]
    if feature_mode == "mean_std":
        cols.append("std_pa")
    events = pd.read_parquet(events_path, columns=cols)
    events["weight"] = (events["end_sample"] - events["start_sample"]
                        ).astype(np.float64).clip(lower=1.0)

    if feature_mode == "mean_std":
        events = events.dropna(subset=["std_pa"])
        events = _plan_a_scale_std(events)

    merged = align.merge(events, on=["read_id", "event_idx"], how="inner")

    if apply_shift:
        cal = merged["scale"].astype(np.float64) * merged["mean_pa"].astype(
            np.float64) + merged["shift"].astype(np.float64)
    else:
        cal = merged["mean_pa"].astype(np.float64)

    merged["w_mean"] = cal * merged["weight"]

    if feature_mode == "mean_std":
        merged["w_std"] = (merged["cal_std"].astype(np.float64)
                           * merged["weight"])
        agg = (merged.groupby(["read_idx", "pos_idx"])
                      [["w_mean", "w_std", "weight"]].sum())
        agg["cal_mean"] = agg["w_mean"] / agg["weight"]
        agg["cal_std_mean"] = agg["w_std"] / agg["weight"]
        agg = agg.reset_index()
        pos_events: Dict[int, np.ndarray] = {}
        for pos, group in agg.groupby("pos_idx"):
            pos_events[int(pos)] = (group[["cal_mean", "cal_std_mean"]]
                                    .to_numpy())
        return pos_events

    agg = (merged.groupby(["read_idx", "pos_idx"])
                  [["w_mean", "weight"]].sum())
    agg["cal_mean"] = agg["w_mean"] / agg["weight"]
    agg = agg.reset_index()
    pos_events = {}
    for pos, vals in agg.groupby("pos_idx")["cal_mean"]:
        pos_events[int(pos)] = vals.to_numpy()
    return pos_events


def _xpore_em(
    c: np.ndarray,
    t: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-5,
    reg_var: float = 1e-3,
) -> float:
    """xpore-style joint 2-component GMM, EM point estimate.

    Per-position score = max(0, π_t[mod] - π_c[mod]), where the two
    Gaussian components share their means and variances across the two
    conditions, but each condition gets its own mixing weights. The
    "modified" component is identified post-hoc as the one with the
    smaller mixing weight in control (assumes control is mostly
    unmodified — same assumption as the original xpore).

    Reference: Pratanwanich et al. 2021, Nat Biotech (xPore paper,
    https://www.nature.com/articles/s41587-021-00949-w). Differences
    from the published method:
      * Maximum-likelihood EM, not full Bayesian posterior over (μ, σ², π).
      * Uniform priors (no spike-in informative priors on the modified-
        base mean shift; we don't have spike-ins).
      * No posterior credible interval — score is a point estimate.
      * 1-D feature only (cal_mean), as with our other methods.

    Initialization: component means at the 25 % / 75 % quantiles of
    pooled (control + treated) data; shared variance from pooled
    variance; mixing weights uniform [0.5, 0.5] in both conditions.

    Returns NaN if pooled |c| + |t| < 4. Returns 0.0 if components
    collapse (|μ_0 - μ_1| < 0.1 σ̄), which means the data is effectively
    single-component and there is no modification signal to extract.
    """
    n_c, n_t = len(c), len(t)
    pooled = np.concatenate([c, t]).astype(np.float64)
    if len(pooled) < 4:
        return float("nan")
    mu = np.array([np.quantile(pooled, 0.25),
                   np.quantile(pooled, 0.75)], dtype=np.float64)
    if mu[1] - mu[0] < 1e-6:
        return 0.0
    var = np.full(2, max(float(pooled.var()), reg_var), dtype=np.float64)
    pi_c = np.array([0.5, 0.5], dtype=np.float64)
    pi_t = np.array([0.5, 0.5], dtype=np.float64)
    c64 = c.astype(np.float64)
    t64 = t.astype(np.float64)

    def _log_resp(x: np.ndarray, pi: np.ndarray):
        log_norm = -0.5 * np.log(2.0 * np.pi * var)            # (2,)
        log_p = log_norm[None, :] - 0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :]
        log_w = np.log(pi)[None, :] + log_p                    # (n, 2)
        m = np.max(log_w, axis=1, keepdims=True)
        log_z = m + np.log(np.exp(log_w - m).sum(axis=1, keepdims=True))
        return log_w - log_z, float(log_z.sum())

    prev_ll = -np.inf
    for _ in range(max_iter):
        log_g_c, ll_c = _log_resp(c64, pi_c)
        log_g_t, ll_t = _log_resp(t64, pi_t)
        ll = ll_c + ll_t
        if abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            break
        prev_ll = ll
        g_c = np.exp(log_g_c)
        g_t = np.exp(log_g_t)
        sum_g = g_c.sum(0) + g_t.sum(0)
        sum_g = np.clip(sum_g, 1e-9, None)
        mu = ((g_c * c64[:, None]).sum(0) + (g_t * t64[:, None]).sum(0)) / sum_g
        var = np.maximum(
            ((g_c * (c64[:, None] - mu[None, :]) ** 2).sum(0)
             + (g_t * (t64[:, None] - mu[None, :]) ** 2).sum(0)) / sum_g,
            reg_var,
        )
        pi_c = g_c.sum(0) / n_c
        pi_t = g_t.sum(0) / n_t

    if abs(mu[1] - mu[0]) < 0.1 * float(np.sqrt(var.mean())):
        return 0.0
    mod_idx = int(np.argmin(pi_c))
    return float(max(0.0, pi_t[mod_idx] - pi_c[mod_idx]))


METHODS = ("if-1D", "if-2D",
           "ks", "wass", "dmed",
           "ocsvm-1D", "ocsvm-2D",
           "gmm-1D", "gmm-2D",
           "xpore")
METHODS_2D = ("if-2D", "ocsvm-2D", "gmm-2D")
METHODS_IF = ("if-1D", "if-2D")
METHODS_OCSVM = ("ocsvm-1D", "ocsvm-2D")
METHODS_GMM = ("gmm-1D", "gmm-2D")


def _as_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure (n,) → (n, 1); pass (n, d) through unchanged."""
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def compute_metric(
    ctrl: Dict[int, np.ndarray],
    trt: Dict[int, np.ndarray],
    L: int,
    method: str,
    *,
    contamination=None,
    nu: float = 0.005,
    gmm_quantile: float = 0.005,
    gmm_n_comp: str = "auto",
    if_n_estimators: int = 100,
    if_max_samples="auto",
    min_n_c: int = 256,
    min_n_t: int = 100,
    seed: int = 42,
    subsample_cap: int = 5000,
    ks_method: str = "asymp",
) -> np.ndarray:
    """Per-position modification-rate test.

    Method-specific kwargs (each used by exactly one method, ignored otherwise):
      contamination   : if    — IF training-set outlier upper bound; float, 'auto', or 'adaptive'
      nu              : ocsvm — One-Class SVM nu (float in (0, 0.5])
      gmm_quantile    : gmm   — control log-prob quantile threshold (float in (0, 0.5])
      gmm_n_comp      : gmm   — 'auto' (BIC-pick from {1, 2}; default), '1', '2'
      if_n_estimators : if    — number of isolation trees (sklearn default 100)
      if_max_samples  : if    — per-tree sub-sample size: 'auto' (= min(256, n_c),
                                sklearn default), int, or float in (0.0, 1.0]
      subsample_cap   : ks / wass — max events per side before random sub-sample
                                  (default 5000; caps make per-position distances
                                  comparable across coverage)
      ks_method       : ks    — scipy ``ks_2samp(method=...)`` solver
                              ('auto' / 'exact' / 'asymp'; default 'asymp')

    Shared kwargs:
      seed          : rng seed used by ``if`` (IsolationForest random_state),
                      ``gmm`` (GaussianMixture random_state) and the
                      ``ks`` / ``wass`` sub-sample rng. ``dmed`` / ``ocsvm``
                      don't use it.
    """
    rates = np.full(L, np.nan)
    rng = np.random.default_rng(seed)
    for i in range(L):
        c = ctrl.get(i, np.array([]))
        t = trt.get(i, np.array([]))
        if len(c) < min_n_c or len(t) < min_n_t:
            continue
        if method in METHODS_IF:
            # if-1D and if-2D share one branch — sklearn IsolationForest
            # accepts (n, d) for any d. Caller decides feature dim by
            # passing 1-D or 2-D arrays via collect_per_pos_events
            # feature_mode.
            from sklearn.ensemble import IsolationForest
            kw = {"contamination": contamination if contamination is not None else "auto"}
            iso = IsolationForest(n_estimators=if_n_estimators,
                                  max_samples=if_max_samples,
                                  random_state=seed, **kw)
            iso.fit(_as_2d(c))
            rates[i] = (iso.predict(_as_2d(t)) == -1).mean()
        elif method == "ks":
            n = min(len(c), len(t), subsample_cap)
            cs = rng.choice(c, n, replace=False)
            ts = rng.choice(t, n, replace=False)
            rates[i] = ks_2samp(cs, ts, method=ks_method).statistic
        elif method == "wass":
            n = min(len(c), len(t), subsample_cap)
            cs = rng.choice(c, n, replace=False)
            ts = rng.choice(t, n, replace=False)
            rates[i] = wasserstein_distance(cs, ts)
        elif method == "dmed":
            rates[i] = abs(np.median(t) - np.median(c))
        elif method in METHODS_OCSVM:
            # ---------------------------------------------------------------
            # ocsvm-1D: 1-D simplification of PORE-cupine
            # ocsvm-2D: 2-D (cal_mean, plan-A scaled std), closer to
            #           PORE-cupine's feature set but still differs in
            #           hyperparameters (see below).
            # ---------------------------------------------------------------
            # Faithful PORE-cupine reproduction lives at
            #   /scratch/cs/nanopore/chengg1/segSHAPE/baselines/PORE-cupine_reproduce/
            # (R + e1071 + nanopolish input). Differences from that pipeline:
            #
            #   feature dim   : ocsvm-1D 1-D (cal_mean only)
            #                   ocsvm-2D 2-D (cal_mean, plan-A cal_std)
            #                   PORE-cupine 2-D (cal_mean, log(event_stdv))
            #   nu            : ours default 0.05  (CLI configurable)
            #                   PORE-cupine 0.001  (extremely tight boundary)
            #   gamma         : ours 'scale' = 1 / (n_features · X.var())
            #                   PORE-cupine 0.0009  (fixed across datasets)
            #   read filter   : ours --coverage-filter (default off)
            #                   PORE-cupine n_pos > 0.5·ref_len enforced
            #
            # Net effect: ocsvm-1D reads modification from mean only;
            # ocsvm-2D additionally captures variance changes via the
            # plan-A-scaled std dimension. To do a proper benchmark vs
            # PORE-cupine, run the legacy R pipeline (see
            # baselines/PORE-cupine_reproduce/README.md), not this branch.
            from sklearn.svm import OneClassSVM
            nu_clip = max(1e-3, min(0.5, float(nu)))
            svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu_clip)
            svm.fit(_as_2d(c))
            rates[i] = float((svm.predict(_as_2d(t)) == -1).mean())
        elif method in METHODS_GMM:
            # gmm-1D / gmm-2D: fit GMM on control; threshold at the
            # gmm_quantile of control log-prob; rate = fraction of treated
            # below threshold. With gmm_n_comp='auto', try n_comp ∈ {1, 2}
            # and pick the lower BIC (penalised log-likelihood); fall back
            # to n_comp=1 if c is too small to fit 2 components. 2-D uses
            # full 2x2 covariance per component.
            from sklearn.mixture import GaussianMixture
            q = max(1e-3, min(0.5, float(gmm_quantile)))
            c_col = _as_2d(c)
            if gmm_n_comp == "auto":
                candidates = (1, 2) if len(c) >= 20 else (1,)
            elif gmm_n_comp in ("1", 1):
                candidates = (1,)
            elif gmm_n_comp in ("2", 2):
                if len(c) < 20:
                    candidates = (1,)              # forced fallback
                else:
                    candidates = (2,)
            else:
                raise ValueError(
                    f"gmm_n_comp must be 'auto', '1', or '2'; got {gmm_n_comp!r}")
            best_gm = None
            best_bic = np.inf
            for nc in candidates:
                try:
                    gm = GaussianMixture(
                        n_components=nc, covariance_type="full",
                        random_state=seed, reg_covar=1e-6)
                    gm.fit(c_col)
                    bic = gm.bic(c_col)
                    if bic < best_bic:
                        best_bic = bic
                        best_gm = gm
                except Exception:
                    continue
            if best_gm is None:
                continue                           # leave rates[i] = NaN
            log_p_c = best_gm.score_samples(c_col)
            thresh = float(np.quantile(log_p_c, q))
            log_p_t = best_gm.score_samples(_as_2d(t))
            rates[i] = float((log_p_t < thresh).mean())
        elif method == "xpore":
            rates[i] = _xpore_em(c, t)
        else:
            raise ValueError(method)
    return rates


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-position z-score on the finite slots only. NaN slots stay NaN."""
    out = x.copy().astype(np.float64)
    m = ~np.isnan(out)
    if m.sum() < 2:
        return out
    mu, sd = out[m].mean(), out[m].std()
    if sd > 0:
        out[m] = (out[m] - mu) / sd
    return out


def normalize_2_8(x: np.ndarray) -> np.ndarray:
    """Standard SHAPE 2–8 % normalization (Weeks lab convention).

    Operates on the finite slots only:
      1. Sort finite values ascending.
      2. Drop the **top 2 %** (extreme outliers — typically RT artifacts /
         spike-ins, not real modifications).
      3. Take the **mean of the top 8 %** of the remaining values (≈
         percentile 92–98 of the original sorted data) as the unit
         reference.
      4. Divide every finite value by that reference; NaN slots stay NaN.

    Output: reactive bases ≳ 0.7, protected bases ≈ 0; range loosely
    [0, ~2]. Unlike z-score, the absolute magnitudes are interpretable
    across positions / datasets and match the convention used in
    ``.shape`` / ``.dat`` files."""
    out = x.copy().astype(np.float64)
    m = ~np.isnan(out)
    finite = out[m]
    n = len(finite)
    if n < 50:
        return out                                  # not enough data
    finite_sorted = np.sort(finite)
    upper = finite_sorted[: int(n * 0.98)]          # drop top 2 %
    n_upper = len(upper)
    ref_slice = upper[int(0.92 * n_upper):]         # top 8 % of survivors
    if len(ref_slice) == 0:
        return out
    ref = float(ref_slice.mean())
    if ref > 0:
        out[m] = out[m] / ref
    return out


def normalize_boxplot(x: np.ndarray) -> np.ndarray:
    """SHAPE-MaP / ShapeMapper2 box-plot normalization (Weeks lab; the
    long-RNA variant, used for ≳ 100 nt).

    Operates on the finite slots only:
      1. Exclude high outliers: drop values above
         ``max(1.5·IQR, p90)`` (``p95`` when n ≤ 100).
      2. Reference = **mean of the top 10 %** of the survivors.
      3. Divide every finite value by that reference; NaN slots stay NaN.

    Like ``normalize_2_8`` (the '2–8 %' short-RNA variant) this only SCALES
    (no mean subtraction), so non-negative input stays non-negative — every
    position remains a usable RNAfold/Deigan constraint. The two are
    rank-identical and differ only by a scalar (outlier rule: IQR-based vs
    fixed top-2 %; reference: top-10 % vs top-8 %). Mirrors the normalization
    in ``baselines/nanoSHAPE_reproduce`` so a direct comparison is exact."""
    out = x.copy().astype(np.float64)
    m = ~np.isnan(out)
    finite = out[m]
    n = len(finite)
    if n < 50:
        return out                                  # not enough data
    q1, q3 = np.percentile(finite, 25), np.percentile(finite, 75)
    iqr = q3 - q1
    pct = np.percentile(finite, 90 if n > 100 else 95)
    threshold = max(1.5 * iqr, pct)
    survivors = finite[finite <= threshold]
    k = int(len(survivors) * 0.1)
    if k < 1:
        return out
    ref = float(np.sort(survivors)[-k:].mean())
    if ref > 0:
        out[m] = out[m] / ref
    return out


def normalize(x: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Dispatch ``method`` ∈ {'zscore', 'shape_28', 'boxplot', 'none'} to
    the corresponding normalizer. All methods preserve NaN slots."""
    if method == "zscore":
        return zscore(x)
    if method == "shape_28":
        return normalize_2_8(x)
    if method == "boxplot":
        return normalize_boxplot(x)
    if method == "none":
        return x.copy().astype(np.float64)
    raise ValueError(f"unknown --normalize method: {method!r}; "
                     "expected one of zscore / shape_28 / boxplot / none")


def moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centered moving average over a 1-D reactivity profile.

    For each position ``i`` with a finite input, the output value is the
    mean of the finite values in the window ``[i - W/2, i + W/2]`` (window
    size W = ``window``, centered). Edge positions use a partial window
    (``min_periods=1``). **Originally-NaN slots stay NaN** — we don't
    fabricate values at uncovered positions; only smooth the finite slots
    using neighboring information.

    ``window <= 1`` is a no-op (returns a float64 copy).

    Replicates the 5-nt smoothing step from Aw et al. 2020 (Nat Biotech;
    nanoSHAPE), which applies a centered moving average to the per-position
    reactivity profile **before** z-score normalization."""
    if window <= 1:
        return x.copy().astype(np.float64)
    out = (pd.Series(x).rolling(window, center=True, min_periods=1)
                       .mean().to_numpy().astype(np.float64))
    out[np.isnan(x)] = np.nan                  # restore NaN slots
    return out


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _resolve_paths(args: argparse.Namespace):
    """Resolve (ctrl_subevents, ctrl_align_dir, trt_subevents, trt_align_dir,
    out_dir).

    Mode A (structured): --root-dir --dataset --sweep-cell [--ctrl-sample
    --trt-sample]; subevents.parquet sits at
    <sample>/3_alignment/subevents.parquet and the alignment outputs at
    <sample>/3_alignment/<sweep_cell>/.
    Mode B (explicit): --ctrl-subevents --ctrl-align-dir --trt-subevents
    --trt-align-dir; --out-dir overrides the default
    `<trt_align_dir>/mod_rate/`.
    """
    if args.ctrl_align_dir and args.trt_align_dir:
        ctrl_align_dir = args.ctrl_align_dir
        trt_align_dir = args.trt_align_dir
        ctrl_subevents = args.ctrl_subevents or os.path.join(
            os.path.dirname(ctrl_align_dir.rstrip(os.sep)), "subevents.parquet")
        trt_subevents = args.trt_subevents or os.path.join(
            os.path.dirname(trt_align_dir.rstrip(os.sep)), "subevents.parquet")
    else:
        if not (args.root_dir and args.dataset and args.sweep_cell):
            raise SystemExit(
                "mod-calling: provide either --ctrl-align-dir + --trt-align-dir "
                "(Mode B) or --root-dir + --dataset + --sweep-cell (Mode A).")
        base = os.path.join(args.root_dir, "datasets", args.dataset)
        ctrl_dir = os.path.join(base, args.ctrl_sample, "3_alignment")
        trt_dir = os.path.join(base, args.trt_sample, "3_alignment")
        ctrl_subevents = os.path.join(ctrl_dir, "subevents.parquet")
        trt_subevents = os.path.join(trt_dir, "subevents.parquet")
        ctrl_align_dir = os.path.join(ctrl_dir, args.sweep_cell)
        trt_align_dir = os.path.join(trt_dir, args.sweep_cell)

    out_dir = args.out_dir or os.path.join(trt_align_dir, "mod_rate")
    return (ctrl_subevents, ctrl_align_dir, trt_subevents, trt_align_dir,
            out_dir)


def _parse_methods(arg_str: str) -> list:
    """Parse --method CLI string into an ordered list of method names.

    Accepts:
      single method        --method if-1D       -> ['if-1D']
      comma-separated      --method ks,wass     -> ['ks', 'wass']
      special keyword      --method all         -> all 10 methods in METHODS order
                           --method 1d / 2d     -> only the 1-D / 2-D variants

    Whitespace around commas is stripped; duplicates are dropped while
    preserving first-occurrence order. Each method must appear in
    METHODS or be a recognized keyword; otherwise SystemExit."""
    s = arg_str.strip()
    if s == "all":
        return list(METHODS)
    if s == "1d":
        return [m for m in METHODS if m not in METHODS_2D]
    if s == "2d":
        return list(METHODS_2D)
    methods = [m.strip() for m in s.split(",") if m.strip()]
    if not methods:
        raise SystemExit("--method: empty; pass at least one method name")
    invalid = [m for m in methods if m not in METHODS]
    if invalid:
        raise SystemExit(
            f"--method: unknown method(s) {invalid}; "
            f"valid: {', '.join(METHODS)} or one of 'all' / '1d' / '2d'")
    seen = set()
    deduped = []
    for m in methods:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


def _parse_max_samples(s: str):
    """Parse --max-samples CLI string to sklearn IF accepted form.

    Accepts:
      'auto' (sklearn IF default = min(256, n_samples))
      int string (e.g. '512' -> 512)
      float string in (0.0, 1.0] (e.g. '0.5' -> 0.5; fraction of n_c per tree)
    """
    if s == "auto":
        return "auto"
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            raise SystemExit(
                f"--max-samples must be 'auto', int, or float in (0, 1]; "
                f"got {s!r}")


def _resolve_contamination(args: argparse.Namespace,
                            ctrl: Dict[int, np.ndarray],
                            methods: list):
    """Resolve --contamination for method=if-1D / if-2D. Returns 'auto',
    a float, or None (other methods don't use it).

    `methods` is the parsed method list; contamination is only resolved
    if at least one IF method is in the list. The resolved value is
    shared across all IF methods in the run (no per-method
    contamination override)."""
    if not any(m in METHODS_IF for m in methods):
        return None
    if args.contamination == "auto":
        return "auto"
    if args.contamination == "adaptive":
        valid_n = [len(c) for c in ctrl.values() if len(c) > 0]
        if valid_n:
            n_c_p10 = int(np.percentile(valid_n, 10))
            n_c_med = int(np.median(valid_n))
        else:
            n_c_p10 = n_c_med = 0
        k_target = float(args.k_target)
        contam = max(0.0001, k_target / n_c_p10) if n_c_p10 > 0 else 0.0001
        if contam > 0.5:
            print(f"  WARNING: adaptive contam {contam:.4f} exceeds sklearn IF "
                  f"max 0.5; capping to 0.5. n_c_p10={n_c_p10} is too low — "
                  f"alignment likely under-covered.", flush=True)
            contam = 0.5
        print(f"  adaptive contam (k_target={k_target:g}): "
              f"n_c p10={n_c_p10} med={n_c_med}"
              f" -> max(0.0001, {k_target:g}/{n_c_p10}) = {contam:.4f}",
              flush=True)
        return contam
    return float(args.contamination)


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g_in = p.add_argument_group("input (Mode A: structured)")
    g_in.add_argument("--root-dir",
                      help="dataset root (contains datasets/<DATASET>/...)")
    g_in.add_argument("--dataset")
    g_in.add_argument("--sweep-cell",
                      help="event-align output sub-dir, e.g. "
                           "rna002_de50_dk15_bc0.0_sp50_shift_only")
    g_in.add_argument("--ctrl-sample", default="control")
    g_in.add_argument("--trt-sample", default="treated")

    g_in2 = p.add_argument_group("input (Mode B: explicit paths, override A)")
    g_in2.add_argument("--ctrl-align-dir",
                       help="directory containing alignment.csv + scale.csv (control)")
    g_in2.add_argument("--trt-align-dir",
                       help="directory containing alignment.csv + scale.csv (treated)")
    g_in2.add_argument("--ctrl-subevents",
                       help="control subevents.parquet "
                            "(default: <ctrl-align-dir>/../subevents.parquet)")
    g_in2.add_argument("--trt-subevents",
                       help="treated subevents.parquet "
                            "(default: <trt-align-dir>/../subevents.parquet)")

    g_m = p.add_argument_group("method")
    g_m.add_argument("--method", default="if-1D",
                     help="per-position test(s). Accepts a single method, "
                          "a comma-separated list, or one of the keywords "
                          "'all' / '1d' / '2d'. When multiple methods are "
                          "given, each writes to its own folder (no "
                          "conflicts) and the per-pos buckets are loaded "
                          "once per feature_mode and reused. Examples: "
                          "--method if-2D, --method ks,wass,ocsvm-1D, "
                          "--method all (= all 10 methods), --method 1d "
                          "(= 1-D variants + ks/wass/dmed/xpore), --method "
                          "2d (= if-2D, ocsvm-2D, gmm-2D only). Default "
                          "if-1D. Valid single methods: if-1D, if-2D, ks, "
                          "wass, dmed, ocsvm-1D, ocsvm-2D, gmm-1D, gmm-2D, "
                          "xpore. The '-1D' / '-2D' suffix on if/ocsvm/gmm "
                          "encodes feature dim ('-1D' = cal_mean only; "
                          "'-2D' = (cal_mean, plan-A scaled std) where "
                          "cal_std = std_pa / read_median(std_pa); see "
                          "docs/attention.md §5 for why per-read median, "
                          "NOT v_scale). ks/wass/dmed/xpore are 1-D by "
                          "definition.")
    g_m.add_argument("--min-n-c", type=int, default=256,
                     help="min control events/pos (default 256)")
    g_m.add_argument("--min-n-t", type=int, default=100,
                     help="min treated events/pos (default 100)")
    g_m.add_argument("--seed", type=int, default=42,
                     help="rng seed (default 42). Used by if-1D "
                          "(IsolationForest random_state), gmm-1D "
                          "(GaussianMixture random_state), and "
                          "ks/wass (sub-sample rng). dmed/ocsvm-1D/xpore "
                          "don't use it.")

    g_if = p.add_argument_group("method=if-1D (IsolationForest) parameters")
    g_if.add_argument("--contamination", default="0.005",
                      help="if-1D-only: float, 'auto', or 'adaptive'. Default "
                           "0.005 — matches ocsvm-1D --nu and gmm-1D "
                           "--gmm-quantile so the three methods share the "
                           "same control-tail-fraction calibration. Use "
                           "'adaptive' (= max(0.005, k_target/n_c_p10)) only "
                           "when you have wildly varying coverage and need "
                           "the threshold to scale with n_c.")
    g_if.add_argument("--k-target", type=float, default=100,
                      help="if-1D-only, --contamination=adaptive: target "
                           "tail-sample count at p10 of n_c (default 100). "
                           "Ignored when --contamination is a float or "
                           "'auto'.")
    g_if.add_argument("--n-estimators", type=int, default=None,
                      help="if-1D-only: number of isolation trees (sklearn "
                           "IF default 100). Increasing reduces score "
                           "variance ~1/sqrt(N) but rarely changes AUC since "
                           "AUC is rank-based; 300/1000 only worth trying if "
                           "you observe rank instability across seeds.")
    g_if.add_argument("--max-samples", default=None,
                      help="if-1D-only: per-tree sub-sample size. Accepts "
                           "'auto' (= min(256, n_c); sklearn default), an "
                           "int (e.g. 512, 1024), or a float in (0.0, 1.0] "
                           "(fraction of n_c). At our min_n_c=256 floor, "
                           "'auto' caps each tree at 256 samples even when "
                           "n_c is much higher; raising to 512/1024 lets "
                           "trees see more data per partition (deeper trees, "
                           "finer structure on high-coverage positions).")

    g_ks_wass = p.add_argument_group("method=ks / method=wass parameters")
    g_ks_wass.add_argument("--subsample-cap", type=int, default=None,
                           help="ks/wass-only: max events per side before "
                                "random sub-sample (default 5000). Caps make "
                                "per-position KS / Wasserstein distances "
                                "comparable across coverage.")

    g_ks = p.add_argument_group("method=ks parameters")
    g_ks.add_argument("--ks-method", default=None,
                     choices=["auto", "exact", "asymp"],
                     help="ks-only: scipy.stats.ks_2samp(method=...) solver "
                          "(default 'asymp'; 'exact' is O(n^2) and only used "
                          "for tiny n).")

    g_ocsvm = p.add_argument_group("method=ocsvm-1D (One-Class SVM) parameters")
    g_ocsvm.add_argument("--nu", type=float, default=0.005,
                         help="ocsvm-1D-only: nu, upper bound on training-set "
                              "outlier fraction. (0, 0.5]; default 0.005 — "
                              "tighter boundary closer to PORE-cupine's "
                              "0.001 than to sklearn's loose 0.05 default. "
                              "NOTE: ocsvm-1D is a 1-D simplification of "
                              "PORE-cupine (2-D, nu=0.001, gamma=0.0009). For "
                              "faithful PORE-cupine numbers run the legacy R "
                              "pipeline at "
                              "baselines/PORE-cupine_reproduce/ — see the "
                              "comment in compute_metric() for the gap.")

    g_gmm = p.add_argument_group("method=gmm-1D (GaussianMixture) parameters")
    g_gmm.add_argument("--gmm-quantile", type=float, default=0.005,
                       help="gmm-1D-only: quantile of control log-prob used "
                            "as threshold. (0, 0.5]; default 0.005.")
    g_gmm.add_argument("--gmm-n-comp", default=None,
                       choices=["auto", "1", "2"],
                       help="gmm-1D-only: number of GMM components. 'auto' "
                            "(default) picks 1 vs 2 by BIC on control; '1' / "
                            "'2' force the value (with auto-fallback to 1 if "
                            "n_c < 20).")

    g_f = p.add_argument_group("filters")
    g_f.add_argument("--coverage-filter", type=float, default=0.0,
                     help="drop reads whose pos coverage fraction < this")
    g_f.add_argument("--no-shift", action="store_true",
                     help="ignore per-read scale/shift; use raw mean_pa")
    g_f.add_argument("--no-qc-pass-only", action="store_true",
                     help="include reads with qc_tag != PASS")
    g_f.add_argument("--ctrl-whitelist", default=None,
                     help="path to text file with one control read_id per line")
    g_f.add_argument("--trt-whitelist", default=None,
                     help="path to text file with one treated read_id per line")

    g_o = p.add_argument_group("output")
    g_o.add_argument("--smooth-window", type=int, default=0,
                     help="N-nt centered moving-average smoothing of the "
                          "reactivity profile, applied BEFORE --normalize. "
                          "Default 0 = no smoothing (the production setting; "
                          "raw per-position z-score). Set to 5 for the Aw et "
                          "al. 2020 (Nat Biotech, nanoSHAPE) convention. "
                          "NaN slots stay NaN; finite slots get mean of "
                          "finite neighbours within the window (partial "
                          "window at edges). Encoded in the .dat filename "
                          "as `_smooth<W>` (always; the folder name is "
                          "config-free so multiple smooth/norm sweeps share "
                          "one folder)." )
    g_o.add_argument("--normalize", default="zscore",
                     choices=["zscore", "shape_28", "boxplot", "none"],
                     help="post-method normalization written to the "
                          ".dat reactivity. zscore (default) = "
                          "(x - mean(finite)) / std(finite) — NOTE this is "
                          "the only option that subtracts the mean, so ~half "
                          "the values go negative and RNAfold's Deigan method "
                          "treats negatives as missing (no constraint). "
                          "shape_28 = SHAPE 2-8%% normalization (drop top 2%%, "
                          "divide by mean of next 8%%; Weeks-lab short-RNA "
                          "variant). boxplot = SHAPE-MaP/ShapeMapper2 box-plot "
                          "normalization (drop IQR outliers, divide by mean of "
                          "top 10%%; Weeks-lab long-RNA variant, ≳100 nt). "
                          "shape_28 / boxplot only SCALE (no subtraction) so "
                          "non-negative rates stay non-negative — every "
                          "position remains a usable constraint. "
                          "none = pass-through (raw mod_rate copied).")
    g_o.add_argument("--variant-name", default="default",
                     help="leading tag of the per-run output folder name. "
                          "Use this to disambiguate sweeps over min_n_c / "
                          "min_n_t / xpore / dmed / etc. that don't get "
                          "their own auto-encoded suffix.")
    g_o.add_argument("--out-dir",
                     help="parent directory for the per-run folder. Default: "
                          "<trt-align-dir>/mod_rate/. Each run creates a "
                          "sub-folder <variant>_<method><param_tag>/ "
                          "containing mod_rate.csv (always) + "
                          "reactivity_smooth<W>_norm-<NORM>.dat (only "
                          "when --ref-fa is given). The smooth/norm tag "
                          "is on the .dat filename, NOT the folder, so "
                          "multiple post-processing configs share one "
                          "folder without re-running compute_metric.")
    g_o.add_argument("--ref-fa",
                     help="reference fasta. When provided, the run folder "
                          "also gets reactivity_z.dat in RNAfold / "
                          "RNAstructure SHAPE-constraint format (1-based "
                          "reference position, NaN -> -999). Without "
                          "--ref-fa the .dat is NOT written, because the "
                          "alignment-internal pos_idx is reverse-oriented "
                          "and offset from the reference (see "
                          "docs/attention.md §2 / evaluate.pipeline) and "
                          "would mis-align the SHAPE constraints.")
    g_o.add_argument("--contig", default=None,
                     help="contig in --ref-fa to use for ref_len. Default: "
                          "first record. Only consulted when --ref-fa is "
                          "provided.")
    g_o.add_argument("--offset", type=int, default=2,
                     help="kmer-center offset for the pos_idx -> reference "
                          "mapping: fa_pos = ref_len - pos_idx - offset "
                          "(1-indexed). Default 2 — matches both RNA002 "
                          "5-mer and RNA004 9-mer per docs/attention.md §2. "
                          "Sweep with `segshape evaluate --offsets 0,1,2,3,4` "
                          "to pick the best offset empirically per "
                          "(chemistry, dataset).")
    return p


def _validate_method_specific_args(args: argparse.Namespace,
                                     methods: list) -> None:
    """Reject explicit overrides of method-gated args when none of the
    methods being run can use them.

    Uses ``None`` sentinels for the strictly-gated args (subsample_cap,
    ks_method, gmm_n_comp, n_estimators, max_samples) so we can tell
    ``user passed`` from ``default``. Existing soft-gated args
    (--contamination, --k-target, --nu, --gmm-quantile) keep their
    concrete defaults and are silently ignored by methods that don't
    consume them.

    With a method list (--method ks,if-1D), the rule is "at least one
    of the methods must accept this flag". Flags that apply to a
    subset of methods are silently ignored by the others — e.g.
    --n-estimators 300 with --method ks,if-1D applies to if-1D only,
    no error."""
    if args.ks_method is not None and "ks" not in methods:
        raise SystemExit(
            f"--ks-method only applies to --method ks "
            f"(got --method {','.join(methods)})")
    if (args.subsample_cap is not None
            and not any(m in ("ks", "wass") for m in methods)):
        raise SystemExit(
            f"--subsample-cap only applies to --method ks or wass "
            f"(got --method {','.join(methods)})")
    if (args.gmm_n_comp is not None
            and not any(m in METHODS_GMM for m in methods)):
        raise SystemExit(
            f"--gmm-n-comp only applies to --method gmm-1D / gmm-2D "
            f"(got --method {','.join(methods)})")
    if (args.n_estimators is not None
            and not any(m in METHODS_IF for m in methods)):
        raise SystemExit(
            f"--n-estimators only applies to --method if-1D / if-2D "
            f"(got --method {','.join(methods)})")
    if (args.max_samples is not None
            and not any(m in METHODS_IF for m in methods)):
        raise SystemExit(
            f"--max-samples only applies to --method if-1D / if-2D "
            f"(got --method {','.join(methods)})")


def run(args: argparse.Namespace) -> int:
    methods = _parse_methods(args.method)
    _validate_method_specific_args(args, methods)
    (ctrl_subevents, ctrl_align_dir, trt_subevents, trt_align_dir,
     out_dir) = _resolve_paths(args)

    for label, path in [("ctrl subevents.parquet", ctrl_subevents),
                         ("trt subevents.parquet", trt_subevents),
                         ("ctrl alignment.csv",
                          os.path.join(ctrl_align_dir, "alignment.csv")),
                         ("trt alignment.csv",
                          os.path.join(trt_align_dir, "alignment.csv")),
                         ("ctrl scale.csv",
                          os.path.join(ctrl_align_dir, "scale.csv")),
                         ("trt scale.csv",
                          os.path.join(trt_align_dir, "scale.csv"))]:
        if not os.path.exists(path):
            raise SystemExit(f"mod-calling: missing {label}: {path}")

    ctrl_wl = _load_whitelist(args.ctrl_whitelist)
    trt_wl = _load_whitelist(args.trt_whitelist)

    apply_shift = not args.no_shift
    qc_pass_only = not args.no_qc_pass_only
    needed_modes = sorted({"mean_std" if m in METHODS_2D else "mean"
                           for m in methods})

    print(f"variant={args.variant_name} methods={','.join(methods)} "
          f"contam={args.contamination} apply_shift={apply_shift} "
          f"qc_pass_only={qc_pass_only} cov_filter={args.coverage_filter} "
          f"feature_modes={','.join(needed_modes)}",
          flush=True)
    print(f"  ctrl: {ctrl_align_dir}\n  trt : {trt_align_dir}", flush=True)

    # Load per-pos buckets once per feature_mode and reuse across all
    # methods that need that mode. Two modes max ('mean' and 'mean_std');
    # most runs only need one.
    ctrl_by_mode: Dict[str, dict] = {}
    trt_by_mode: Dict[str, dict] = {}
    for mode in needed_modes:
        print(f"Collect per-pos events feature_mode={mode} (control)...",
              flush=True)
        ctrl_by_mode[mode] = collect_per_pos_events(
            ctrl_subevents,
            os.path.join(ctrl_align_dir, "alignment.csv"),
            os.path.join(ctrl_align_dir, "scale.csv"),
            apply_shift=apply_shift,
            coverage_filter=args.coverage_filter,
            read_id_whitelist=ctrl_wl,
            qc_pass_only=qc_pass_only,
            feature_mode=mode,
        )
        print(f"Collect per-pos events feature_mode={mode} (treated)...",
              flush=True)
        trt_by_mode[mode] = collect_per_pos_events(
            trt_subevents,
            os.path.join(trt_align_dir, "alignment.csv"),
            os.path.join(trt_align_dir, "scale.csv"),
            apply_shift=apply_shift,
            coverage_filter=args.coverage_filter,
            read_id_whitelist=trt_wl,
            qc_pass_only=qc_pass_only,
            feature_mode=mode,
        )
        if not ctrl_by_mode[mode] or not trt_by_mode[mode]:
            raise SystemExit("mod-calling: empty per-pos buckets — check inputs")

    # L is consistent across modes (same alignment.csv, just different
    # feature dims), but compute it from whichever mode we have.
    sample_ctrl = ctrl_by_mode[needed_modes[0]]
    sample_trt = trt_by_mode[needed_modes[0]]
    L = max(max(sample_ctrl.keys()), max(sample_trt.keys())) + 1
    print(f"  L = {L} (max pos_idx + 1)", flush=True)

    min_n_c, min_n_t = args.min_n_c, args.min_n_t
    print(f"  min_n_c={min_n_c}  min_n_t={min_n_t}", flush=True)

    contam = _resolve_contamination(args, sample_ctrl, methods)

    # Resolve None sentinels to method defaults; the sentinel pattern lets
    # _validate_method_specific_args distinguish "user passed" from "default".
    subsample_cap = 5000 if args.subsample_cap is None else args.subsample_cap
    ks_method = "asymp" if args.ks_method is None else args.ks_method
    gmm_n_comp = "auto" if args.gmm_n_comp is None else args.gmm_n_comp
    if_n_estimators = 100 if args.n_estimators is None else args.n_estimators
    if_max_samples = ("auto" if args.max_samples is None
                       else _parse_max_samples(args.max_samples))

    # Read ref fasta once if provided (reused for all method outputs).
    ref_len = None
    if args.ref_fa:
        ref_seq = _read_fasta_seq(args.ref_fa, args.contig)
        ref_len = len(ref_seq)

    # smooth/norm tags go on the .dat filename, NOT on the folder name —
    # mod_rate (raw rate) is independent of (smooth, normalize), so a
    # single folder can host multiple .dat files for sweeps over those
    # post-processing parameters without re-running compute_metric.
    smooth_tag = f"_smooth{args.smooth_window}"
    norm_tag = f"_norm-{args.normalize.replace('_', '')}"

    written = []
    for method in methods:
        mode = "mean_std" if method in METHODS_2D else "mean"
        ctrl = ctrl_by_mode[mode]
        trt = trt_by_mode[mode]

        print(f"\n=== method={method} (feature_mode={mode}) ===", flush=True)
        rates = compute_metric(ctrl, trt, L, method,
                               contamination=contam,
                               nu=args.nu,
                               gmm_quantile=args.gmm_quantile,
                               gmm_n_comp=gmm_n_comp,
                               if_n_estimators=if_n_estimators,
                               if_max_samples=if_max_samples,
                               min_n_c=min_n_c, min_n_t=min_n_t,
                               seed=args.seed,
                               subsample_cap=subsample_cap,
                               ks_method=ks_method)
        n_eval = int(np.isfinite(rates).sum())
        print(f"  n_eval = {n_eval} / {L}", flush=True)

        if args.smooth_window > 0:
            rates_smooth = moving_avg(rates, args.smooth_window)
        else:
            rates_smooth = rates
        rates_z = normalize(rates_smooth, method=args.normalize)
        print(f"  smooth_window={args.smooth_window} "
              f"normalize={args.normalize}", flush=True)

        if method in METHODS_IF:
            param_tag = (f"_c{contam:.4f}" if isinstance(contam, float)
                         else f"_c{contam}")
        elif method in METHODS_OCSVM:
            param_tag = f"_nu{args.nu:.4f}"
        elif method in METHODS_GMM:
            param_tag = f"_q{args.gmm_quantile:.4f}"
        else:
            param_tag = ""
        folder_name = f"{args.variant_name}_{method}{param_tag}"
        run_dir = os.path.join(out_dir, folder_name)
        os.makedirs(run_dir, exist_ok=True)

        # mod_rate.csv: pos_idx, mod_rate (0-based pos_idx). Only the RAW
        # per-position rate is stored — it is independent of (smooth,
        # normalize), so this CSV is stable across post-processing sweeps.
        # The normalized/z-scored reactivity lives in the smooth/norm-tagged
        # .dat file below (the canonical per-config artefact). Downstream
        # `segshape fold` / `segshape evaluate` recompute the z-score from
        # `mod_rate` when they need it.
        mr_path = os.path.join(run_dir, "mod_rate.csv")
        pd.DataFrame({
            "pos_idx": np.arange(L),
            "mod_rate": rates,
        }).to_csv(mr_path, index=False, float_format="%.6f", na_rep="nan")

        print(f"  wrote {run_dir}/mod_rate.csv", flush=True)

        # The .dat encodes (smooth, norm) in its filename — multiple
        # post-processing configs can coexist in one folder.
        if ref_len is not None:
            dat_name = f"reactivity{smooth_tag}{norm_tag}.dat"
            dat_path = os.path.join(run_dir, dat_name)
            n_valid = _write_reactivity_dat(rates_z, ref_len, args.offset,
                                             dat_path)
            print(f"  wrote {run_dir}/{dat_name} "
                  f"(ref_len={ref_len}, offset={args.offset}, "
                  f"valid={n_valid}/{ref_len})",
                  flush=True)
        written.append(run_dir)

    print(f"\nDone. {len(written)} method(s) written under {out_dir}/",
          flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="segshape mod-calling")
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
