"""Smoke tests: package imports cleanly and CLI builds correctly."""

import subprocess
import sys

import pytest


def test_version_string():
    import segshape
    assert isinstance(segshape.__version__, str)
    parts = segshape.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])


def test_data_dir_kmer_models_present():
    """The k-mer tables the default (--norm med-mad) pipeline loads must
    ship with the package: the RNA002 5-mer and RNA004 9-mer
    normalized-level `_with_stdv` tables selected by event-align (see
    KMER_TABLES in align/anchored.py)."""
    from segshape.data import kmer_model_path
    import os
    for fname in ("ont_rna002_5mer_levels_v1_with_stdv_5_to_3.txt",
                  "ont_rna004_9mer_levels_v1_with_stdv_5_to_3.txt"):
        p = kmer_model_path(fname)
        assert os.path.exists(p), f"bundled kmer table missing: {p}"


def test_top_cli_builds():
    from segshape.cli import build_parser
    p = build_parser()
    # Each top-level sub-command must be present.
    actions = {a.dest: a for a in p._actions}
    sub = actions["cmd"]
    expected = {"pod5index", "dorado-extract", "segment", "event-align",
                "mod-calling", "fold", "evaluate", "plot"}
    assert expected.issubset(set(sub.choices.keys())), (
        f"missing subcommands: {expected - set(sub.choices.keys())}")


@pytest.mark.parametrize("path", [
    [],
    ["pod5index"],
    ["dorado-extract"],
    ["segment"],
    ["event-align"],
    ["mod-calling"],
    ["fold"],
    ["evaluate", "pipeline"],
    ["evaluate", "filter-ll"],
    ["evaluate", "summarize"],
    ["evaluate", "build-summary"],
    ["plot", "alignment-path"],
    ["plot", "dorado-mv"],
    ["plot", "segment"],
])
def test_help_invokes_cleanly(path):
    """Every advertised sub-command must accept --help without crashing."""
    res = subprocess.run(
        [sys.executable, "-m", "segshape.cli", *path, "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, (
        f"`segshape {' '.join(path)} --help` exited {res.returncode}\n"
        f"stderr:\n{res.stderr}")
    assert "usage:" in res.stdout.lower()


@pytest.mark.parametrize("flag", ["--version", "-v"])
def test_version_flag(flag):
    """`segshape --version` / `-v` prints 3 lines: version, copyright, license."""
    import segshape
    res = subprocess.run(
        [sys.executable, "-m", "segshape.cli", flag],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, f"exit {res.returncode}; stderr={res.stderr}"
    lines = res.stdout.strip().splitlines()
    assert len(lines) == 3, f"expected 3 lines, got: {lines!r}"
    assert lines[0] == f"segSHAPE {segshape.__version__}"
    assert lines[1].startswith("Copyright (C) ") and "Guangzhao Cheng" in lines[1]
    assert lines[2] == "License Apache-2.0"


def test_modification_cli_resolves_paths():
    """`segshape mod-calling` registers cleanly; without inputs it errs early
    via the path-resolution SystemExit (Mode A or Mode B required)."""
    from segshape.reactivity import cli as r_cli
    import argparse
    p = argparse.ArgumentParser()
    r_cli.register(p)
    args = p.parse_args([])  # no inputs at all
    with pytest.raises(SystemExit, match="Mode A.*Mode B|Mode B.*Mode A|--ctrl-align-dir"):
        args.func(args)


@pytest.mark.parametrize("argv,expect_msg", [
    # --ks-method only valid with --method ks
    (["--method", "wass",     "--ks-method", "exact"],     "--ks-method only applies"),
    (["--method", "if-1D",    "--ks-method", "auto"],      "--ks-method only applies"),
    (["--method", "dmed",     "--ks-method", "asymp"],     "--ks-method only applies"),
    # --subsample-cap only valid with ks / wass
    (["--method", "if-1D",    "--subsample-cap", "1000"],  "--subsample-cap only applies"),
    (["--method", "dmed",     "--subsample-cap", "1000"],  "--subsample-cap only applies"),
    (["--method", "ocsvm-1D", "--subsample-cap", "1000"],  "--subsample-cap only applies"),
    # --gmm-n-comp only valid with --method gmm-1D
    (["--method", "if-1D",    "--gmm-n-comp", "auto"],     "--gmm-n-comp only applies"),
    (["--method", "ks",       "--gmm-n-comp", "2"],        "--gmm-n-comp only applies"),
    (["--method", "ocsvm-1D", "--gmm-n-comp", "1"],        "--gmm-n-comp only applies"),
    # --n-estimators / --max-samples only valid with --method if-1D
    (["--method", "ks",       "--n-estimators", "300"],    "--n-estimators only applies"),
    (["--method", "ocsvm-1D", "--n-estimators", "100"],    "--n-estimators only applies"),
    (["--method", "wass",     "--max-samples", "512"],     "--max-samples only applies"),
    (["--method", "gmm-1D",   "--max-samples", "auto"],    "--max-samples only applies"),
])
def test_modification_method_specific_arg_gating(argv, expect_msg):
    """--ks-method must reject any --method != ks; --subsample-cap must
    reject any --method ∉ {ks, wass}. Validation runs before path-resolution
    so we don't need real input files."""
    from segshape.reactivity import cli as r_cli
    import argparse
    p = argparse.ArgumentParser()
    r_cli.register(p)
    args = p.parse_args(argv)
    with pytest.raises(SystemExit, match=expect_msg):
        args.func(args)


@pytest.mark.parametrize("argv", [
    # Valid combinations: gating must not fire
    ["--method", "ks",   "--ks-method", "exact",  "--subsample-cap", "1000"],
    ["--method", "ks",   "--subsample-cap", "1000"],
    ["--method", "wass", "--subsample-cap", "1000"],
    ["--method", "gmm-1D", "--gmm-n-comp", "auto"],
    ["--method", "gmm-1D", "--gmm-n-comp", "1"],
    ["--method", "gmm-1D", "--gmm-n-comp", "2", "--gmm-quantile", "0.1"],
    ["--method", "if-1D", "--n-estimators", "300"],
    ["--method", "if-1D", "--max-samples", "512"],
    ["--method", "if-1D", "--n-estimators", "1000", "--max-samples", "0.5"],
])
def test_modification_method_specific_arg_gating_passes(argv):
    """Valid (method, flag) combinations must NOT trigger the gating
    SystemExit. They should still hit the downstream path-resolution
    SystemExit (Mode A/B required), which is fine."""
    from segshape.reactivity import cli as r_cli
    import argparse
    p = argparse.ArgumentParser()
    r_cli.register(p)
    args = p.parse_args(argv)
    with pytest.raises(SystemExit) as excinfo:
        args.func(args)
    msg = str(excinfo.value)
    assert "ks-method" not in msg and "subsample-cap" not in msg \
        and "gmm-n-comp" not in msg, (
        f"unexpected gating fire: {msg!r}")


def test_normalize_zscore_filters_nan():
    """zscore must compute μ/σ on finite values only and leave NaN slots
    unchanged. Output's finite slots have mean ≈ 0, std ≈ 1."""
    import numpy as np
    from segshape.reactivity.calling import normalize
    x = np.array([1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0])
    z = normalize(x, "zscore")
    assert np.isnan(z[1]) and np.isnan(z[3])           # NaN preserved
    finite = z[~np.isnan(z)]
    assert abs(finite.mean()) < 1e-10                  # mean ≈ 0
    assert abs(finite.std() - 1.0) < 1e-10             # std ≈ 1


def test_normalize_shape_28_unit_reference():
    """SHAPE 2-8 % normalization: drop top 2 %, mean of next 8 % becomes
    the unit reference. Verify that the 92-98 percentile slice averages
    to ≈ 1 in the output."""
    import numpy as np
    from segshape.reactivity.calling import normalize
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.uniform(0, 10, n)
    # inject NaN at a few positions; they must stay NaN
    x[[5, 100, 999]] = np.nan
    y = normalize(x, "shape_28")
    assert np.isnan(y[5]) and np.isnan(y[100]) and np.isnan(y[999])
    finite = y[~np.isnan(y)]
    n_fin = len(finite)
    # reproduce the slicing: drop top 2 %, mean of top 8 % of survivors
    sorted_fin = np.sort(finite)
    upper = sorted_fin[: int(n_fin * 0.98)]
    ref_slice = upper[int(0.92 * len(upper)):]
    assert abs(ref_slice.mean() - 1.0) < 1e-6, (
        f"top 8% of survivors should mean to 1.0; got {ref_slice.mean()}")


def test_normalize_shape_28_short_arrays_pass_through():
    """Arrays with < 50 finite values must pass through unchanged
    (insufficient data for the 2-8 % statistic)."""
    import numpy as np
    from segshape.reactivity.calling import normalize
    x = np.array([1.0, 2.0, np.nan, 3.0])
    y = normalize(x, "shape_28")
    np.testing.assert_array_equal(y[~np.isnan(y)], x[~np.isnan(x)])


def test_normalize_boxplot_nonnegative_and_scales():
    """SHAPE-MaP/ShapeMapper2 box-plot normalization: divide by the mean of
    the top 10 % of IQR-outlier-filtered survivors. Like shape_28 it only
    scales (no subtraction), so non-negative input stays non-negative, and
    it is rank-identical to the raw input (monotonic)."""
    import numpy as np
    from segshape.reactivity.calling import normalize
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 500)
    x[[7, 250]] = np.nan
    y = normalize(x, "boxplot")
    assert np.isnan(y[7]) and np.isnan(y[250])
    fin_x, fin_y = x[~np.isnan(x)], y[~np.isnan(y)]
    assert (fin_y >= 0).all()                                  # non-negative
    # monotonic (rank-identical) since it is just division by a positive scalar
    assert np.corrcoef(fin_x, fin_y)[0, 1] > 0.9999
    # the IQR-filtered top-10% reference maps to ≈ 1
    q1, q3 = np.percentile(fin_x, 25), np.percentile(fin_x, 75)
    thr = max(1.5 * (q3 - q1), np.percentile(fin_x, 90))
    surv = np.sort(fin_x[fin_x <= thr])
    ref = surv[-int(len(surv) * 0.1):].mean()
    assert ref > 0
    np.testing.assert_allclose(fin_y, fin_x / ref, rtol=1e-9)


def test_normalize_boxplot_short_arrays_pass_through():
    """< 50 finite values must pass through unchanged."""
    import numpy as np
    from segshape.reactivity.calling import normalize
    x = np.array([1.0, 2.0, np.nan, 3.0])
    y = normalize(x, "boxplot")
    np.testing.assert_array_equal(y[~np.isnan(y)], x[~np.isnan(x)])


def test_normalize_none_passthrough():
    """method='none' must return the input as a float64 copy."""
    import numpy as np
    from segshape.reactivity.calling import normalize
    x = np.array([1.0, np.nan, 3.0])
    y = normalize(x, "none")
    assert y is not x                                  # is a copy
    assert y.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(y), np.isnan(x))
    np.testing.assert_array_equal(y[~np.isnan(y)], x[~np.isnan(x)])


def test_normalize_unknown_raises():
    import numpy as np
    from segshape.reactivity.calling import normalize
    with pytest.raises(ValueError, match="unknown --normalize"):
        normalize(np.array([1.0, 2.0]), "log")


@pytest.mark.parametrize("window", [0, 1])
def test_moving_avg_passthrough_for_window_le_1(window):
    import numpy as np
    from segshape.reactivity.calling import moving_avg
    x = np.array([1.0, 2.0, np.nan, 4.0])
    y = moving_avg(x, window)
    np.testing.assert_array_equal(np.isnan(y), np.isnan(x))
    np.testing.assert_array_equal(y[~np.isnan(y)], x[~np.isnan(x)])


def test_moving_avg_uniform_data_unchanged():
    """A constant signal must come through smoothing unchanged."""
    import numpy as np
    from segshape.reactivity.calling import moving_avg
    x = np.full(20, 0.7)
    y = moving_avg(x, 5)
    np.testing.assert_allclose(y, x, atol=1e-12)


def test_moving_avg_centered_on_linear_ramp():
    """5-nt centered MA on a perfectly linear ramp returns the ramp itself
    in the interior (centered MA of a linear function is the function),
    with edge values pulled toward the local mean (partial window)."""
    import numpy as np
    from segshape.reactivity.calling import moving_avg
    x = np.arange(20, dtype=np.float64)
    y = moving_avg(x, 5)
    # Interior (positions 2..17): centered MA of [-2..+2] equals x[i]
    np.testing.assert_allclose(y[2:18], x[2:18], atol=1e-12)
    # Edge: pos 0 has window of values [0, 1, 2] (min_periods=1) → mean = 1
    assert abs(y[0] - 1.0) < 1e-12
    # pos 1: window [0, 1, 2, 3] → mean = 1.5
    assert abs(y[1] - 1.5) < 1e-12


def test_moving_avg_preserves_nan_slots_and_smooths_neighbours():
    """A NaN slot stays NaN; its finite neighbours pick up the local mean
    of finite values within the window (excluding the NaN)."""
    import numpy as np
    from segshape.reactivity.calling import moving_avg
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
    y = moving_avg(x, 3)
    assert np.isnan(y[2])                              # NaN preserved
    # pos 1 window = {1, 2, NaN} → mean of finite = 1.5
    assert abs(y[1] - 1.5) < 1e-12
    # pos 3 window = {NaN, 4, 5} → mean of finite = 4.5
    assert abs(y[3] - 4.5) < 1e-12


def test_segment_trim_stats_zero_is_raw():
    """trim=0 must return the un-trimmed mean/std verbatim."""
    import numpy as np
    from segshape.segment.events import _trim_stats
    rng = np.random.default_rng(0)
    seg = rng.normal(80.0, 1.5, 50)
    m, s = _trim_stats(seg, trim=0.0)
    assert abs(m - float(seg.mean())) < 1e-12
    assert abs(s - float(seg.std())) < 1e-12


def test_segment_trim_stats_matches_scipy_trim_mean():
    """Check that _trim_stats(seg, 0.1).mean() ≈ scipy trim_mean(seg, 0.1)
    — required so existing alignment numerics are byte-identical when we
    re-segment with the new (fixed-asymmetry) code."""
    import numpy as np
    from scipy.stats import trim_mean
    from segshape.segment.events import _trim_stats
    rng = np.random.default_rng(0)
    seg = rng.normal(80.0, 1.5, 50)
    m, _ = _trim_stats(seg, trim=0.1)
    assert abs(m - float(trim_mean(seg, 0.1))) < 1e-12


def test_segment_trim_stats_drops_boundary_transitions():
    """Synthesise a per-subevent signal: 2 boundary samples in transition
    (large |Δ| from core) at each end, then steady-state core. trim=0
    inflates std from the transitions; trim=0.1 must drop them and
    recover the steady core's std (≈ core σ)."""
    import numpy as np
    from segshape.segment.events import _trim_stats
    rng = np.random.default_rng(0)
    core = rng.normal(80.0, 1.0, 16)              # steady, σ ≈ 1
    seg = np.concatenate([[100.0, 95.0], core, [60.0, 65.0]])  # 20 samples
    _, s_raw = _trim_stats(seg, trim=0.0)
    m_trim, s_trim = _trim_stats(seg, trim=0.1)
    # Trimmed std must be much smaller (close to core σ ≈ 1)
    assert s_raw > 5 * s_trim, (
        f"expected trim to drop boundary jitter; raw σ={s_raw:.2f} "
        f"vs trim σ={s_trim:.2f}")
    # Trimmed std close to core σ
    assert abs(s_trim - core.std()) < 0.2, (
        f"trim σ should match core σ; got {s_trim:.3f} vs core "
        f"σ={core.std():.3f}")
    # Trimmed mean equals core mean (boundary samples gone)
    assert abs(m_trim - core.mean()) < 1e-12


def test_segment_trim_stats_short_seg_falls_back_to_raw():
    """When 2·int(n·trim) >= n trimming would empty the slice; the helper
    must gracefully fall back to seg.mean/std."""
    import numpy as np
    from segshape.segment.events import _trim_stats
    seg = np.array([70.0, 90.0, 80.0])             # n=3, trim 0.4 → 0
    # trim 0.4 → int(3*0.4)=1 each end → only 1 sample left, OK
    m, s = _trim_stats(seg, trim=0.4)
    assert np.isfinite(m) and np.isfinite(s)
    # n=2, trim 0.5 would be illegal CLI value but helper must not crash
    seg2 = np.array([70.0, 90.0])
    m2, s2 = _trim_stats(seg2, trim=0.49)         # int(2*0.49)=0 → no trim
    assert abs(m2 - 80.0) < 1e-12


def test_segment_cli_writes_subevents_parquet_filename():
    """--help text references subevents.parquet (filename rename to match
    paper terminology)."""
    res = subprocess.run(
        [sys.executable, "-m", "segshape.cli", "segment", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert "subevents.parquet" in res.stdout
    assert "--trim P" in res.stdout
    assert "--resplit-std" in res.stdout


def test_segment_resplit_off_is_passthrough():
    """``resplit_std=0`` (default) must leave bounds bit-identical."""
    import numpy as np
    from segshape.segment.events import _resplit_high_std_subevents
    rng = np.random.default_rng(0)
    clip = rng.normal(80.0, 1.5, 200).astype(np.float32)
    bounds = np.array([0, 50, 100, 150, 200], dtype=np.int64)
    out = _resplit_high_std_subevents(
        clip, bounds,
        resplit_std=0.0, resplit_distance=5, resplit_smooth=1)
    np.testing.assert_array_equal(out, bounds)


def test_segment_resplit_below_threshold_unchanged():
    """If all subevents have std below threshold, bounds stay the same."""
    import numpy as np
    from segshape.segment.events import _resplit_high_std_subevents
    rng = np.random.default_rng(0)
    clip = rng.normal(80.0, 1.5, 200).astype(np.float32)
    bounds = np.array([0, 50, 100, 150, 200], dtype=np.int64)
    out = _resplit_high_std_subevents(
        clip, bounds,
        resplit_std=10.0, resplit_distance=5, resplit_smooth=1)
    np.testing.assert_array_equal(out, bounds)


def test_segment_resplit_splits_bimodal_subevent():
    """Synthetic bimodal subevent: 35 samples at ~80 pA + 65 samples at
    ~120 pA. Trimmed std on the merged 100-sample subevent is ~18 pA.
    The asymmetric dwell keeps the linear-R² below the shape-gate ceiling
    (a perfectly symmetric 50/50 step has R²≈0.75 > 0.70 and is treated as
    a ramp — see _resplit_high_std_subevents gate (4)). With resplit_std=10
    we should detect and split into ≥ 2 pieces."""
    import numpy as np
    from segshape.segment.events import (_resplit_high_std_subevents,
                                          _trim_stats)
    rng = np.random.default_rng(0)
    low = rng.normal(80.0, 1.0, 35).astype(np.float32)
    high = rng.normal(120.0, 1.0, 65).astype(np.float32)
    clip = np.concatenate([low, high])
    bounds = np.array([0, 100], dtype=np.int64)              # one big subevent

    # Confirm the merged subevent's std is high
    _, sd = _trim_stats(clip, 0.1)
    assert sd > 15, f"setup: expected high merged std, got {sd}"

    out = _resplit_high_std_subevents(
        clip, bounds,
        resplit_std=10.0, resplit_distance=5, resplit_smooth=1)
    # Must have inserted at least one new boundary
    assert len(out) > len(bounds), (
        f"expected split, bounds unchanged: {out}")
    # And one of the new boundaries should land near the transition (sample 35)
    interior = [b for b in out if 0 < b < 100]
    assert any(abs(int(b) - 35) <= 6 for b in interior), (
        f"no boundary near the bimodal transition; got interior={interior}")


def test_segment_resplit_max_pieces_caps_new_boundaries():
    """A high-std subevent with many internal transitions must produce at
    most ``resplit_max_pieces - 1`` new boundaries, keeping only the
    strongest sub-peaks by slope amplitude. (Post-split verification may
    reject splits that don't reduce overall noise — the cap is an
    upper bound, not a guarantee.)"""
    import numpy as np
    from segshape.segment.events import _resplit_high_std_subevents
    rng = np.random.default_rng(0)
    # 4-level signal: 70 → 110 → 70 → 110 → 70 pA with sharp transitions
    # every 30 samples. find_peaks at distance=5 detects 4 transitions;
    # the cap controls how many we accept.
    levels = [70, 110, 70, 110, 70]
    pieces = [rng.normal(L, 1.0, 30) for L in levels]
    clip = np.concatenate(pieces).astype(np.float32)
    bounds = np.array([0, len(clip)], dtype=np.int64)
    for max_pieces in (2, 3, 4):
        out = _resplit_high_std_subevents(
            clip, bounds,
            resplit_std=5.0, resplit_distance=5, resplit_smooth=1,
            resplit_max_pieces=max_pieces)
        n_new_bounds = len(out) - len(bounds)
        assert n_new_bounds <= max_pieces - 1, (
            f"max_pieces={max_pieces} should cap new boundaries to "
            f"{max_pieces - 1}; got {n_new_bounds} new bounds: {out}")


def test_segment_resplit_bimodal_one_transition_accepts():
    """Single-transition bimodal (35/65 asymmetric step, so linear-R² stays
    under the shape gate): max_pieces=2 must produce exactly 1 new boundary
    (post-split verification accepts because both halves' std drops
    significantly relative to the merged whole)."""
    import numpy as np
    from segshape.segment.events import _resplit_high_std_subevents
    rng = np.random.default_rng(0)
    low = rng.normal(80.0, 1.0, 35).astype(np.float32)
    high = rng.normal(120.0, 1.0, 65).astype(np.float32)
    clip = np.concatenate([low, high])
    bounds = np.array([0, 100], dtype=np.int64)
    out = _resplit_high_std_subevents(
        clip, bounds,
        resplit_std=5.0, resplit_distance=5, resplit_smooth=1,
        resplit_max_pieces=2)
    assert len(out) == 3, f"expected 1 new boundary; got bounds={out}"
    interior = int(out[1])
    assert abs(interior - 35) <= 6, (
        f"new boundary should land near transition (sample 35); got {interior}")


def test_segment_resplit_pure_noise_rejected():
    """A high-std subevent that's pure noise (no real transition) — split
    candidates exist but post-split verification rejects them because
    neither half has lower std than the original."""
    import numpy as np
    from segshape.segment.events import _resplit_high_std_subevents
    rng = np.random.default_rng(0)
    # 100 samples of pure white noise with σ ≈ 6 pA → trim std ≈ 6
    clip = rng.normal(80.0, 6.0, 100).astype(np.float32)
    bounds = np.array([0, 100], dtype=np.int64)
    out = _resplit_high_std_subevents(
        clip, bounds,
        resplit_std=4.0, resplit_distance=5, resplit_smooth=1,
        resplit_max_pieces=3)
    # Verification should reject: both halves of pure noise have
    # similar std; cannot improve.
    np.testing.assert_array_equal(out, bounds)


def test_segment_resplit_short_subevent_skipped():
    """Subevents shorter than 2*resplit_min_piece_len are not re-split
    (would produce too-tiny pieces). Default min_piece_len=4 → anything
    < 8 samples is skipped."""
    import numpy as np
    from segshape.segment.events import _resplit_high_std_subevents
    rng = np.random.default_rng(0)
    clip = np.concatenate([
        rng.normal(80.0, 1.0, 3),
        rng.normal(120.0, 1.0, 3),
    ]).astype(np.float32)
    bounds = np.array([0, 6], dtype=np.int64)         # 6 < 2*4 = 8 → skip
    out = _resplit_high_std_subevents(
        clip, bounds,
        resplit_std=5.0, resplit_distance=5, resplit_smooth=1)
    np.testing.assert_array_equal(out, bounds)


# --- segment --norm med-mad path -------------------------------------------

def test_segment_med_mad_normalisation_returns_robust_scale():
    """med_mad_normalisation(sig) on a noise-free Gaussian: shift=median,
    scale=1.4826·MAD ≈ σ. Verifies the helper itself, not the pipeline."""
    import numpy as np
    from segshape.segment.events import med_mad_normalisation
    rng = np.random.default_rng(0)
    sig = rng.normal(80.0, 1.5, 100_000).astype(np.float32)
    shift, scale = med_mad_normalisation(sig)
    assert abs(shift - 80.0) < 0.05         # median ≈ mean for Gaussian
    assert abs(scale - 1.5)  < 0.05         # 1.4826·MAD ≈ σ for Gaussian


def test_segment_med_mad_zero_mad_floor():
    """Constant signal → MAD=0; scale must not be 0 (would div-by-zero
    downstream). The 1e-6 floor in med_mad_normalisation kicks in."""
    import numpy as np
    from segshape.segment.events import med_mad_normalisation
    sig = np.full(1000, 95.0, dtype=np.float32)
    shift, scale = med_mad_normalisation(sig)
    assert shift == 95.0
    assert scale == 1e-6


def test_segment_signal_to_events_norm_negative_mean_legal():
    """With norm='med-mad' the m<0 guard is bypassed: low-current
    subevents emit a finite negative mean rather than NaN. (Under
    norm='none', the same negative-mean subevent would be blanked.)"""
    import numpy as np
    from segshape.segment.events import _signal_to_events
    # Two noisy plateaus straddling 0; std > 0 so the sd==0 guard never
    # fires, and find_peaks puts a boundary at the level transition.
    rng = np.random.default_rng(0)
    clip = np.concatenate([
        rng.normal(-1.5, 0.3, 30),                # negative mean, σ-units
        rng.normal(+1.5, 0.3, 30),
    ]).astype(np.float32)
    out_norm = _signal_to_events(clip, peak_distance=5, smooth_box=1,
                                  abs_offset=0, trim=0.0, norm="med-mad")
    finite_norm = [m for (_, _, _, m, _) in out_norm if np.isfinite(m)]
    # At least one finite negative mean survives under norm='med-mad'
    assert any(m < 0 for m in finite_norm), (
        f"norm='med-mad' should keep finite negative means; got {finite_norm}")
    # Same clip under norm='none': every negative-mean subevent gets
    # blanked to NaN by the pA<0 guard.
    out_pa = _signal_to_events(clip, peak_distance=5, smooth_box=1,
                                abs_offset=0, trim=0.0, norm="none")
    finite_pa = [m for (_, _, _, m, _) in out_pa if np.isfinite(m)]
    assert not any(m < 0 for m in finite_pa), (
        f"norm='none' must blank pA<0 subevents; got {finite_pa}")


def test_segment_signal_to_events_norm_outlier_blanked():
    """abs(m) > 50 sanity guard fires under norm='med-mad' and blanks
    the subevent to (NaN, NaN). Single huge subevent → single NaN row."""
    import numpy as np
    from segshape.segment.events import _signal_to_events
    clip = np.full(40, 100.0, dtype=np.float32)          # m=100 → |m|>50
    out = _signal_to_events(clip, peak_distance=5, smooth_box=1,
                            abs_offset=0, trim=0.0, norm="med-mad")
    assert len(out) >= 1
    # Every emitted subevent should be NaN because |m|=100 > 50.
    for (_, _, _, m, sd) in out:
        assert np.isnan(m) and np.isnan(sd), \
            f"|m|>50 outlier should be blanked; got m={m}, sd={sd}"


def test_segment_signal_to_events_norm_outlier_not_in_pa_mode():
    """The abs(m)>50 guard is norm-specific: in norm='none' (raw pA), a
    legit mean like 100 pA is healthy and must NOT be blanked."""
    import numpy as np
    from segshape.segment.events import _signal_to_events
    rng = np.random.default_rng(0)
    clip = rng.normal(100.0, 1.0, 40).astype(np.float32)   # m ≈ 100 pA OK
    out = _signal_to_events(clip, peak_distance=5, smooth_box=1,
                            abs_offset=0, trim=0.0, norm="none")
    assert len(out) >= 1
    # Healthy pA subevent stays finite (NOT blanked by the abs>50 guard).
    assert any(np.isfinite(m) for (_, _, _, m, _) in out), \
        "abs(m)>50 guard must NOT fire under norm='none' for legit pA"


def test_segment_signal_to_events_norm_peak_positions_invariant():
    """find_peaks operates on |slope|; med-mad rescaling is linear and
    only scales slope amplitudes — peak POSITIONS must be bit-identical
    to the raw-pA case (segmentation boundaries don't shift with norm).
    """
    import numpy as np
    from segshape.segment.events import (_signal_to_events,
                                          med_mad_normalisation)
    rng = np.random.default_rng(0)
    # Bimodal-ish signal: 2 plateaus with a sharp transition
    sig = np.concatenate([
        rng.normal(80.0, 1.0, 60),
        rng.normal(95.0, 1.0, 60),
        rng.normal(80.0, 1.0, 60),
    ]).astype(np.float32)
    raw = _signal_to_events(sig, peak_distance=5, smooth_box=1,
                            abs_offset=0, trim=0.0, norm="none")
    # Same signal, but pre-normalised in σ-units
    shift, scale = med_mad_normalisation(sig)
    sig_norm = ((sig - shift) / scale).astype(np.float32)
    norm = _signal_to_events(sig_norm, peak_distance=5, smooth_box=1,
                             abs_offset=0, trim=0.0, norm="med-mad")
    raw_bounds  = [(s, e) for (_, s, e, _, _) in raw]
    norm_bounds = [(s, e) for (_, s, e, _, _) in norm]
    assert raw_bounds == norm_bounds, (
        f"peak positions must be scale-invariant; raw={raw_bounds} "
        f"vs norm={norm_bounds}")


def test_segment_schema_metadata_records_norm_field():
    """_schema_with_metadata must record the `norm` choice so downstream
    readers can verify whether mean_pa/std_pa are in pA or σ-units."""
    from segshape.segment.events import _schema_with_metadata
    s_none = _schema_with_metadata(trim=0.1, resplit_std=3.0, norm="none")
    s_norm = _schema_with_metadata(trim=0.1, resplit_std=0.2, norm="med-mad")
    assert s_none.metadata[b"norm"] == b"none"
    assert s_norm.metadata[b"norm"] == b"med-mad"
    # resplit_std value is recorded verbatim (no auto-rescaling)
    assert s_none.metadata[b"resplit_std"] == b"3.0"
    assert s_norm.metadata[b"resplit_std"] == b"0.2"


def test_segment_cli_exposes_norm_flag():
    """--help text must advertise the --norm flag and its choices."""
    res = subprocess.run(
        [sys.executable, "-m", "segshape.cli", "segment", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert "--norm" in res.stdout
    assert "med-mad" in res.stdout
    # subevents.norm.parquet auto-rename is documented
    assert "subevents.norm.parquet" in res.stdout


# --- end segment --norm med-mad path ---------------------------------------


def test_scale_shift_clip_summary_off_returns_empty():
    """fit_mode='off' must short-circuit (no calibration → no clip rate)."""
    from segshape.align.anchored import _scale_shift_clip_summary
    n_p, n_sh, n_v, s = _scale_shift_clip_summary(
        [], (-30.0, 30.0), (0.5, 2.0), 'off')
    assert (n_p, n_sh, n_v, s) == (0, 0, 0, "")


def test_scale_shift_clip_summary_counts_bound_hits():
    """Synthetic reads: verify shift_only and shift_v_scale modes count
    the right axis clips."""
    import numpy as np
    from segshape.align.anchored import ReadData, _scale_shift_clip_summary
    hb = (-30.0, 30.0)
    vb = (0.5, 2.0)

    def mk(shift_raw, v_raw=1.0, has_align=True):
        em = np.zeros(10, dtype=np.float64)
        r = ReadData(read_idx=0, read_id="x", k_seed=0, k_end=1, j_min=0,
                     j_max=1, event_means=em, raw_idx=np.zeros(10, np.int64))
        r.alignment = np.full(10, 0, np.int64) if has_align else None
        r.shift_raw = float(shift_raw)
        r.v_scale_raw = float(v_raw)
        return r

    reads = [
        mk(-30.0, 1.0),   # shift clipped at low
        mk(30.0, 1.0),    # shift clipped at high
        mk(0.0, 1.0),     # not clipped
        mk(0.0, 0.5),     # v_scale clipped at low (also)
        mk(50.0, 2.5),    # shift outside, v_scale outside (also)
    ]
    # shift_only: only shift dim reported
    n_p, n_sh, n_v, s = _scale_shift_clip_summary(reads, hb, vb, 'shift_only')
    assert n_p == 5
    assert n_sh == 3       # rows 0, 1, 4
    assert n_v == 0        # not reported under shift_only
    assert "shift clipped 3/5 (60.0%)" in s
    assert "v_scale clipped" not in s

    # shift_v_scale: both reported
    n_p, n_sh, n_v, s = _scale_shift_clip_summary(reads, hb, vb,
                                                  'shift_v_scale')
    assert n_p == 5
    assert n_sh == 3
    assert n_v == 2        # rows 3, 4
    assert "shift clipped 3/5" in s
    assert "v_scale clipped 2/5" in s


def test_fit_scale_shift_rejects_theil_sen():
    """The legacy 'theil_sen' fit mode was removed — calling it must raise."""
    import numpy as np
    from segshape.align.anchored import fit_scale_shift
    with pytest.raises(ValueError, match="unknown fit mode.*theil_sen"):
        fit_scale_shift(np.arange(50, dtype=float),
                        np.arange(50, dtype=float),
                        sigma_aligned=None,
                        shift_bounds=(-30.0, 30.0),
                        mode='theil_sen')


def test_fit_scale_shift_shift_v_scale_closed_form():
    """shift_v_scale mode: shift = median(μ - event); v_scale² = mean(resid²/σ²).
    Verify the closed-form values for a synthetic case where shift = 5 pA
    and the residuals (after shift correction) have RMS = 2 × σ_aligned."""
    import numpy as np
    from segshape.align.anchored import fit_scale_shift
    rng = np.random.default_rng(0)
    n = 200
    sigma_aligned = np.full(n, 1.5)
    true_shift = 5.0
    true_v = 2.0
    # Build data: μ - event = true_shift, plus residual noise N(0, true_v·σ).
    mu = rng.normal(80.0, 5.0, n)
    events = mu - true_shift + rng.normal(0.0, true_v * 1.5, n)
    sc, sh, v, sc_r, sh_r, v_r = fit_scale_shift(
        events, mu, sigma_aligned,
        shift_bounds=(-30.0, 30.0),
        v_scale_bounds=(0.5, 3.0),
        mode='shift_v_scale')
    # shift recovered to ≈ true (median is robust; tolerance accounts for
    # finite-sample SE ≈ 1.25·σ_resid/√n on median estimator)
    assert abs(sh - true_shift) < 1.0, f"shift={sh}, expected ~{true_shift}"
    # v_scale recovered to ≈ true (closed-form ML)
    assert abs(v - true_v) < 0.2, f"v_scale={v}, expected ~{true_v}"
    # scale always 1
    assert sc == 1.0


def test_fit_scale_shift_shift_v_scale_requires_sigma():
    """shift_v_scale mode without sigma_aligned must error clearly."""
    import numpy as np
    from segshape.align.anchored import fit_scale_shift
    with pytest.raises(ValueError, match="requires sigma_aligned"):
        fit_scale_shift(np.zeros(50), np.zeros(50), sigma_aligned=None,
                        shift_bounds=(-30.0, 30.0),
                        mode='shift_v_scale')


def test_fit_scale_shift_off_returns_identity():
    """mode='off' returns (1, 0, 1) with NaN raw fields."""
    import numpy as np
    from segshape.align.anchored import fit_scale_shift
    sc, sh, v, sc_r, sh_r, v_r = fit_scale_shift(
        np.zeros(100), np.zeros(100), sigma_aligned=None,
        shift_bounds=(-30.0, 30.0),
        mode='off')
    assert (sc, sh, v) == (1.0, 0.0, 1.0)
    assert np.isnan(sc_r) and np.isnan(sh_r) and np.isnan(v_r)


def test_event_align_cli_rejects_theil_sen():
    """`segshape event-align --fit-mode theil_sen` must fail at argparse
    with a non-zero exit and a 'invalid choice' message in stderr."""
    res = subprocess.run(
        [sys.executable, "-m", "segshape.cli", "event-align",
         "--root-dir", "/tmp/_does_not_matter",
         "--dataset", "x", "--sample", "control",
         "--reference-file", "ref.fa", "--contig", "x",
         "--fit-mode", "theil_sen"],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode != 0
    assert "invalid choice" in res.stderr.lower() and "theil_sen" in res.stderr


def test_compute_metric_gmm_bic_prefers_two_on_bimodal():
    """gmm_n_comp='auto' must pick 2 components on a clearly bimodal
    control (BIC for 2 should be lower than for 1).

    Constructs synthetic per-position ctrl/trt buckets where ctrl is
    bimodal (two well-separated Gaussians) and trt is single-modal at a
    third location. Then runs compute_metric three ways:
      - gmm_n_comp='1' : forces 1-component fit (over-smoothed)
      - gmm_n_comp='2' : forces 2-component fit (correct)
      - gmm_n_comp='auto' : BIC selects 2 (matches the '2' result)

    With min_n_c=256, min_n_t=100 we ensure the position is evaluated."""
    import numpy as np
    from segshape.reactivity.calling import compute_metric

    rng = np.random.default_rng(0)
    pos = 0
    # bimodal control: two Gaussians at 70 and 90 pA, σ=2
    c = np.concatenate([rng.normal(70, 2, 300), rng.normal(90, 2, 300)])
    # treated: shifted mode at 110 (well outside both control modes)
    t = rng.normal(110, 2, 200)

    ctrl = {pos: c}
    trt = {pos: t}
    L = pos + 1

    rates_1 = compute_metric(ctrl, trt, L, "gmm-1D", gmm_n_comp="1",
                              min_n_c=256, min_n_t=100)
    rates_2 = compute_metric(ctrl, trt, L, "gmm-1D", gmm_n_comp="2",
                              min_n_c=256, min_n_t=100)
    rates_auto = compute_metric(ctrl, trt, L, "gmm-1D", gmm_n_comp="auto",
                                 min_n_c=256, min_n_t=100)

    # All three must produce a finite rate (treated is far from ctrl, so
    # rate should be very high — close to 1.0).
    for r in (rates_1, rates_2, rates_auto):
        assert np.isfinite(r[pos]), f"got NaN: {r}"
    # For this synthetic case, rate should be near 1.0 (treated entirely
    # outside both control modes).
    assert rates_2[pos] > 0.9
    # auto must match the 2-component result on this bimodal control
    # (BIC-2 wins decisively because data is generated as a 2-Gaussian mix).
    assert rates_auto[pos] == rates_2[pos], (
        f"BIC should pick 2 on bimodal control; "
        f"got auto={rates_auto[pos]:.3f} vs 2={rates_2[pos]:.3f}, "
        f"1-comp={rates_1[pos]:.3f}")


def test_parse_methods_single_list_and_keywords():
    """_parse_methods must dispatch:
       single  : 'if-1D'           -> ['if-1D']
       list    : 'ks,wass,if-2D'   -> ['ks', 'wass', 'if-2D'] (order preserved)
       'all'   : -> all 10 in METHODS order
       '1d'    : -> 1-D + 1-D-only methods (ks/wass/dmed/xpore)
       '2d'    : -> only METHODS_2D
       dedupe  : 'ks,ks,wass'      -> ['ks', 'wass']
       invalid : 'ks,bogus'        -> SystemExit
    """
    from segshape.reactivity.calling import (_parse_methods, METHODS,
                                              METHODS_2D)

    assert _parse_methods("if-1D") == ["if-1D"]
    assert _parse_methods("ks,wass,if-2D") == ["ks", "wass", "if-2D"]
    assert _parse_methods("ks, wass , if-2D") == ["ks", "wass", "if-2D"]  # whitespace
    assert _parse_methods("ks,ks,wass") == ["ks", "wass"]
    assert _parse_methods("all") == list(METHODS)
    assert _parse_methods("2d") == list(METHODS_2D)
    one_d = _parse_methods("1d")
    assert all(m not in METHODS_2D for m in one_d)
    # 1d should include all non-2D methods
    assert set(one_d) == set(METHODS) - set(METHODS_2D)
    with pytest.raises(SystemExit, match="unknown method"):
        _parse_methods("ks,bogus")


def test_mod_calling_runs_multi_method_writes_one_folder_each(tmp_path, monkeypatch):
    """`--method ks,wass,if-1D` must produce three sibling output folders
    under --out-dir, each with its own mod_rate.csv. Per-pos buckets
    should be loaded ONCE per feature_mode (here all three are 1-D,
    so collect_per_pos_events is called twice total: ctrl + trt)."""
    import argparse
    import os
    import numpy as np
    import pandas as pd
    from segshape.reactivity import calling

    rng = np.random.default_rng(42)
    L = 30
    ctrl_buckets = {i: rng.normal(0.0, 1.0, 300) for i in range(L)}
    trt_buckets = {i: rng.normal(0.0, 1.0, 200) for i in range(L)}
    trt_buckets[10] = rng.normal(3.0, 1.0, 200)         # mod signal

    n_calls = {"count": 0}
    def fake_collect(*a, **kw):
        n_calls["count"] += 1
        return ctrl_buckets if "ctrl" in str(a[0]) else trt_buckets

    monkeypatch.setattr(calling, "collect_per_pos_events", fake_collect)
    monkeypatch.setattr(calling, "_resolve_paths", lambda args: (
        "ctrl_subevents", "ctrl_align", "trt_subevents", "trt_align",
        str(tmp_path)))
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    p = argparse.ArgumentParser()
    calling.add_arguments(p)
    args = p.parse_args(["--method", "ks,wass,if-1D",
                          "--variant-name", "multi",
                          "--out-dir", str(tmp_path)])
    rc = calling.run(args)
    assert rc == 0

    # Three sibling folders (one per method)
    expected = [
        tmp_path / "multi_ks",
        tmp_path / "multi_wass",
        tmp_path / "multi_if-1D_c0.0050",
    ]
    for folder in expected:
        assert folder.is_dir(), folder
        assert (folder / "mod_rate.csv").is_file()

    # Per-pos buckets loaded twice (ctrl + trt) for the single 1-D mode,
    # NOT 6 times (would happen if reused per method).
    assert n_calls["count"] == 2, n_calls["count"]


def test_mod_calling_runs_2d_loads_buckets_once_per_mode(tmp_path, monkeypatch):
    """`--method if-1D,if-2D` spans two feature_modes (mean + mean_std).
    Buckets must be loaded ONCE per mode (= 4 calls: 2 modes × ctrl/trt),
    not once per method."""
    import argparse
    import os
    import numpy as np
    from segshape.reactivity import calling

    rng = np.random.default_rng(43)
    L = 30
    # 1-D buckets (returned when feature_mode='mean')
    ctrl_1d = {i: rng.normal(0.0, 1.0, 300) for i in range(L)}
    trt_1d = {i: rng.normal(0.0, 1.0, 200) for i in range(L)}
    trt_1d[5] = rng.normal(3.0, 1.0, 200)
    # 2-D buckets (returned when feature_mode='mean_std')
    ctrl_2d = {i: np.column_stack([rng.normal(0, 1, 300),
                                    rng.normal(1, 0.1, 300)]) for i in range(L)}
    trt_2d = {i: np.column_stack([rng.normal(0, 1, 200),
                                   rng.normal(1, 0.1, 200)]) for i in range(L)}
    trt_2d[5] = np.column_stack([rng.normal(3, 1, 200),
                                  rng.normal(1.5, 0.1, 200)])

    n_calls = {"count": 0, "modes": []}
    def fake_collect(*a, **kw):
        n_calls["count"] += 1
        n_calls["modes"].append(kw.get("feature_mode", "?"))
        is_ctrl = "ctrl" in str(a[0])
        if kw.get("feature_mode") == "mean_std":
            return ctrl_2d if is_ctrl else trt_2d
        return ctrl_1d if is_ctrl else trt_1d

    monkeypatch.setattr(calling, "collect_per_pos_events", fake_collect)
    monkeypatch.setattr(calling, "_resolve_paths", lambda args: (
        "ctrl_subevents", "ctrl_align", "trt_subevents", "trt_align",
        str(tmp_path)))
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    p = argparse.ArgumentParser()
    calling.add_arguments(p)
    args = p.parse_args(["--method", "if-1D,if-2D",
                          "--variant-name", "modes",
                          "--out-dir", str(tmp_path)])
    rc = calling.run(args)
    assert rc == 0

    assert (tmp_path / "modes_if-1D_c0.0050").is_dir()
    assert (tmp_path / "modes_if-2D_c0.0050").is_dir()
    # 4 collect calls: 2 modes × {ctrl, trt}
    assert n_calls["count"] == 4
    assert n_calls["modes"].count("mean") == 2
    assert n_calls["modes"].count("mean_std") == 2


def test_mod_calling_method_arg_gating_strict_for_irrelevant_flags(tmp_path):
    """If NONE of the methods in the list can use a strictly-gated
    flag, the gating SystemExit fires. If at least one method accepts
    it, gating passes."""
    from segshape.reactivity import cli as r_cli
    import argparse

    p = argparse.ArgumentParser()
    r_cli.register(p)

    # Pure 1-D methods + --n-estimators (IF-only) → reject
    args = p.parse_args(["--method", "ks,wass", "--n-estimators", "300"])
    with pytest.raises(SystemExit, match="--n-estimators only applies"):
        args.func(args)

    # Mixed list including if-1D + --n-estimators → must pass gating
    # (will hit downstream path-resolution SystemExit, which is fine)
    args = p.parse_args(["--method", "ks,if-1D", "--n-estimators", "300"])
    with pytest.raises(SystemExit) as excinfo:
        args.func(args)
    assert "--n-estimators" not in str(excinfo.value)


def test_mod_calling_smooth_norm_sweep_coexist_in_one_folder(tmp_path, monkeypatch):
    """Two runs of the same method with different (smooth_window,
    normalize) into the same --out-dir must produce ONE folder
    (named without smooth/norm tags) and TWO .dat files (named with
    smooth/norm tags). The mod_rate.csv is overwritten but its
    `mod_rate` column is invariant under (smooth, normalize)."""
    import argparse
    import os
    import numpy as np
    import pandas as pd
    from segshape.reactivity import calling

    rng = np.random.default_rng(13)
    L = 30
    ctrl_buckets = {i: rng.normal(0.0, 1.0, 300) for i in range(L)}
    trt_buckets = {i: rng.normal(0.0, 1.0, 200) for i in range(L)}
    trt_buckets[10] = rng.normal(3.0, 1.0, 200)

    monkeypatch.setattr(calling, "collect_per_pos_events",
                         lambda *a, **kw: ctrl_buckets if "ctrl" in str(
                             a[0]) else trt_buckets)
    monkeypatch.setattr(calling, "_resolve_paths", lambda args: (
        "ctrl_subevents", "ctrl_align", "trt_subevents", "trt_align",
        str(tmp_path)))
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    fa = tmp_path / "ref.fa"
    fa.write_text(">testseq\n" + "ACGU" * 7 + "AC\n")

    p = argparse.ArgumentParser()
    calling.add_arguments(p)

    # Run 1: smooth=5, zscore (explicit; default is now 0)
    args1 = p.parse_args(["--method", "dmed",
                           "--variant-name", "sweep",
                           "--out-dir", str(tmp_path),
                           "--ref-fa", str(fa),
                           "--smooth-window", "5"])
    assert calling.run(args1) == 0

    # Run 2: smooth=0, zscore (default; same folder, different post-proc)
    args2 = p.parse_args(["--method", "dmed",
                           "--variant-name", "sweep",
                           "--out-dir", str(tmp_path),
                           "--ref-fa", str(fa),
                           "--smooth-window", "0"])
    assert calling.run(args2) == 0

    folder = tmp_path / "sweep_dmed"
    assert folder.is_dir()
    # mod_rate.csv overwritten but only one copy
    assert (folder / "mod_rate.csv").is_file()
    # Two distinct .dat files for the two post-proc configs
    dats = sorted(p.name for p in folder.glob("reactivity*.dat"))
    assert dats == [
        "reactivity_smooth0_norm-zscore.dat",
        "reactivity_smooth5_norm-zscore.dat",
    ], dats

    # mod_rate column unchanged across runs (smooth/norm only affect
    # the .dat content, not the raw mod_rate)
    mr = pd.read_csv(folder / "mod_rate.csv")
    assert "mod_rate" in mr.columns
    assert mr["mod_rate"].iloc[10] > 0


def test_mod_calling_output_layout_no_dat_without_ref_fa(tmp_path, monkeypatch):
    """Without --ref-fa, mod-calling writes a folder with mod_rate.csv
    only — NO reactivity .dat (since the pos_idx -> reference mapping
    requires ref_len + offset, and writing a misaligned .dat would
    feed wrong SHAPE constraints to RNAfold)."""
    import argparse
    import os
    import numpy as np
    import pandas as pd
    from segshape.reactivity import calling

    rng = np.random.default_rng(99)
    L = 30
    ctrl_buckets = {i: rng.normal(0.0, 1.0, 300) for i in range(L)}
    trt_buckets = {i: rng.normal(0.0, 1.0, 200) for i in range(L)}
    trt_buckets[10] = rng.normal(3.0, 1.0, 200)
    trt_buckets[5] = rng.normal(0.0, 1.0, 50)            # gated by min_n_t

    monkeypatch.setattr(calling, "collect_per_pos_events",
                         lambda *a, **kw: ctrl_buckets if "ctrl" in str(
                             a[0]) else trt_buckets)
    monkeypatch.setattr(calling, "_resolve_paths", lambda args: (
        "ctrl_subevents", "ctrl_align", "trt_subevents", "trt_align",
        str(tmp_path)))
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    p = argparse.ArgumentParser()
    calling.add_arguments(p)
    args = p.parse_args(["--method", "dmed",
                          "--variant-name", "smoketest",
                          "--out-dir", str(tmp_path)])
    rc = calling.run(args)
    assert rc == 0

    folder = tmp_path / "smoketest_dmed"
    assert folder.is_dir()
    assert (folder / "mod_rate.csv").is_file()
    # No --ref-fa => .dat must NOT be written
    assert not list(folder.glob("reactivity*.dat"))

    mr = pd.read_csv(folder / "mod_rate.csv")
    assert list(mr.columns) == ["pos_idx", "mod_rate"]
    assert np.isnan(mr.loc[mr["pos_idx"] == 5, "mod_rate"].iloc[0])
    assert mr.loc[mr["pos_idx"] == 10, "mod_rate"].iloc[0] > 0


def test_mod_calling_output_layout_dat_uses_reverse_offset(tmp_path, monkeypatch):
    """With --ref-fa, reactivity_smooth<W>_norm-<NORM>.dat must use the
       fa_pos = ref_len - pos_idx - offset
    mapping (1-indexed), NOT a naive pos_idx+1. Filename encodes
    smooth/norm so multiple post-proc configs share one folder.
    Construct synthetic data where pos_idx=10 has a clear modification
    spike, ref_len=30, offset=2, then check that fa = 30 - 10 - 2 = 18
    has a finite value and fa=29/30 are -999 (no mapping coverage)."""
    import argparse
    import os
    import numpy as np
    from segshape.reactivity import calling

    rng = np.random.default_rng(101)
    L = 30
    ctrl_buckets = {i: rng.normal(0.0, 1.0, 300) for i in range(L)}
    trt_buckets = {i: rng.normal(0.0, 1.0, 200) for i in range(L)}
    trt_buckets[10] = rng.normal(5.0, 1.0, 200)          # strong mod signal

    monkeypatch.setattr(calling, "collect_per_pos_events",
                         lambda *a, **kw: ctrl_buckets if "ctrl" in str(
                             a[0]) else trt_buckets)
    monkeypatch.setattr(calling, "_resolve_paths", lambda args: (
        "ctrl_subevents", "ctrl_align", "trt_subevents", "trt_align",
        str(tmp_path)))
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    # fake ref fasta (any 30-base seq; only ref_len matters for .dat)
    fa = tmp_path / "ref.fa"
    fa.write_text(">testseq\n" + "ACGU" * 7 + "AC\n")     # 30 bases

    p = argparse.ArgumentParser()
    calling.add_arguments(p)
    args = p.parse_args(["--method", "dmed",
                          "--variant-name", "smoketest",
                          "--out-dir", str(tmp_path),
                          "--ref-fa", str(fa),
                          "--offset", "2"])
    rc = calling.run(args)
    assert rc == 0

    folder = tmp_path / "smoketest_dmed"
    # Default smooth=0, normalize=zscore -> reactivity_smooth0_norm-zscore.dat
    dat_path = folder / "reactivity_smooth0_norm-zscore.dat"
    assert dat_path.is_file(), list(folder.glob("*.dat"))
    lines = dat_path.read_text().strip().split("\n")
    assert len(lines) == 30                               # ref_len lines

    # Parse to dict {pos: val_str}
    parsed = {int(line.split("\t")[0]): line.split("\t")[1] for line in lines}

    # pos_idx=10 maps to fa = 30 - 10 - 2 = 18 — must NOT be -999
    assert parsed[18] != "-999", (
        f"pos_idx=10 (mod signal) should land at fa=18, got {parsed[18]!r}")

    # The first `offset` positions and the last `offset` positions of
    # the .dat are unmapped (no pos_idx maps there) — must be -999.
    # ref_len - i - offset is in [1, ref_len] iff i in [0, ref_len-offset-1]
    # = [0, 27]. So fa positions 1..2 (the `offset` positions) and
    # 29..30 (top end) — actually let me recompute:
    #   i=0 -> fa = 30 - 0 - 2 = 28
    #   i=L-1=29 -> fa = 30 - 29 - 2 = -1 (out of range, dropped)
    # So fa positions 1, 2 never receive a mapping (lowest fa = 30-29-2 = -1
    # for i=29 which is out of range; for i=L-1=29 nothing). Top end:
    # fa=28 from i=0 is the highest. So fa=29, fa=30 are never mapped.
    assert parsed[29] == "-999"
    assert parsed[30] == "-999"


def test_plan_a_scale_std_normalizes_per_read():
    """_plan_a_scale_std must divide each event's std_pa by that read's
    median std_pa (not the global median). Reads with all-equal std_pa
    must end up with cal_std == 1.0 everywhere."""
    import numpy as np
    import pandas as pd
    from segshape.reactivity.calling import _plan_a_scale_std

    df = pd.DataFrame({
        "read_id":  ["a", "a", "a", "b", "b"],
        "std_pa":   [2.0, 4.0, 6.0, 1.0, 1.0],
        # noise columns to confirm only std_pa is used:
        "mean_pa":  [50.0, 60.0, 70.0, 80.0, 90.0],
    })
    out = _plan_a_scale_std(df)
    # read 'a': median(2, 4, 6) = 4.0 → cal_std = [0.5, 1.0, 1.5]
    np.testing.assert_allclose(out.loc[out["read_id"] == "a", "cal_std"],
                                [0.5, 1.0, 1.5])
    # read 'b': median(1, 1) = 1.0 → cal_std = [1.0, 1.0]
    np.testing.assert_allclose(out.loc[out["read_id"] == "b", "cal_std"],
                                [1.0, 1.0])
    # original std_pa unchanged
    np.testing.assert_allclose(out["std_pa"], [2.0, 4.0, 6.0, 1.0, 1.0])


def test_plan_a_scale_std_floors_zero_median():
    """A read with all-zero std_pa would divide-by-zero → cal_std = inf
    without the 1e-3 floor. The helper must clip the median to 1e-3."""
    import numpy as np
    import pandas as pd
    from segshape.reactivity.calling import _plan_a_scale_std

    df = pd.DataFrame({
        "read_id": ["x", "x"],
        "std_pa":  [0.0, 0.0],
    })
    out = _plan_a_scale_std(df)
    # 0 / max(median, 1e-3) = 0 / 1e-3 = 0  (finite, not inf/nan)
    assert np.all(np.isfinite(out["cal_std"]))
    np.testing.assert_allclose(out["cal_std"], [0.0, 0.0])


def test_compute_metric_2d_methods_run_and_produce_finite_rates():
    """if-2D, ocsvm-2D, gmm-2D must accept (n, 2) feature arrays and
    produce finite mod-rates on synthetic well-separated 2-D data."""
    import numpy as np
    from segshape.reactivity.calling import compute_metric

    rng = np.random.default_rng(11)
    # control: cal_mean ~ N(0, 1), cal_std ~ N(1, 0.1)
    c = np.column_stack([rng.normal(0.0, 1.0, 600),
                          rng.normal(1.0, 0.1, 600)])
    # treated: shifted mean AND broadened std (modification signature)
    t = np.column_stack([rng.normal(3.0, 1.0, 200),
                          rng.normal(1.5, 0.1, 200)])

    for m in ("if-2D", "ocsvm-2D", "gmm-2D"):
        rates = compute_metric({0: c}, {0: t}, 1, m,
                                contamination=0.05,
                                nu=0.05,
                                gmm_quantile=0.05,
                                gmm_n_comp="auto",
                                if_n_estimators=100,
                                if_max_samples="auto",
                                min_n_c=256, min_n_t=100)
        assert np.isfinite(rates[0]), m
        assert 0.0 < rates[0] <= 1.0, (m, rates[0])


def test_compute_metric_1d_methods_unchanged_on_1d_input():
    """1-D methods must still accept (n,) arrays after the 2-D refactor.
    Regression check that _as_2d() inside the if/ocsvm/gmm branches
    doesn't break the 1-D path."""
    import numpy as np
    from segshape.reactivity.calling import compute_metric

    rng = np.random.default_rng(12)
    c = rng.normal(0.0, 1.0, 600)
    t = rng.normal(3.0, 1.0, 200)
    for m in ("if-1D", "ocsvm-1D", "gmm-1D"):
        rates = compute_metric({0: c}, {0: t}, 1, m,
                                contamination=0.05,
                                nu=0.05,
                                gmm_quantile=0.05,
                                gmm_n_comp="auto",
                                if_n_estimators=100,
                                if_max_samples="auto",
                                min_n_c=256, min_n_t=100)
        assert np.isfinite(rates[0]), m
        assert 0.0 < rates[0] <= 1.0, (m, rates[0])


def test_parse_max_samples_accepts_auto_int_and_float():
    """_parse_max_samples must dispatch by string content: 'auto' →
    'auto', '512' → 512, '0.5' → 0.5; nonsense raises SystemExit."""
    from segshape.reactivity.calling import _parse_max_samples

    assert _parse_max_samples("auto") == "auto"
    assert _parse_max_samples("512") == 512
    assert isinstance(_parse_max_samples("512"), int)
    assert _parse_max_samples("0.5") == 0.5
    assert isinstance(_parse_max_samples("0.5"), float)
    with pytest.raises(SystemExit, match="--max-samples"):
        _parse_max_samples("bogus")


def test_compute_metric_if_1d_n_estimators_and_max_samples_forward():
    """compute_metric must forward if_n_estimators / if_max_samples to
    sklearn IsolationForest. Verify by calling with each combination on
    well-separated synthetic data and checking the rate is finite and
    in (0, 1] (high modification signal in treated)."""
    import numpy as np
    from segshape.reactivity.calling import compute_metric

    rng = np.random.default_rng(7)
    c = rng.normal(0.0, 1.0, 600)
    t = rng.normal(4.0, 1.0, 200)
    ctrl, trt = {0: c}, {0: t}

    for n_est, max_s in [(100, "auto"), (300, "auto"),
                          (100, 512), (100, 0.5)]:
        rates = compute_metric(ctrl, trt, 1, "if-1D",
                               contamination=0.05,
                               if_n_estimators=n_est,
                               if_max_samples=max_s,
                               min_n_c=256, min_n_t=100)
        assert np.isfinite(rates[0]), (n_est, max_s)
        assert 0.0 < rates[0] <= 1.0, (n_est, max_s, rates[0])


def test_compute_metric_xpore_no_modification_returns_low():
    """xpore on identical control / treated distributions should return a
    score close to 0 (no per-component mixing-weight gap)."""
    import numpy as np
    from segshape.reactivity.calling import compute_metric

    rng = np.random.default_rng(0)
    c = rng.normal(0.0, 1.0, 600)
    t = rng.normal(0.0, 1.0, 400)
    rates = compute_metric({0: c}, {0: t}, 1, "xpore",
                           min_n_c=256, min_n_t=100)
    assert np.isfinite(rates[0])
    assert 0.0 <= rates[0] <= 1.0
    # No real mixing-weight separation → score should be small.
    assert rates[0] < 0.15, rates[0]


def test_compute_metric_xpore_detects_partial_modification():
    """xpore on (unimodal control) vs (50% modified treated) should
    return a score near 0.5 — π_t[mod] ≈ 0.5, π_c[mod] ≈ 0."""
    import numpy as np
    from segshape.reactivity.calling import compute_metric

    rng = np.random.default_rng(1)
    c = rng.normal(0.0, 1.0, 600)
    t = np.concatenate([rng.normal(0.0, 1.0, 200),
                        rng.normal(4.0, 1.0, 200)])
    rates = compute_metric({0: c}, {0: t}, 1, "xpore",
                           min_n_c=256, min_n_t=100)
    assert np.isfinite(rates[0])
    assert 0.30 < rates[0] < 0.70, rates[0]


def test_compute_metric_xpore_gated_by_min_n():
    """xpore must respect min_n_c / min_n_t (NaN when under-coverage)."""
    import numpy as np
    from segshape.reactivity.calling import compute_metric

    rng = np.random.default_rng(2)
    c = rng.normal(0.0, 1.0, 50)         # below default min_n_c=256
    t = rng.normal(2.0, 1.0, 50)
    rates = compute_metric({0: c}, {0: t}, 1, "xpore",
                           min_n_c=256, min_n_t=100)
    assert np.isnan(rates[0])


# ---------------------------------------------------------------------------
# segshape.io.pod5_index — read_id → filename Parquet index for a pod5 folder
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_pod5_folder(tmp_path, monkeypatch):
    """Create an empty pod5 folder + monkey-patch pod5.Reader to return
    predetermined read_ids per filename, so we can exercise build_index
    without needing a real pod5 writer fixture."""
    folder = tmp_path / "pod5"
    folder.mkdir()
    layout = {
        "a.pod5": ["r1", "r2", "r3"],
        "b.pod5": ["r4", "r5"],
        "c.pod5": ["r6"],
    }
    for name in layout:
        (folder / name).touch()

    class FakeReader:
        def __init__(self, path):
            self._ids = layout[path.name]
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        @property
        def read_ids(self):
            return self._ids

    from segshape.io import pod5_index
    monkeypatch.setattr(pod5_index.pod5, "Reader", FakeReader)
    return folder, layout


def test_pod5_index_path_convention(tmp_path):
    from segshape.io.pod5_index import index_path_for
    p = tmp_path / "x" / "y" / "pod5"
    p.mkdir(parents=True)
    out = index_path_for(p)
    assert out.name == "pod5.index"
    assert out.parent == p.parent.resolve()


def test_pod5_index_build_and_load(fake_pod5_folder):
    from segshape.io.pod5_index import build_index, load_index
    folder, layout = fake_pod5_folder
    out = build_index(folder)

    assert out.exists()
    assert out.name == "pod5.index"
    df = load_index(out)
    assert list(df.columns) == ["read_id", "filename"]
    expected_rows = sum(len(v) for v in layout.values())
    assert len(df) == expected_rows
    expected_pairs = {(rid, fn) for fn, ids in layout.items() for rid in ids}
    assert set(zip(df.read_id, df.filename)) == expected_pairs


def test_pod5_index_refuses_overwrite_without_force(fake_pod5_folder):
    from segshape.io.pod5_index import build_index
    folder, _ = fake_pod5_folder
    build_index(folder)
    with pytest.raises(FileExistsError):
        build_index(folder)
    # --force overwrites cleanly
    out = build_index(folder, force=True)
    assert out.exists()


def test_pod5_index_empty_folder_raises(tmp_path):
    from segshape.io.pod5_index import build_index
    empty = tmp_path / "empty_pod5"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        build_index(empty)


def test_pod5_index_duplicate_ids_warn_but_keep_all(tmp_path, monkeypatch):
    """Same read_id in two pod5 files: warn, but keep every occurrence."""
    folder = tmp_path / "pod5"
    folder.mkdir()
    layout = {"a.pod5": ["dup", "u1"], "b.pod5": ["dup", "u2"]}
    for n in layout:
        (folder / n).touch()

    class FakeReader:
        def __init__(self, p):
            self._ids = layout[p.name]
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        @property
        def read_ids(self): return self._ids

    from segshape.io import pod5_index
    monkeypatch.setattr(pod5_index.pod5, "Reader", FakeReader)

    with pytest.warns(UserWarning, match="appear in >1 pod5 file"):
        out = pod5_index.build_index(folder)
    df = pod5_index.load_index(out)
    assert len(df) == 4               # all rows kept
    assert df.read_id.nunique() == 3  # but only 3 distinct ids


def _verify_dups_fixture(tmp_path, monkeypatch, conflicting=False):
    """Helper for the two --verify-dups tests.

    Builds a 2-file fake pod5 folder where read_id ``dup`` appears in both
    files. With conflicting=False both copies share an identical biological
    fingerprint (chunk-overlap case); with conflicting=True the second
    file's record has a different num_samples (real id collision).
    """
    from types import SimpleNamespace
    folder = tmp_path / "pod5"
    folder.mkdir()
    layout = {"a.pod5": ["dup", "u1"], "b.pod5": ["dup", "u2"]}
    for n in layout:
        (folder / n).touch()

    run_info = dict(
        acquisition_id="A1", acquisition_start_time="2026-01-01T00:00:00",
        flow_cell_id="FC1", protocol_run_id="PR1",
        sample_id="S1", sample_rate=4000,
    )

    def make_rec(rid, num_samples=1000):
        return SimpleNamespace(
            read_id=rid,
            num_samples=num_samples,
            start_sample=12345,
            median_before=200.0,
            calibration=SimpleNamespace(scale=0.5, offset=10.0),
            run_info=SimpleNamespace(**run_info),
        )

    fps = {
        "a.pod5": {"dup": make_rec("dup"), "u1": make_rec("u1")},
        "b.pod5": {
            "dup": make_rec("dup", num_samples=999 if conflicting else 1000),
            "u2": make_rec("u2"),
        },
    }

    class FakeReader:
        def __init__(self, p):
            self._name = p.name
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        @property
        def read_ids(self):
            return layout[self._name]
        def reads(self, selection=None):
            store = fps[self._name]
            ids = selection if selection is not None else layout[self._name]
            for rid in ids:
                if rid in store:
                    yield store[rid]

    from segshape.io import pod5_index
    monkeypatch.setattr(pod5_index.pod5, "Reader", FakeReader)
    return folder


def test_pod5_index_verify_dups_byte_identical_succeeds(tmp_path, monkeypatch, capsys):
    """Dup with identical fingerprints across files: build succeeds, prints
    'all dups are byte-redundant', and writes the index."""
    folder = _verify_dups_fixture(tmp_path, monkeypatch, conflicting=False)
    from segshape.io import pod5_index
    with pytest.warns(UserWarning, match="appear in >1 pod5 file"):
        out = pod5_index.build_index(folder, verify_dups=True)
    assert out.exists()
    captured = capsys.readouterr()
    assert "all 1 duplicate read_id(s) are byte-redundant" in captured.err
    df = pod5_index.load_index(out)
    assert len(df) == 4
    assert df.read_id.nunique() == 3


def test_pod5_index_verify_dups_conflict_raises(tmp_path, monkeypatch):
    """Dup with mismatched fingerprints across files: build raises ValueError
    naming the offending read_id."""
    folder = _verify_dups_fixture(tmp_path, monkeypatch, conflicting=True)
    from segshape.io import pod5_index
    with pytest.warns(UserWarning, match="appear in >1 pod5 file"), \
         pytest.raises(ValueError, match="INCONSISTENT metadata"):
        pod5_index.build_index(folder, verify_dups=True)


# ---------------------------------------------------------------------------
# segshape.io.dorado_mv — extract per-read alignment + mv signal interval CSV
# ---------------------------------------------------------------------------


def _write_minimal_dorado_bam(path, records, ref_len=50):
    """Write a tiny valid BAM with the dorado tags extract_mv looks for.

    `records` is a list of dicts; each dict yields one AlignedSegment with:
      name, flag, ref_start, cigar (list of (op, len)), seq, ts, ns,
      stride, n_moves, pt (poly-A length, optional), pi (parent id, optional).
    `ref_len` controls the chr1 SQ length in the header (so tests can drive
    ref_coverage = (ref_end - ref_start) / ref_len).
    """
    import array
    pysam = pytest.importorskip("pysam")
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chr1", "LN": ref_len}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for r in records:
            a = pysam.AlignedSegment(bam.header)
            a.query_name = r["name"]
            a.flag = r["flag"]
            a.reference_id = 0
            a.reference_start = r["ref_start"]
            a.cigar = r["cigar"]
            a.mapping_quality = 60
            a.next_reference_id = -1
            a.next_reference_start = -1
            a.template_length = 0
            a.query_sequence = r["seq"]
            a.query_qualities = pysam.qualitystring_to_array("I" * len(r["seq"]))
            # mv:B:c (signed char array). pysam infers BAM B subtype from
            # array.array typecode -- 'b' = signed char, matches the spec.
            mv_arr = array.array("b", [r["stride"]] + [1] * r["n_moves"])
            a.set_tag("mv", mv_arr)
            a.set_tag("ts", r["ts"], value_type="i")
            a.set_tag("ns", r["ns"], value_type="i")
            a.set_tag("pt", r.get("pt", 0), value_type="i")
            if "pi" in r:
                a.set_tag("pi", r["pi"], value_type="Z")
            bam.write(a)


@pytest.fixture
def minimal_dorado_bam(tmp_path):
    """A 3-record BAM with reads fully covering a 50 nt reference, so they
    pass the default --min-ref-coverage 0.8.
      - r1: primary forward, no polyA
      - r2: reverse-strand (dropped by default -F 2324)
      - r3: primary forward + pi:Z child (dropped by default --drop-split-reads)
    """
    pytest.importorskip("pysam")
    seq = "ACGT" * 12 + "AC"   # 50 bp
    common = dict(ref_start=0, cigar=[(0, 50)], seq=seq,
                  ts=100, ns=350, stride=5, n_moves=50)
    records = [
        dict(name="r1", flag=0,  **common),
        dict(name="r2", flag=16, **common),
        dict(name="r3", flag=0,  pt=12, pi="r0", **common),
    ]
    bam = tmp_path / "dorado.sorted.bam"
    _write_minimal_dorado_bam(bam, records, ref_len=50)
    return bam


def test_dorado_extract_basic(minimal_dorado_bam, tmp_path):
    """Default behaviour (drop_dup_reads=on, drop_split_reads=on, min_ref_coverage=0.8):
    keeps only r1; r2 dropped by -F 2324 (reverse), r3 by drop_split_reads.
    CSV must have the documented 25 columns and the right field values."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pysam")
    from segshape.io.dorado_mv import extract_mv, COLS

    out = tmp_path / "extract.csv"
    rc = extract_mv(str(minimal_dorado_bam), str(out))
    assert rc == 0
    df = pd.read_csv(out, comment="#")

    assert list(df.columns) == COLS
    assert len(df) == 1
    assert set(df.read_id) == {"r1"}

    row = df.iloc[0]
    assert row["is_child"] == 0
    assert row["is_reverse"] == 0
    assert row["polya_tail"] == 0    # r1 had pt=0 → no tail
    assert row["ts"] == 100 and row["ns"] == 350
    assert row["n_moves"] == 50 and row["stride"] == 5
    assert row["ref_start"] == 0 and row["ref_end"] == 50
    assert row["ref_len"] == 50


def test_dorado_extract_keep_split_reads_via_no_flag(minimal_dorado_bam, tmp_path):
    """--no-drop-split-reads keeps the pi:Z child (r3)."""
    pd = pytest.importorskip("pandas")
    from segshape.io.dorado_mv import extract_mv

    out = tmp_path / "extract.csv"
    extract_mv(str(minimal_dorado_bam), str(out), drop_split_reads=False)
    df = pd.read_csv(out, comment="#")
    assert set(df.read_id) == {"r1", "r3"}     # r2 still dropped by -F 2324
    by_id = df.set_index("read_id")
    assert by_id.loc["r3", "is_child"] == 1
    assert by_id.loc["r3", "polya_tail"] == 12


def test_dorado_extract_reverse_raises_via_filter_flag(minimal_dorado_bam, tmp_path):
    """Lifting the 0x10 bit from --filter-flag (e.g. 2308) lets reverse-strand
    reads reach the mv-walking step. mv-walking math is undefined for those
    (pysam reverse-complements the SEQ, breaking the seq_len-1-j ↔ base
    mapping), so extract_mv must SystemExit explicitly rather than emit
    silently-wrong rows."""
    from segshape.io.dorado_mv import extract_mv

    out = tmp_path / "extract.csv"
    with pytest.raises(SystemExit, match="reverse-strand read passed"):
        extract_mv(str(minimal_dorado_bam), str(out),
                   filter_flag=2308, drop_split_reads=False)


def test_dorado_extract_min_ref_coverage_default_drops_short(tmp_path):
    """Default --min-ref-coverage 0.8 drops reads that don't cover enough of
    the reference. Build a separate BAM with reads covering only 40/200 = 20%."""
    pd = pytest.importorskip("pandas")
    from segshape.io.dorado_mv import extract_mv

    seq = "ACGT" * 10            # 40 bp
    short_records = [
        dict(name="s1", flag=0, ref_start=0, cigar=[(0, 40)], seq=seq,
             ts=100, ns=300, stride=5, n_moves=40),
    ]
    bam = tmp_path / "short.bam"
    _write_minimal_dorado_bam(bam, short_records, ref_len=200)

    # Default 0.8 → s1 has coverage 0.2 → dropped.
    out_default = tmp_path / "default.csv"
    extract_mv(str(bam), str(out_default))
    assert len(pd.read_csv(out_default, comment="#")) == 0

    # Disable coverage filter → s1 kept.
    out_disabled = tmp_path / "disabled.csv"
    extract_mv(str(bam), str(out_disabled), min_ref_coverage=0.0)
    assert set(pd.read_csv(out_disabled, comment="#").read_id) == {"s1"}


def test_dorado_extract_resolve_paths_explicit_bam(tmp_path):
    """resolve_paths: --bam alone derives default CSV next to it."""
    from segshape.io.dorado_mv import resolve_paths, DEFAULT_CSV
    import argparse
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"")  # touch
    args = argparse.Namespace(
        bam=str(bam), out=None,
        root_dir=None, dataset=None, sample=None,
    )
    bam_p, out_p = resolve_paths(args)
    assert bam_p == str(bam)
    assert out_p == str(bam.parent / DEFAULT_CSV)


def test_dorado_extract_resolve_paths_missing_args_raises(tmp_path):
    """resolve_paths: with no --bam and incomplete --root-dir/--dataset/--sample
    must SystemExit with a clear message."""
    from segshape.io.dorado_mv import resolve_paths
    import argparse
    args = argparse.Namespace(
        bam=None, out=None,
        root_dir=str(tmp_path), dataset=None, sample=None,
    )
    with pytest.raises(SystemExit, match="--bam .* or all of --root-dir"):
        resolve_paths(args)


def test_dorado_extract_provenance_header(minimal_dorado_bam, tmp_path):
    """The CSV must start with two `#` lines recording segshape + dorado
    versions and the filter parameters used. pd.read_csv(comment='#') must
    skip them and still recover the data."""
    pd = pytest.importorskip("pandas")
    from segshape.io.dorado_mv import extract_mv
    from segshape import __version__ as ssv

    out = tmp_path / "extract.csv"
    extract_mv(str(minimal_dorado_bam), str(out),
               filter_flag=2324, min_mapq=20, min_ref_coverage=0.8)

    with open(out) as fh:
        l1 = fh.readline()
        l2 = fh.readline()
    assert l1.startswith("#") and "segshape_version=" + ssv in l1
    # the test BAM has no @PG dorado line → dorado_version=unknown
    assert "dorado_version=" in l1
    assert l2.startswith("#") and "filter_flag=2324" in l2 and "min_mapq=20" in l2

    # round-trip: pandas with comment='#' yields the usable data frame
    df = pd.read_csv(out, comment="#")
    assert len(df) == 1 and set(df.read_id) == {"r1"}


# ---------------------------------------------------------------------------
# segshape.fold — SHAPE → RNAfold prediction primitives (no RNAfold binary)
# ---------------------------------------------------------------------------


def test_fold_pos_idx_mapping_uniform():
    """anchor_off=2 must map pos_idx range [0, ref_len-5] symmetrically to
    1-indexed ref positions [3, ref_len-2] (the documented dead-zone)."""
    from segshape.fold.rnafold import pos_idx_to_ref_pos
    ref_len = 100
    L = ref_len - 4   # = ref_len - k + 1 + 2*edge_pad  (uniform across chemistries)
    # last extended kmer: pos_idx = L-1 = 95 → ref_pos = 100 - 95 - 2 = 3
    assert pos_idx_to_ref_pos(L - 1, ref_len) == 3
    # first kmer: pos_idx = 0 → ref_pos = ref_len - 2 = 98
    assert pos_idx_to_ref_pos(0, ref_len) == ref_len - 2
    # range covers [3, ref_len-2] = ref_len-4 positions (matches L)
    pos_set = {pos_idx_to_ref_pos(i, ref_len) for i in range(L)}
    assert pos_set == set(range(3, ref_len - 1))


def test_fold_write_shape_dat_roundtrip(tmp_path):
    """write_shape_dat writes 1-indexed positions in 5'→3' order, with -999
    for missing positions and the configured anchor offset."""
    import numpy as np
    from segshape.fold.rnafold import write_shape_dat

    ref_len = 20
    # values per pos_idx (3'→5' axis); inject 3 finite values at i=0,1,2
    values = np.full(ref_len - 4, np.nan)
    values[0] = 0.5    # → ref_pos 18
    values[1] = 0.7    # → ref_pos 17
    values[2] = -0.3   # → ref_pos 16
    out = tmp_path / "test.shape"
    n_valid = write_shape_dat(values, ref_len=ref_len, anchor_off=2,
                              out_path=str(out))
    assert n_valid == 3
    lines = out.read_text().strip().splitlines()
    assert len(lines) == ref_len
    # positions 1..ref_len present; only 16, 17, 18 carry values
    rows = {int(line.split("\t")[0]): line.split("\t")[1] for line in lines}
    assert rows[16] == f"{-0.3:.6f}"
    assert rows[17] == f"{0.7:.6f}"
    assert rows[18] == f"{0.5:.6f}"
    # NA marker
    for p in (1, 5, 15, 19, 20):
        assert rows[p] == "-999"


def test_fold_parse_rnafold_output_extracts_dotbracket_and_energy():
    """parse_rnafold_output must skip the seq line and extract dot-brackets
    plus their parenthesized energies, in document order."""
    from segshape.fold.rnafold import parse_rnafold_output
    fake = (
        "AUGCGCAU\n"                                # sequence echo
        "....(...)    (-1.20)\n"                   # MFE
        "....{...}    [-2.34]\n"                   # ensemble (skipped: { } not . ( ))
        "....(...)    {-2.34 d=0.50}\n"            # ensemble — dot-bracket OK, no energy match
        "(((....)))    {-3.10 MEA=0.95}\n"         # MEA (energy in {} not parsed)
        ".(((..)))     ( -2.50)\n"                 # centroid
    )
    out = parse_rnafold_output(fake)
    # Expect: lines 1, 3, 4, 5 keep dot-bracket (line 2 has '{' which fails our
    # pure-dotbracket filter). Energy parsed only when a `( -X.XX )` or
    # `[ -X.XX ]` pattern is on the SAME line.
    dbs = [d for d, _ in out]
    energies = [e for _, e in out]
    assert dbs[0] == "....(...)"
    assert energies[0] == -1.20
    assert dbs[-1] == ".(((..)))"
    assert energies[-1] == -2.50
