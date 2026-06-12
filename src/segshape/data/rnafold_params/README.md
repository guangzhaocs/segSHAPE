# Bundled ViennaRNA `-P` parameter files

Loaded via `segshape.data.rnafold_par_path("andronescu2007")` and used as the
default for `segshape fold --par-path`.

## `rna_andronescu2007.par`

- **Source**: Andronescu et al. (2007) "Efficient parameter estimation for
  RNA secondary structure prediction." *Bioinformatics* 23(13):i19-i28.
- **Format**: ViennaRNA RNAfold parameter file v2.0 (used with `RNAfold -P`).
- **Why bundled**: matches the legacy `code/eval_full_pipeline.py` baseline
  so MCC/F1 numbers stay comparable. Verified to give different MFE than
  ViennaRNA's built-in Turner2004 on miR17-92 (≈-253 vs ≈-282 kcal/mol).
- **Pinned copy**: identical to
  `/scratch/cs/nanopore/chengg1/segSHAPE/meta/rna_andronescu2007.par`
  (the canonical project copy).
