"""SHAPE-constrained secondary structure prediction (Step 7 of segshape).

Consumes ``mod_rate.csv`` from ``segshape mod-calling`` and emits a predicted
dot-bracket structure via ViennaRNA's RNAfold with the Deigan SHAPE
pseudo-energy method.
"""
