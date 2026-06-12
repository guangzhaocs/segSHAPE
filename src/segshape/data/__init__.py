"""Bundled data files (ONT kmer models, etc.).

Use ``importlib.resources.files("segshape.data")`` to access bundled resources
in a way that works both for installed wheels and editable installs.
"""

from importlib.resources import files


def kmer_model_path(name: str) -> str:
    """Return absolute path to a bundled kmer model under ``data/kmer_models/``."""
    return str(files("segshape.data").joinpath("kmer_models", name))


def rnafold_par_path(name: str) -> str:
    """Return absolute path to a bundled ViennaRNA -P param file under
    ``data/rnafold_params/``. Pass either the bare name (``andronescu2007``)
    or the full filename (``rna_andronescu2007.par``)."""
    if not name.endswith(".par"):
        name = f"rna_{name}.par"
    return str(files("segshape.data").joinpath("rnafold_params", name))
