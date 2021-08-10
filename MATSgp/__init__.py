"""
MATS Gaussian process regression
"""
from .gp_models import SwitchedGPR
from .lineshape_meanfuncs import (
    Base_Mean_Func,
    Baseline,
    ComboMeanFunc,
    Etalon,
    LineMixing,
    LineShape,
    SpectralDataInfo,
    linemix_from_dataframe,
    lineshape_from_dataframe,
)

try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("MATSgp").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"


__all__ = [
    "SpectralDataInfo",
    "LineMixing",
    "linemix_from_dataframe",
    "Base_Mean_Func",
    "LineShape",
    "lineshape_from_dataframe",
    "Etalon",
    "Baseline",
    "ComboMeanFunc",
    "SwitchedGPR",
]
