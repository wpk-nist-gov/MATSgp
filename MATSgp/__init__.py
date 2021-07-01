"""
MATS Gaussian process regression
"""


try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("MATSgp").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"
