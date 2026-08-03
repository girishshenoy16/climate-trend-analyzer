"""Shared test configuration and fixtures."""

import warnings

# Suppress all known warnings from dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", message=".*set_bad.*")
warnings.filterwarnings("ignore", message=".*cmap.with_extremes.*")

try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*Optimization failed to converge.*")
except ImportError:
    pass
