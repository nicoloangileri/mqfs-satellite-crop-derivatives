"""
bindings.cpp_pricer_wrapper
===========================
Drop-in pricer backed by the OpenMP-parallel C++ engine
(``cpp/libmqfs_cpp.so``, built from ``cpp/capi.cpp``).

Two ways in:

* **ctypes ABI** (default, zero extra dependencies) — :func:`price_contract_cpp`,
  signature-compatible with :func:`pricing_engine.pricer.price_contract`.
* **pybind11 module** ``mqfs_cpp`` — used automatically if importable; otherwise
  the ctypes path is taken transparently.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pricing_engine.payoff import Contract
from pricing_engine.pricer import MarketParams, PricingResult
from pricing_engine.monte_carlo import MCConfig

from ._native import load_library, price_native, CPP_LIB_PATH


@lru_cache(maxsize=4)
def _lib(path: str = CPP_LIB_PATH):
    return load_library(path)


def price_contract_cpp(seasonal, kappa: float, sigma: float, dt: float,
                       contract: Contract, market: MarketParams,
                       x0: float = 0.0, mc: Optional[MCConfig] = None,
                       lib_path: str = CPP_LIB_PATH) -> PricingResult:
    """Price one parametric drought contract with the C++ engine (ctypes ABI)."""
    return price_native(_lib(lib_path), seasonal, kappa, sigma, dt,
                        contract, market, x0=x0, mc=mc)


def has_pybind_module() -> bool:
    """True if the optional native pybind11 module ``mqfs_cpp`` is importable."""
    try:
        import mqfs_cpp  # noqa: F401
        return True
    except Exception:
        return False
