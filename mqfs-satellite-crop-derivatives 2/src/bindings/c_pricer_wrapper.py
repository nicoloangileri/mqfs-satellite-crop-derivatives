"""
bindings.c_pricer_wrapper
=========================
Drop-in pricer backed by the pure-C audit kernel (``c_core/libmqfs_ou_mc.so``).

Usage
-----
>>> from bindings.c_pricer_wrapper import price_contract_c
>>> res = price_contract_c(seasonal, kappa, sigma, dt, contract, market)
>>> res.fair_value

The call signature matches :func:`pricing_engine.pricer.price_contract`, so the
C engine can be swapped in anywhere the NumPy reference is used.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pricing_engine.payoff import Contract
from pricing_engine.pricer import MarketParams, PricingResult
from pricing_engine.monte_carlo import MCConfig

from ._native import load_library, price_native, C_LIB_PATH


@lru_cache(maxsize=4)
def _lib(path: str = C_LIB_PATH):
    return load_library(path)


def price_contract_c(seasonal, kappa: float, sigma: float, dt: float,
                     contract: Contract, market: MarketParams,
                     x0: float = 0.0, mc: Optional[MCConfig] = None,
                     lib_path: str = C_LIB_PATH) -> PricingResult:
    """Price one parametric drought contract with the C kernel."""
    return price_native(_lib(lib_path), seasonal, kappa, sigma, dt,
                        contract, market, x0=x0, mc=mc)
