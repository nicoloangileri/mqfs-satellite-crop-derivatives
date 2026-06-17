"""
bindings._native
================
Shared ctypes plumbing for the native Monte Carlo engines. Both the pure-C
kernel (``c_core/libmqfs_ou_mc.so``) and the C++ engine (``cpp/libmqfs_cpp.so``)
export the *same* symbol ``mqfs_price_ou`` with the *same* struct layout, so a
single binding drives either one — you only change which shared object is
loaded. This is what makes the three-way (NumPy / C / C++) cross-validation
in :mod:`tests.test_pricing` a one-liner.

The native engines return aggregated risk statistics directly (they do not
materialise the full path matrix), so the Wang-transform premium and the
Bachelier analytic check — both cheap, both Python-side — are reported as NaN
here and filled in by the NumPy pricer when a full sheet is wanted.
"""
from __future__ import annotations

import ctypes as C
import math
import os
from typing import Optional

from pricing_engine.payoff import Contract
from pricing_engine.pricer import MarketParams, PricingResult
from pricing_engine.monte_carlo import MCConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

# Default locations of the two compiled engines, relative to the repo root.
C_LIB_PATH = os.path.join(_REPO, "c_core", "libmqfs_ou_mc.so")
CPP_LIB_PATH = os.path.join(_REPO, "cpp", "libmqfs_cpp.so")

_INDEX_CODE = {"mean": 0, "integral": 1, "min": 2}
_STYLE_CODE = {"put": 0, "digital": 1}


class MqfsContract(C.Structure):
    _fields_ = [
        ("strike", C.c_double),
        ("tick", C.c_double),
        ("limit", C.c_double),
        ("index_type", C.c_int),
        ("style", C.c_int),
    ]


class MqfsResult(C.Structure):
    _fields_ = [
        ("fair_value", C.c_double),
        ("sd_premium", C.c_double),
        ("expected_payout", C.c_double),
        ("payout_std", C.c_double),
        ("trigger_probability", C.c_double),
        ("expected_payout_if_triggered", C.c_double),
        ("var_95", C.c_double),
        ("cvar_95", C.c_double),
        ("n_paths", C.c_long),
    ]


def load_library(path: str) -> C.CDLL:
    """Load a native engine and declare the ``mqfs_price_ou`` prototype."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"native engine not found: {path}\n"
            f"  build the C engine with:  make -C {os.path.join(_REPO, 'c_core')}\n"
            f"  build the C++ engine with: "
            f"g++ -O3 -fopenmp -std=c++17 -shared -fPIC "
            f"{os.path.join(_REPO, 'cpp', 'capi.cpp')} -o {CPP_LIB_PATH}"
        )
    lib = C.CDLL(path)
    lib.mqfs_price_ou.restype = C.c_int
    lib.mqfs_price_ou.argtypes = [
        C.POINTER(C.c_double), C.c_int,          # seasonal, M
        C.c_double, C.c_double, C.c_double, C.c_double,  # kappa, sigma, dt, x0
        C.c_double, C.c_double,                  # floor, cap
        C.c_long, C.c_int, C.c_uint64,           # n_paths, antithetic, seed
        C.POINTER(MqfsContract),                 # contract
        C.c_double, C.c_double, C.c_double,      # risk_free, horizon, sd_theta
        C.POINTER(MqfsResult),                   # out
    ]
    return lib


def price_native(lib: C.CDLL,
                 seasonal, kappa: float, sigma: float, dt: float,
                 contract: Contract, market: MarketParams,
                 x0: float = 0.0, mc: Optional[MCConfig] = None) -> PricingResult:
    """Price one contract with a native engine, returning a :class:`PricingResult`.

    Signature deliberately mirrors :func:`pricing_engine.pricer.price_contract`
    so the wrappers are drop-in replacements in any pipeline.
    """
    mc = mc or MCConfig()
    arr = (C.c_double * len(seasonal))(*[float(s) for s in seasonal])

    cc = MqfsContract(
        strike=float(contract.strike),
        tick=float(contract.tick),
        limit=float(contract.limit),
        index_type=_INDEX_CODE[contract.index],
        style=_STYLE_CODE[contract.style],
    )
    out = MqfsResult()

    rc = lib.mqfs_price_ou(
        arr, C.c_int(len(seasonal)),
        C.c_double(kappa), C.c_double(sigma), C.c_double(dt), C.c_double(x0),
        C.c_double(mc.ndvi_floor), C.c_double(mc.ndvi_cap),
        C.c_long(int(mc.n_paths)), C.c_int(1 if mc.antithetic else 0),
        C.c_uint64(int(mc.seed)),
        C.byref(cc),
        C.c_double(market.risk_free), C.c_double(market.horizon_years),
        C.c_double(market.sd_loading_theta),
        C.byref(out),
    )
    if rc != 0:
        raise RuntimeError(f"native mqfs_price_ou failed with code {rc}")

    sd_prem = out.sd_premium
    return PricingResult(
        fair_value=out.fair_value,
        sd_premium=sd_prem,
        wang_premium=math.nan,        # native engines do not compute the distortion
        expected_payout=out.expected_payout,
        trigger_probability=out.trigger_probability,
        expected_payout_if_triggered=out.expected_payout_if_triggered,
        payout_std=out.payout_std,
        var_95=out.var_95,
        cvar_95=out.cvar_95,
        loss_ratio_at_sd=(out.expected_payout / sd_prem) if sd_prem > 0 else float("nan"),
        n_paths=int(out.n_paths),
        bachelier_check=math.nan,
    )
