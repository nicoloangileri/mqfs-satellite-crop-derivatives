#!/usr/bin/env python3
"""
run_pipeline.py
===============
End-to-end driver for the MQFS satellite-indexed crop-derivative pipeline.

Stages
------
1. Oracle      — synthesise (or, with credentials, ingest) a dekadal NDVI panel
                 for the Sicilian zones in ``config/config.yaml``.
2. Quant layer — smooth, fit the harmonic climatology, de-seasonalise, and
                 build the seasonal settlement index.
3. Pricing     — calibrate the OU anomaly, set the strike at the configured
                 historical percentile, and price the parametric drought put
                 with the NumPy reference *and* the C and C++ engines.
4. Outputs     — write publication figures to ``figures/`` and LaTeX-ready
                 numbers/tables to ``paper/generated/`` so the working paper
                 reflects real, reproducible results.

Runs fully offline on the synthetic generator (no Earth-observation credentials
needed), which is what makes the paper build deterministic and the CI green.

    python scripts/run_pipeline.py            # uses config/config.yaml
    python scripts/run_pipeline.py --paths 400000

(c) Nicolo Angileri / MQFS — MIT Licence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "src"))

from oracle.synthetic import generate_zone                       # noqa: E402
from quant_layer.feature_engineering import build_features        # noqa: E402
from quant_layer.anomaly_detection import seasonal_index, detect_episodes  # noqa: E402
from pricing_engine.ou_calibration import calibrate_ou            # noqa: E402
from pricing_engine.payoff import Contract, strike_from_quantile  # noqa: E402
from pricing_engine.pricer import MarketParams, price_contract    # noqa: E402
from pricing_engine.monte_carlo import MCConfig, simulate_paths, seasonal_grid  # noqa: E402

# --------------------------------------------------------------------------- #
# Plot style — sober, publication-grade
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "legend.frameon": False,
})
BLUE, NAVY, GREEN, RED, GREY = "#2f6db0", "#1b3a66", "#2e8b57", "#c0392b", "#9aa0a6"


def load_config():
    path = os.path.join(_REPO, "config", "config.yaml")
    try:
        import yaml
        with open(path) as fh:
            return yaml.safe_load(fh)
    except Exception:
        return {}


def cfg_get(cfg, path, default):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main():
    cfg = load_config()

    ap = argparse.ArgumentParser(description="MQFS satellite crop-derivative pipeline")
    ap.add_argument("--paths", type=int,
                    default=int(cfg_get(cfg, "pricing.monte_carlo.n_paths", 200_000)))
    ap.add_argument("--seed", type=int,
                    default=int(cfg_get(cfg, "pricing.monte_carlo.seed", 20260506)))
    ap.add_argument("--zone", default="SIC-CL")
    args = ap.parse_args()

    figdir = os.path.join(_REPO, "figures")
    gendir = os.path.join(_REPO, "paper", "generated")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(gendir, exist_ok=True)

    # --- risk window + model + contract params from config -----------------
    start_doy = int(cfg_get(cfg, "pricing.risk_window.start_doy", 75))
    end_doy = int(cfg_get(cfg, "pricing.risk_window.end_doy", 151))
    step_days = int(cfg_get(cfg, "pricing.risk_window.step_days", 10))
    dt = step_days / 365.0
    floor = float(cfg_get(cfg, "pricing.model.ndvi_floor", -0.20))
    cap = float(cfg_get(cfg, "pricing.model.ndvi_cap", 0.95))
    q_strike = float(cfg_get(cfg, "pricing.contract.strike_quantile", 0.20))
    tick = float(cfg_get(cfg, "pricing.contract.tick_eur", 250_000.0))
    limit = float(cfg_get(cfg, "pricing.contract.limit_eur", 50_000.0))
    index_kind = str(cfg_get(cfg, "pricing.contract.index", "mean"))

    market = MarketParams(
        risk_free=float(cfg_get(cfg, "pricing.market.risk_free", 0.03)),
        horizon_years=float(cfg_get(cfg, "pricing.market.horizon_years", 0.5)),
        sd_loading_theta=float(cfg_get(cfg, "pricing.market.sd_loading_theta", 0.25)),
        wang_lambda=float(cfg_get(cfg, "pricing.market.wang_lambda", 0.15)),
    )
    mc = MCConfig(n_paths=args.paths, antithetic=True, seed=args.seed,
                  ndvi_floor=floor, ndvi_cap=cap)

    print("=" * 64)
    print("MQFS satellite crop-derivative pipeline")
    print("=" * 64)

    # === 1-2. Oracle + Quant layer =========================================
    print(f"[1] Oracle: synthesising dekadal NDVI for zone {args.zone} ...")
    df = generate_zone(seed=7)
    print(f"    {len(df)} dekadal observations, "
          f"{int(df['NDVI'].isna().sum())} cloud gaps")

    print("[2] Quant layer: smoothing, climatology, anomaly, seasonal index ...")
    feat = build_features(df, ndvi_col="NDVI")
    clim = feat.attrs["climatology"]
    seas_idx = seasonal_index(feat, start_doy, end_doy, how=index_kind)
    episodes = detect_episodes(feat, max_gap=1)
    print(f"    seasons: {len(seas_idx)} | drought episodes: {len(episodes)}")

    # === 3. Calibration + pricing ==========================================
    print("[3] Pricing: OU calibration + Monte Carlo (NumPy / C / C++) ...")
    ou = calibrate_ou(feat["anomaly_raw"].to_numpy(), dt)
    print(f"    OU: kappa={ou.kappa:.3f}  sigma={ou.sigma:.3f}  "
          f"half-life={ou.half_life_years:.3f} yr  x0={ou.x0:.3f}")

    seasonal = seasonal_grid(clim, start_doy, end_doy, step_days)
    strike = strike_from_quantile(seas_idx.to_numpy(), q=q_strike)
    contract = Contract(strike=strike, tick=tick, limit=limit,
                        index=index_kind, style="put", dt=dt)
    print(f"    strike (q={q_strike:.2f}) = {strike:.4f}  | M={len(seasonal)} dekads")

    res = price_contract(seasonal, ou.kappa, ou.sigma, dt, contract, market,
                         x0=0.0, mc=mc)
    print("\n" + res.pretty())

    # --- three-engine cross-validation -------------------------------------
    engines = {"NumPy": res}
    try:
        from bindings.c_pricer_wrapper import price_contract_c
        engines["C"] = price_contract_c(seasonal, ou.kappa, ou.sigma, dt,
                                        contract, market, x0=0.0, mc=mc)
    except Exception as e:
        print(f"    [warn] C engine unavailable: {e}")
    try:
        from bindings.cpp_pricer_wrapper import price_contract_cpp
        engines["C++"] = price_contract_cpp(seasonal, ou.kappa, ou.sigma, dt,
                                            contract, market, x0=0.0, mc=mc)
    except Exception as e:
        print(f"    [warn] C++ engine unavailable: {e}")

    disc = np.exp(-market.risk_free * market.horizon_years)
    se_fair = disc * res.payout_std / np.sqrt(mc.n_paths)

    # === 4. Figures ========================================================
    print("\n[4] Writing figures ...")
    _fig_climatology(df, feat, figdir)
    _fig_anomaly(feat, episodes, figdir)
    _fig_fan_chart(seasonal, ou, dt, mc, strike, start_doy, end_doy, step_days, figdir)
    _fig_payoff(seasonal, ou, dt, contract, mc, res, figdir)
    strikes, curves, trig = _fig_premium_vs_strike(
        seasonal, ou, dt, contract, market, seas_idx, figdir)
    _fig_validation(engines, se_fair, figdir)
    _fig_seasonal_index(seas_idx, strike, figdir)

    # === 5. Generated LaTeX + JSON =========================================
    print("[5] Writing generated tables for the paper ...")
    _write_generated(gendir, res, engines, ou, contract, market, mc,
                     seas_idx, episodes, se_fair, start_doy, end_doy, len(seasonal))

    print("\nDone. Figures -> figures/   Generated tables -> paper/generated/")


# --------------------------------------------------------------------------- #
# Figure builders
# --------------------------------------------------------------------------- #
def _fig_climatology(df, feat, figdir):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(df["date"], df["NDVI"], ".", color=GREY, ms=4, label="Observed dekadal NDVI")
    ax.plot(feat.index, feat["ndvi_smooth"], "-", color=BLUE, lw=1.0, alpha=0.8,
            label="Whittaker-smoothed")
    ax.plot(feat.index, feat["seasonal"], "-", color=NAVY, lw=1.8, label="Harmonic climatology")
    ax.set_ylabel("NDVI")
    ax.set_title("Durum-wheat NDVI: observations, smoothing and fitted climatology")
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_ndvi_climatology.png"))
    plt.close(fig)


def _fig_anomaly(feat, episodes, figdir):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(feat.index, feat["anomaly_z"], "-", color=BLUE, lw=1.0)
    ax.axhline(-1.5, ls="--", color=RED, lw=1.0, label="Shock trigger (z = -1.5)")
    ax.axhline(0.0, ls="-", color="black", lw=0.6, alpha=0.4)
    for e in episodes:
        ax.axvspan(e.start, e.end, color=RED, alpha=0.12)
    ax.set_ylabel("Standardised anomaly (z)")
    ax.set_title("De-seasonalised NDVI anomaly with detected drought episodes")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_anomaly_episodes.png"))
    plt.close(fig)


def _fig_fan_chart(seasonal, ou, dt, mc, strike, start_doy, end_doy, step_days, figdir):
    paths = simulate_paths(seasonal, ou.kappa, ou.sigma, dt,
                           cfg=MCConfig(n_paths=min(mc.n_paths, 40_000), seed=mc.seed,
                                        ndvi_floor=mc.ndvi_floor, ndvi_cap=mc.ndvi_cap))
    qs = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    doy = np.arange(start_doy, end_doy + 1, step_days)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.fill_between(doy, qs[0], qs[4], color=BLUE, alpha=0.15, label="90% band")
    ax.fill_between(doy, qs[1], qs[3], color=BLUE, alpha=0.30, label="50% band")
    ax.plot(doy, qs[2], "-", color=NAVY, lw=1.8, label="Median path")
    ax.axhline(strike, ls="--", color=RED, lw=1.3, label=f"Strike K = {strike:.3f}")
    ax.set_xlabel("Day of year"); ax.set_ylabel("NDVI")
    ax.set_title("Simulated in-season NDVI under the OU model")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_fan_chart.png"))
    plt.close(fig)


def _fig_payoff(seasonal, ou, dt, contract, mc, res, figdir):
    from pricing_engine.payoff import aggregate_index, payoff
    paths = simulate_paths(seasonal, ou.kappa, ou.sigma, dt, cfg=mc)
    idx = aggregate_index(paths, how=contract.index, dt=dt)
    pay = payoff(idx, contract)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    trig = pay[pay > 0]
    ax.hist(trig, bins=50, color=GREEN, alpha=0.85)
    ax.axvline(res.fair_value, color=NAVY, lw=1.6, label=f"Fair value {res.fair_value:,.0f}")
    ax.axvline(res.sd_premium, color=RED, lw=1.6, ls="--",
               label=f"SD premium {res.sd_premium:,.0f}")
    ax.set_xlabel("Payout (EUR), triggered seasons"); ax.set_ylabel("Frequency")
    ax.set_title(f"Payoff distribution  (trigger probability {100*res.trigger_probability:.1f}%)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_payoff_distribution.png"))
    plt.close(fig)


def _fig_premium_vs_strike(seasonal, ou, dt, contract, market, seas_idx, figdir):
    lo, hi = np.quantile(seas_idx.to_numpy(), [0.05, 0.55])
    strikes = np.linspace(lo, hi, 18)
    fair, sd, wang, trig = [], [], [], []
    mc = MCConfig(n_paths=120_000, seed=20260506,
                  ndvi_floor=-0.20, ndvi_cap=0.95)
    for k in strikes:
        c = Contract(strike=float(k), tick=contract.tick, limit=contract.limit,
                     index=contract.index, style="put", dt=dt)
        r = price_contract(seasonal, ou.kappa, ou.sigma, dt, c, market, mc=mc)
        fair.append(r.fair_value); sd.append(r.sd_premium)
        wang.append(r.wang_premium); trig.append(r.trigger_probability)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(strikes, fair, "-o", color=NAVY, ms=3, lw=1.4, label="Fair (burn) value")
    ax.plot(strikes, sd, "-s", color=RED, ms=3, lw=1.4, label="SD-principle premium")
    ax.plot(strikes, wang, "-^", color=GREEN, ms=3, lw=1.4, label="Wang-transform premium")
    ax.axvline(contract.strike, ls=":", color=GREY, lw=1.2)
    ax.set_xlabel("Strike K (seasonal NDVI index)")
    ax.set_ylabel("Premium (EUR)")
    ax.set_title("Premium term structure across strikes")
    ax.legend(loc="upper left", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(strikes, 100 * np.array(trig), color=BLUE, lw=1.0, alpha=0.5)
    ax2.set_ylabel("Trigger probability (%)", color=BLUE)
    ax2.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_premium_vs_strike.png"))
    plt.close(fig)
    return strikes, (fair, sd, wang), trig


def _fig_validation(engines, se_fair, figdir):
    names = list(engines.keys())
    vals = [engines[n].fair_value for n in names]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    colors = [NAVY, BLUE, GREEN][:len(names)]
    ax.bar(names, vals, color=colors, alpha=0.85, width=0.55)
    ref = engines["NumPy"].fair_value
    ax.errorbar(names, vals, yerr=4 * se_fair, fmt="none", ecolor="black",
                elinewidth=1.0, capsize=5, label="±4 MC standard errors")
    ax.axhline(ref, ls="--", color=GREY, lw=1.0)
    ax.set_ylabel("Fair value (EUR)")
    ax.set_title("Engine cross-validation: fair value by implementation")
    ax.legend(fontsize=9)
    lo = min(vals) - 8 * se_fair
    hi = max(vals) + 8 * se_fair
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_engine_validation.png"))
    plt.close(fig)


def _fig_seasonal_index(seas_idx, strike, figdir):
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    years = seas_idx.index.astype(int)
    colors = [RED if v < strike else BLUE for v in seas_idx.to_numpy()]
    ax.bar(years, seas_idx.to_numpy(), color=colors, alpha=0.85)
    ax.axhline(strike, ls="--", color="black", lw=1.2, label=f"Strike K = {strike:.3f}")
    ax.set_xlabel("Season (year)"); ax.set_ylabel("Seasonal NDVI index")
    ax.set_title("Historical seasonal settlement index (red = below strike)")
    ax.set_ylim(0.9 * seas_idx.min(), 1.02 * seas_idx.max())
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_seasonal_index.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Generated LaTeX + JSON
# --------------------------------------------------------------------------- #
def _eur(x):
    return f"{x:,.2f}".replace(",", r"\,")


def _write_generated(gendir, res, engines, ou, contract, market, mc,
                     seas_idx, episodes, se_fair, start_doy, end_doy, M):
    # 1) headline macros
    macros = [
        r"% Auto-generated by scripts/run_pipeline.py — do not edit by hand.",
        r"\newcommand{\resFair}{%s}" % _eur(res.fair_value),
        r"\newcommand{\resSD}{%s}" % _eur(res.sd_premium),
        r"\newcommand{\resWang}{%s}" % _eur(res.wang_premium),
        r"\newcommand{\resTrigger}{%.1f}" % (100 * res.trigger_probability),
        r"\newcommand{\resCVaR}{%s}" % _eur(res.cvar_95),
        r"\newcommand{\resVaR}{%s}" % _eur(res.var_95),
        r"\newcommand{\resEIfTrig}{%s}" % _eur(res.expected_payout_if_triggered),
        r"\newcommand{\ouKappa}{%.2f}" % ou.kappa,
        r"\newcommand{\ouSigma}{%.3f}" % ou.sigma,
        r"\newcommand{\ouHalfLife}{%.2f}" % ou.half_life_years,
        r"\newcommand{\contractStrike}{%.3f}" % contract.strike,
        r"\newcommand{\contractTick}{%s}" % _eur(contract.tick),
        r"\newcommand{\contractLimit}{%s}" % _eur(contract.limit),
        r"\newcommand{\mcPaths}{%s}" % f"{mc.n_paths:,}".replace(",", r"\,"),
        r"\newcommand{\nSeasons}{%d}" % len(seas_idx),
        r"\newcommand{\nEpisodes}{%d}" % len(episodes),
        r"\newcommand{\riskWindow}{%d--%d}" % (start_doy, end_doy),
        r"\newcommand{\nDekads}{%d}" % M,
        r"\newcommand{\mktR}{%.1f}" % (100 * market.risk_free),
        r"\newcommand{\mktTheta}{%.2f}" % market.sd_loading_theta,
        r"\newcommand{\mktLambda}{%.2f}" % market.wang_lambda,
    ]
    with open(os.path.join(gendir, "macros.tex"), "w") as fh:
        fh.write("\n".join(macros) + "\n")

    # 2) pricing sheet table
    rows = [
        ("Fair (burn) value", _eur(res.fair_value)),
        ("SD-principle premium", _eur(res.sd_premium)),
        ("Wang-transform premium", _eur(res.wang_premium)),
        ("Expected payout (undiscounted)", _eur(res.expected_payout)),
        ("Trigger probability", f"{100*res.trigger_probability:.1f}\\%"),
        (r"$\mathbb{E}[\text{payout}\mid\text{triggered}]$", _eur(res.expected_payout_if_triggered)),
        ("Writer VaR 95\\%", _eur(res.var_95)),
        ("Writer CVaR 95\\%", _eur(res.cvar_95)),
    ]
    tab = [r"\begin{tabular}{lr}", r"\toprule",
           r"Quantity & Value (EUR) \\", r"\midrule"]
    tab += [f"{k} & {v} \\\\" for k, v in rows]
    tab += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(gendir, "pricing_table.tex"), "w") as fh:
        fh.write("\n".join(tab) + "\n")

    # 3) engine validation table
    vt = [r"\begin{tabular}{lrrrr}", r"\toprule",
          r"Engine & Fair value & SD premium & Trigger (\%) & CVaR 95\% \\",
          r"\midrule"]
    for name, r in engines.items():
        vt.append(f"{name} & {_eur(r.fair_value)} & {_eur(r.sd_premium)} & "
                  f"{100*r.trigger_probability:.2f} & {_eur(r.cvar_95)} \\\\")
    vt += [r"\midrule",
           r"MC standard error & \multicolumn{4}{l}{$%s$ on the fair value "
           r"(discounted mean payout)} \\" % _eur(se_fair),
           r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(gendir, "validation_table.tex"), "w") as fh:
        fh.write("\n".join(vt) + "\n")

    # 4) machine-readable JSON
    payload = {
        "result": res.to_dict(),
        "engines": {k: v.to_dict() for k, v in engines.items()},
        "ou": {"kappa": ou.kappa, "sigma": ou.sigma,
               "half_life_years": ou.half_life_years, "x0": ou.x0},
        "contract": {"strike": contract.strike, "tick": contract.tick,
                     "limit": contract.limit, "index": contract.index},
        "mc_standard_error_fair": se_fair,
        "n_seasons": int(len(seas_idx)),
        "n_episodes": int(len(episodes)),
    }
    with open(os.path.join(gendir, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()
