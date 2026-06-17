// ====================================================================
// cli.cpp — standalone demo of the header-only C++ pricing engine
// --------------------------------------------------------------------
// Builds a synthetic in-season durum-wheat NDVI climatology, prices a
// capped put on seasonal-mean NDVI, and prints a full risk sheet. Mirrors
// c_core/cli_demo.c so the two native engines can be eyeballed side by side.
//
// Build:  g++ -O3 -fopenmp -std=c++17 cli.cpp -o cli_demo
//
// (c) Nicolo Angileri / MQFS — MIT Licence.
// ====================================================================
#include "mqfs_pricer.hpp"
#include <cstdio>
#include <vector>

int main() {
    // Eight dekads spanning the grain-filling window (DOY 75..151), a
    // plausible Mediterranean winter-cereal greenness arc peaking mid-season.
    std::vector<double> seasonal = {0.55, 0.62, 0.68, 0.71, 0.70, 0.64, 0.55, 0.45};

    mqfs::Contract contract;
    contract.strike     = 0.58;          // ~20th-pct seasonal mean
    contract.tick       = 250000.0;      // EUR per 1.00 NDVI below strike
    contract.limit      = 50000.0;       // EUR cap
    contract.index_type = mqfs::IndexType::Mean;
    contract.style      = mqfs::Style::Put;

    mqfs::Market market;                 // r=3%, T=0.5y, theta=0.25
    mqfs::MCConfig cfg;                  // 200k paths, antithetic, fixed seed

    const double kappa = 6.0, sigma = 0.12, x0 = 0.0;
    const double dt = 10.0 / 365.0;

    mqfs::Result r = mqfs::price_ou<mqfs::DefaultPayoff>(
        seasonal, kappa, sigma, dt, x0, contract, market, cfg);

    std::printf("MQFS C++ engine — parametric drought put (seasonal-mean NDVI)\n");
    std::printf("  paths                 : %ld%s\n", r.n_paths,
                cfg.antithetic ? " (antithetic)" : "");
    std::printf("  fair (burn) value     : EUR %12.2f\n", r.fair_value);
    std::printf("  SD-principle premium  : EUR %12.2f\n", r.sd_premium);
    std::printf("  expected payout       : EUR %12.2f\n", r.expected_payout);
    std::printf("  payout std            : EUR %12.2f\n", r.payout_std);
    std::printf("  trigger probability   : %12.2f %%\n", 100.0 * r.trigger_probability);
    std::printf("  E[payout | triggered] : EUR %12.2f\n", r.expected_payout_if_triggered);
    std::printf("  writer VaR  95%%       : EUR %12.2f\n", r.var_95);
    std::printf("  writer CVaR 95%%       : EUR %12.2f\n", r.cvar_95);
    return 0;
}
