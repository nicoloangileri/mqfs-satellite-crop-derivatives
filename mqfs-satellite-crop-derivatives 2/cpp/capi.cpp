// ====================================================================
// capi.cpp — flat C ABI over the templated C++ engine
// --------------------------------------------------------------------
// Exposes the *identical* symbol and struct layout as the pure-C kernel
// (c_core/ou_mc.h), so a single Python ctypes wrapper can drive either
// implementation merely by loading a different shared object. This is
// the bridge that lets us cross-validate three independent engines
// (NumPy / C / C++) against one another from Python.
//
// Build:  g++ -O3 -fopenmp -std=c++17 -shared -fPIC capi.cpp -o libmqfs_cpp.so
//
// (c) Nicolo Angileri / MQFS — MIT Licence.
// ====================================================================
#include "mqfs_pricer.hpp"

// Single source of truth for the C ABI: reuse the exact struct + enum
// definitions the C kernel publishes, guaranteeing binary compatibility.
#include "../c_core/ou_mc.h"

#include <vector>

extern "C" {

int mqfs_price_ou(const double *seasonal, int M,
                  double kappa, double sigma, double dt, double x0,
                  double floor, double cap,
                  long n_paths, int antithetic, unsigned long long seed,
                  const mqfs_contract *contract,
                  double risk_free, double horizon, double sd_theta,
                  mqfs_result *out)
{
    if (!seasonal || !contract || !out || M <= 0) return 1;

    std::vector<double> s(seasonal, seasonal + M);

    mqfs::Contract c;
    c.strike     = contract->strike;
    c.tick       = contract->tick;
    c.limit      = contract->limit;
    c.index_type = static_cast<mqfs::IndexType>(contract->index_type);
    c.style      = static_cast<mqfs::Style>(contract->style);

    mqfs::Market m;
    m.risk_free = risk_free;
    m.horizon   = horizon;
    m.sd_theta  = sd_theta;

    mqfs::MCConfig cfg;
    cfg.n_paths    = n_paths;
    cfg.antithetic = antithetic != 0;
    cfg.seed       = seed;
    cfg.ndvi_floor = floor;
    cfg.ndvi_cap   = cap;

    mqfs::Result r = mqfs::price_ou<mqfs::DefaultPayoff>(s, kappa, sigma, dt, x0, c, m, cfg);

    out->fair_value                   = r.fair_value;
    out->sd_premium                   = r.sd_premium;
    out->expected_payout              = r.expected_payout;
    out->payout_std                   = r.payout_std;
    out->trigger_probability          = r.trigger_probability;
    out->expected_payout_if_triggered = r.expected_payout_if_triggered;
    out->var_95                       = r.var_95;
    out->cvar_95                      = r.cvar_95;
    out->n_paths                      = r.n_paths;
    return 0;
}

} // extern "C"
