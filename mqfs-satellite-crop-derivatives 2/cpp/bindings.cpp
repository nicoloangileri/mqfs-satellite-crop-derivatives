// ====================================================================
// bindings.cpp — pybind11 module `mqfs_cpp`
// --------------------------------------------------------------------
// A first-class native Python extension over the header-only engine, for
// users who would rather `import mqfs_cpp` than drive the ctypes ABI. The
// ctypes path (capi.cpp) needs no build step beyond g++, so this module is
// optional; it is built by CMakeLists.txt when pybind11 is available.
//
//   pip install pybind11
//   c++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC \
//       $(python3 -m pybind11 --includes) bindings.cpp \
//       -o mqfs_cpp$(python3-config --extension-suffix)
//
// (c) Nicolo Angileri / MQFS — MIT Licence.
// ====================================================================
#include "mqfs_pricer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

// Price one contract; returns a plain dict mirroring PricingResult fields the
// native engine computes (Wang premium / Bachelier check stay Python-side).
static py::dict price(std::vector<double> seasonal,
                      double kappa, double sigma, double dt, double x0,
                      double strike, double tick, double limit,
                      int index_type, int style,
                      double risk_free, double horizon, double sd_theta,
                      long n_paths, bool antithetic, std::uint64_t seed,
                      double ndvi_floor, double ndvi_cap) {
    mqfs::Contract c{strike, tick, limit,
                     static_cast<mqfs::IndexType>(index_type),
                     static_cast<mqfs::Style>(style)};
    mqfs::Market m{risk_free, horizon, sd_theta};
    mqfs::MCConfig cfg{n_paths, antithetic, seed, ndvi_floor, ndvi_cap};

    mqfs::Result r;
    {
        py::gil_scoped_release release;   // let OpenMP threads run lock-free
        r = mqfs::price_ou<mqfs::DefaultPayoff>(seasonal, kappa, sigma, dt, x0, c, m, cfg);
    }

    py::dict out;
    out["fair_value"]                   = r.fair_value;
    out["sd_premium"]                   = r.sd_premium;
    out["expected_payout"]              = r.expected_payout;
    out["payout_std"]                   = r.payout_std;
    out["trigger_probability"]          = r.trigger_probability;
    out["expected_payout_if_triggered"] = r.expected_payout_if_triggered;
    out["var_95"]                       = r.var_95;
    out["cvar_95"]                      = r.cvar_95;
    out["n_paths"]                      = r.n_paths;
    return out;
}

PYBIND11_MODULE(mqfs_cpp, m) {
    m.doc() = "MQFS C++ Monte Carlo pricer for satellite-indexed crop derivatives";
    m.def("price", &price,
          py::arg("seasonal"), py::arg("kappa"), py::arg("sigma"),
          py::arg("dt"), py::arg("x0") = 0.0,
          py::arg("strike"), py::arg("tick"), py::arg("limit"),
          py::arg("index_type") = 0, py::arg("style") = 0,
          py::arg("risk_free") = 0.03, py::arg("horizon") = 0.50,
          py::arg("sd_theta") = 0.25,
          py::arg("n_paths") = 200000, py::arg("antithetic") = true,
          py::arg("seed") = 20260506ULL,
          py::arg("ndvi_floor") = -0.20, py::arg("ndvi_cap") = 0.95,
          "Price one parametric drought contract by Monte Carlo.");
}
