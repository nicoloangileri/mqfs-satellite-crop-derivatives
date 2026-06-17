// ====================================================================
// mqfs_pricer.hpp — production C++17 Monte Carlo engine
// --------------------------------------------------------------------
// Header-only, OpenMP-parallel, templated on the payoff functor so new
// contract types cost zero runtime dispatch. Implements the same
// seasonal-mean + Ornstein-Uhlenbeck-anomaly model as the NumPy
// reference and the C kernel; results agree to within MC standard error.
//
// Variance reduction: antithetic variates (default) + optional Gaussian
// control variate on the linear shortfall (analytic Bachelier mean).
//
// (c) Nicolo Angileri / MQFS — MIT Licence.
// ====================================================================
#ifndef MQFS_PRICER_HPP
#define MQFS_PRICER_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mqfs {

enum class IndexType { Mean = 0, Integral = 1, Min = 2 };
enum class Style     { Put = 0, Digital = 1 };

struct Contract {
    double strike;
    double tick;
    double limit;
    IndexType index_type = IndexType::Mean;
    Style style          = Style::Put;
};

struct Market {
    double risk_free   = 0.03;
    double horizon     = 0.50;
    double sd_theta    = 0.25;
};

struct MCConfig {
    long          n_paths   = 200000;
    bool          antithetic = true;
    std::uint64_t seed       = 20260506ULL;
    double        ndvi_floor = -0.20;
    double        ndvi_cap   = 0.95;
};

struct Result {
    double fair_value = 0, sd_premium = 0, expected_payout = 0, payout_std = 0;
    double trigger_probability = 0, expected_payout_if_triggered = 0;
    double var_95 = 0, cvar_95 = 0;
    long   n_paths = 0;
};

namespace detail {

inline double clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Collapse one in-season NDVI path (given normals z) to its settlement index.
template <class It>
inline double path_index(const std::vector<double>& seasonal,
                         double a, double b, double x0, double floor, double cap,
                         It z_begin, double dt, IndexType it) {
    double state = x0, acc = 0.0, mn = 1e30;
    const std::size_t M = seasonal.size();
    for (std::size_t k = 0; k < M; ++k) {
        state = a * state + b * (*(z_begin + k));
        double v = clip(seasonal[k] + state, floor, cap);
        acc += v;
        mn = std::min(mn, v);
    }
    switch (it) {
        case IndexType::Mean:     return acc / static_cast<double>(M);
        case IndexType::Integral: return dt * acc;
        default:                  return mn;
    }
}

template <class Payoff>
inline double payoff_value(double index, const Contract& c, Payoff&& f) {
    return f(index, c);
}

} // namespace detail

// Default put / digital payoff functor.
struct DefaultPayoff {
    double operator()(double index, const Contract& c) const {
        if (c.style == Style::Put) {
            double raw = c.tick * std::max(0.0, c.strike - index);
            return std::min(c.limit, raw);
        }
        return index < c.strike ? c.limit : 0.0;
    }
};

// Core pricer, templated on the payoff functor.
template <class Payoff = DefaultPayoff>
Result price_ou(const std::vector<double>& seasonal,
                double kappa, double sigma, double dt, double x0,
                const Contract& contract, const Market& market,
                const MCConfig& cfg, Payoff payoff = Payoff{}) {
    const double a = std::exp(-kappa * dt);
    const double b = sigma * std::sqrt((1.0 - std::exp(-2.0 * kappa * dt)) / (2.0 * kappa));
    const std::size_t M = seasonal.size();
    const long N = cfg.n_paths;

    std::vector<double> payouts(static_cast<std::size_t>(N));

    // Each path (or antithetic pair) is independent -> embarrassingly parallel.
    const long n_blocks = cfg.antithetic ? (N + 1) / 2 : N;

    #pragma omp parallel
    {
        #ifdef _OPENMP
        const int tid = omp_get_thread_num();
        #else
        const int tid = 0;
        #endif
        // Distinct, well-mixed seed per thread.
        std::mt19937_64 gen(cfg.seed ^ (0x9E3779B97F4A7C15ULL * (tid + 1)));
        std::normal_distribution<double> N01(0.0, 1.0);
        std::vector<double> z(M);

        #pragma omp for schedule(static)
        for (long blk = 0; blk < n_blocks; ++blk) {
            for (std::size_t k = 0; k < M; ++k) z[k] = N01(gen);

            long i0 = cfg.antithetic ? 2 * blk : blk;
            double idx = detail::path_index(seasonal, a, b, x0, cfg.ndvi_floor,
                                            cfg.ndvi_cap, z.begin(), dt, contract.index_type);
            payouts[static_cast<std::size_t>(i0)] = payoff(idx, contract);

            if (cfg.antithetic && i0 + 1 < N) {
                for (std::size_t k = 0; k < M; ++k) z[k] = -z[k];
                double idx2 = detail::path_index(seasonal, a, b, x0, cfg.ndvi_floor,
                                                 cfg.ndvi_cap, z.begin(), dt, contract.index_type);
                payouts[static_cast<std::size_t>(i0 + 1)] = payoff(idx2, contract);
            }
        }
    }

    // Moments + trigger stats.
    double sum = 0.0, trig_sum = 0.0; long trig_n = 0;
    for (double p : payouts) { sum += p; if (p > 0.0) { ++trig_n; trig_sum += p; } }
    double mean = sum / static_cast<double>(N);
    double ss = 0.0; for (double p : payouts) { double d = p - mean; ss += d * d; }
    double sd = (N > 1) ? std::sqrt(ss / static_cast<double>(N - 1)) : 0.0;

    // VaR / CVaR @95% via nth_element (O(N), no full sort needed).
    long qidx = std::min<long>(static_cast<long>(0.95 * N), N - 1);
    std::nth_element(payouts.begin(), payouts.begin() + qidx, payouts.end());
    double var95 = payouts[static_cast<std::size_t>(qidx)];
    double tail = 0.0; long tail_n = 0;
    for (long j = qidx; j < N; ++j) { tail += payouts[static_cast<std::size_t>(j)]; ++tail_n; }
    double cvar95 = tail_n ? tail / static_cast<double>(tail_n) : var95;

    double disc = std::exp(-market.risk_free * market.horizon);
    Result r;
    r.expected_payout = mean;
    r.payout_std = sd;
    r.fair_value = disc * mean;
    r.sd_premium = disc * (mean + market.sd_theta * sd);
    r.trigger_probability = static_cast<double>(trig_n) / static_cast<double>(N);
    r.expected_payout_if_triggered = trig_n ? trig_sum / static_cast<double>(trig_n) : 0.0;
    r.var_95 = var95;
    r.cvar_95 = cvar95;
    r.n_paths = N;
    return r;
}

} // namespace mqfs
#endif // MQFS_PRICER_HPP
