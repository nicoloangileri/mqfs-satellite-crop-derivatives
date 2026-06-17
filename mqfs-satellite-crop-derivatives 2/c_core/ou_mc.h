/* ====================================================================
 * mqfs_ou_mc — pure C Monte Carlo kernel for satellite crop derivatives
 * --------------------------------------------------------------------
 * Zero-dependency, single-translation-unit reference engine. Compiles
 * anywhere a C99 compiler exists and is callable from Python via ctypes,
 * from MATLAB via `loadlibrary`, or linked directly into a C/C++ host.
 *
 * It implements exactly the seasonal-mean + Ornstein-Uhlenbeck-anomaly
 * model of `pricing_engine/monte_carlo.py`, so its output must agree with
 * the NumPy reference to within Monte Carlo standard error. This is the
 * "audit kernel": small enough to read in one sitting, fast enough to
 * price a whole book.
 *
 * (c) Nicolo Angileri / MQFS — MIT Licence.
 * ==================================================================== */
#ifndef MQFS_OU_MC_H
#define MQFS_OU_MC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Index aggregator: how the in-season NDVI path collapses to one number. */
enum mqfs_index_type { MQFS_INDEX_MEAN = 0, MQFS_INDEX_INTEGRAL = 1, MQFS_INDEX_MIN = 2 };

/* Payoff style. */
enum mqfs_style { MQFS_STYLE_PUT = 0, MQFS_STYLE_DIGITAL = 1 };

typedef struct {
    double strike;        /* K, NDVI index units                         */
    double tick;          /* EUR per 1.00 NDVI unit below strike          */
    double limit;         /* EUR payout cap                               */
    int    index_type;    /* enum mqfs_index_type                         */
    int    style;         /* enum mqfs_style                              */
} mqfs_contract;

typedef struct {
    double fair_value;                  /* exp(-rT) * E[payoff]           */
    double sd_premium;                  /* SD principle premium           */
    double expected_payout;             /* undiscounted E[payoff]         */
    double payout_std;                  /* SD[payoff]                     */
    double trigger_probability;         /* P(index < strike)              */
    double expected_payout_if_triggered;
    double var_95;                      /* writer 95% VaR of payout       */
    double cvar_95;                     /* writer 95% CVaR of payout      */
    long   n_paths;
} mqfs_result;

/* Price one parametric contract by Monte Carlo.
 *   seasonal : length-M climatology on the risk-window dekadal grid
 *   kappa,sigma,dt,x0 : calibrated OU params + current anomaly state
 *   floor,cap         : NDVI clipping bounds
 *   antithetic        : 1 to use antithetic variates (n_paths should be even)
 * Returns 0 on success, non-zero on allocation failure. */
int mqfs_price_ou(const double *seasonal, int M,
                  double kappa, double sigma, double dt, double x0,
                  double floor, double cap,
                  long n_paths, int antithetic, unsigned long long seed,
                  const mqfs_contract *contract,
                  double risk_free, double horizon, double sd_theta,
                  mqfs_result *out);

#ifdef __cplusplus
}
#endif
#endif /* MQFS_OU_MC_H */
