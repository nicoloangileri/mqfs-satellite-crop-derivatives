/* ====================================================================
 * mqfs_ou_mc — implementation
 * RNG : xoshiro256** seeded by splitmix64 (fast, high-quality, public domain)
 * N(0,1) : cached Box-Muller
 * ==================================================================== */
#include "ou_mc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* M_PI is not in strict C99; define the constant we need explicitly. */
#define MQFS_TWO_PI 6.28318530717958647692

/* ---------------------- xoshiro256** PRNG ------------------------- */
typedef struct { unsigned long long s[4]; int has_spare; double spare; } rng_t;

static inline unsigned long long splitmix64(unsigned long long *x) {
    unsigned long long z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_seed(rng_t *r, unsigned long long seed) {
    unsigned long long sm = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; ++i) r->s[i] = splitmix64(&sm);
    r->has_spare = 0;
    r->spare = 0.0;
}

static inline unsigned long long rotl(unsigned long long x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline unsigned long long rng_next(rng_t *r) {
    const unsigned long long result = rotl(r->s[1] * 5ULL, 7) * 9ULL;
    const unsigned long long t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = rotl(r->s[3], 45);
    return result;
}

/* uniform in (0,1) using the top 53 bits */
static inline double rng_uniform(rng_t *r) {
    return ((rng_next(r) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

/* standard normal via cached Box-Muller */
static inline double rng_normal(rng_t *r) {
    if (r->has_spare) { r->has_spare = 0; return r->spare; }
    double u1 = rng_uniform(r), u2 = rng_uniform(r);
    double mag = sqrt(-2.0 * log(u1));
    r->spare = mag * sin(MQFS_TWO_PI * u2);
    r->has_spare = 1;
    return mag * cos(MQFS_TWO_PI * u2);
}

/* ---------------------- helpers ----------------------------------- */
static inline double clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* single-path NDVI index given a pre-drawn normal buffer z[M] */
static double path_index(const double *seasonal, int M,
                         double a, double b, double x0,
                         double floor, double cap,
                         const double *z, double dt, int index_type) {
    double state = x0;
    double acc = 0.0, mn = 1e30;
    for (int k = 0; k < M; ++k) {
        state = a * state + b * z[k];
        double ndvi = clip(seasonal[k] + state, floor, cap);
        acc += ndvi;
        if (ndvi < mn) mn = ndvi;
    }
    if (index_type == MQFS_INDEX_MEAN)     return acc / (double)M;
    if (index_type == MQFS_INDEX_INTEGRAL) return dt * acc;
    return mn; /* MQFS_INDEX_MIN */
}

static double payoff_of(double index, const mqfs_contract *c) {
    if (c->style == MQFS_STYLE_PUT) {
        double raw = c->tick * (c->strike - index > 0.0 ? c->strike - index : 0.0);
        return raw < c->limit ? raw : c->limit;
    }
    /* digital */
    return index < c->strike ? c->limit : 0.0;
}

/* ---------------------- main entry -------------------------------- */
int mqfs_price_ou(const double *seasonal, int M,
                  double kappa, double sigma, double dt, double x0,
                  double floor, double cap,
                  long n_paths, int antithetic, unsigned long long seed,
                  const mqfs_contract *c,
                  double risk_free, double horizon, double sd_theta,
                  mqfs_result *out) {
    if (n_paths <= 0 || M <= 0) return 1;

    const double a = exp(-kappa * dt);
    const double b = sigma * sqrt((1.0 - exp(-2.0 * kappa * dt)) / (2.0 * kappa));

    double *payouts = (double *)malloc((size_t)n_paths * sizeof(double));
    double *zbuf    = (double *)malloc((size_t)M * sizeof(double));
    if (!payouts || !zbuf) { free(payouts); free(zbuf); return 2; }

    rng_t rng; rng_seed(&rng, seed);

    long i = 0;
    while (i < n_paths) {
        for (int k = 0; k < M; ++k) zbuf[k] = rng_normal(&rng);

        double idx = path_index(seasonal, M, a, b, x0, floor, cap, zbuf, dt, c->index_type);
        payouts[i++] = payoff_of(idx, c);

        if (antithetic && i < n_paths) {
            for (int k = 0; k < M; ++k) zbuf[k] = -zbuf[k];   /* antithetic twin */
            double idx2 = path_index(seasonal, M, a, b, x0, floor, cap, zbuf, dt, c->index_type);
            payouts[i++] = payoff_of(idx2, c);
        }
    }

    /* moments + trigger stats */
    double sum = 0.0, trig_sum = 0.0;
    long trig_n = 0;
    for (long j = 0; j < n_paths; ++j) {
        sum += payouts[j];
        if (payouts[j] > 0.0) { trig_n++; trig_sum += payouts[j]; }
    }
    double mean = sum / (double)n_paths;

    double ss = 0.0;
    for (long j = 0; j < n_paths; ++j) { double d = payouts[j] - mean; ss += d * d; }
    double var = (n_paths > 1) ? ss / (double)(n_paths - 1) : 0.0;
    double sd = sqrt(var);

    /* VaR / CVaR at 95% via full sort (audit-grade; book sizes are modest) */
    qsort(payouts, (size_t)n_paths, sizeof(double), cmp_double);
    long qidx = (long)(0.95 * (double)n_paths);
    if (qidx >= n_paths) qidx = n_paths - 1;
    double var95 = payouts[qidx];
    double tail = 0.0; long tail_n = 0;
    for (long j = qidx; j < n_paths; ++j) { tail += payouts[j]; tail_n++; }
    double cvar95 = tail_n ? tail / (double)tail_n : var95;

    double disc = exp(-risk_free * horizon);
    out->expected_payout = mean;
    out->payout_std = sd;
    out->fair_value = disc * mean;
    out->sd_premium = disc * (mean + sd_theta * sd);
    out->trigger_probability = (double)trig_n / (double)n_paths;
    out->expected_payout_if_triggered = trig_n ? trig_sum / (double)trig_n : 0.0;
    out->var_95 = var95;
    out->cvar_95 = cvar95;
    out->n_paths = n_paths;

    free(payouts);
    free(zbuf);
    return 0;
}
