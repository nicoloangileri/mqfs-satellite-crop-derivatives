/* Standalone smoke test for the C kernel: prices a toy durum-wheat contract
 * and prints the pricing sheet. Build with `make demo`, run `./cli_demo`. */
#include "ou_mc.h"
#include <stdio.h>

int main(void) {
    /* 8 dekads across the mid-March..end-May risk window */
    double seasonal[8] = {0.55, 0.62, 0.68, 0.71, 0.70, 0.64, 0.55, 0.45};
    int M = 8;

    mqfs_contract c;
    c.strike = 0.58; c.tick = 250000.0; c.limit = 50000.0;
    c.index_type = MQFS_INDEX_MEAN; c.style = MQFS_STYLE_PUT;

    mqfs_result r;
    int rc = mqfs_price_ou(seasonal, M,
                           /*kappa*/ 6.0, /*sigma*/ 0.12, /*dt*/ 10.0/365.0, /*x0*/ 0.0,
                           /*floor*/ -0.2, /*cap*/ 0.95,
                           /*n_paths*/ 200000, /*antithetic*/ 1, /*seed*/ 20260506ULL,
                           &c, /*r*/ 0.03, /*T*/ 0.5, /*theta*/ 0.25, &r);
    if (rc) { fprintf(stderr, "pricing failed rc=%d\n", rc); return rc; }

    printf("MQFS C kernel — pricing sheet\n");
    printf("  fair value            EUR %12.2f\n", r.fair_value);
    printf("  SD premium            EUR %12.2f\n", r.sd_premium);
    printf("  E[payout]             EUR %12.2f\n", r.expected_payout);
    printf("  trigger probability       %8.4f\n", r.trigger_probability);
    printf("  E[payout|triggered]   EUR %12.2f\n", r.expected_payout_if_triggered);
    printf("  VaR 95%%               EUR %12.2f\n", r.var_95);
    printf("  CVaR 95%%              EUR %12.2f\n", r.cvar_95);
    printf("  paths                     %8ld\n", r.n_paths);
    return 0;
}
