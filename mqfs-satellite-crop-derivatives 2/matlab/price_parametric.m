function res = price_parametric(seasonal, kappa, sigma, dt, contract, market, mc)
% PRICE_PARAMETRIC  Monte Carlo price of a satellite-indexed drought contract.
%
%   res = PRICE_PARAMETRIC(seasonal, kappa, sigma, dt, contract, market, mc)
%   simulates in-season NDVI paths under the seasonal-mean + OU-anomaly model
%   and returns a struct of priced quantities. It is the MATLAB twin of
%   pricing_engine/pricer.py and agrees with the NumPy / C / C++ engines to
%   within Monte Carlo standard error.
%
%   Inputs (structs)
%     contract.strike, .tick, .limit, .index ('mean'|'integral'|'min'),
%             .style ('put'|'digital')
%     market.risk_free, .horizon, .sd_theta, .wang_lambda
%     mc.n_paths, .antithetic (logical), .seed, .ndvi_floor, .ndvi_cap
%
%   Pricing
%     fair (burn)  = exp(-rT) E[payoff]
%     SD principle = exp(-rT) ( E[payoff] + theta SD[payoff] )
%     Wang         = exp(-rT) * int_0^inf g(Shat(y)) dy, g(u)=Phi(Phi^{-1}(u)+lambda)
%
%   (c) Nicolo Angileri / MQFS - MIT Licence.

    if nargin < 7 || isempty(mc), mc = default_mc(); end
    rng(mc.seed, 'twister');

    M = numel(seasonal);
    seasonal = seasonal(:)';                 % row vector

    a = exp(-kappa * dt);
    b = sigma * sqrt((1 - exp(-2 * kappa * dt)) / (2 * kappa));

    N = mc.n_paths;
    if mc.antithetic
        half = ceil(N / 2);
        Zc = randn(half, M);
        Z = [Zc; -Zc];
        Z = Z(1:N, :);
    else
        Z = randn(N, M);
    end

    % Propagate the OU anomaly state across dekads (vectorised over paths).
    X = zeros(N, M);
    state = zeros(N, 1);
    for k = 1:M
        state = a * state + b * Z(:, k);
        X(:, k) = state;
    end

    ndvi = seasonal + X;                      % implicit expansion
    ndvi = min(max(ndvi, mc.ndvi_floor), mc.ndvi_cap);

    % Settlement index.
    switch contract.index
        case 'mean',     idx = mean(ndvi, 2);
        case 'integral', idx = dt * sum(ndvi, 2);
        case 'min',      idx = min(ndvi, [], 2);
        otherwise, error('unknown index aggregator %s', contract.index);
    end

    % Payoff (to the insured).
    switch contract.style
        case 'put'
            payouts = min(contract.limit, contract.tick * max(0, contract.strike - idx));
        case 'digital'
            payouts = contract.limit * double(idx < contract.strike);
        otherwise, error('unknown payoff style %s', contract.style);
    end

    disc = exp(-market.risk_free * market.horizon);
    e_pay = mean(payouts);
    sd_pay = std(payouts, 0);                 % sample std (N-1)

    trig = payouts > 0;
    p_trig = mean(trig);
    e_if_trig = 0; if any(trig), e_if_trig = mean(payouts(trig)); end

    var95 = quantile(payouts, 0.95);
    tail = payouts(payouts >= var95);
    cvar95 = var95; if ~isempty(tail), cvar95 = mean(tail); end

    res = struct();
    res.fair_value = disc * e_pay;
    res.sd_premium = disc * (e_pay + market.sd_theta * sd_pay);
    res.wang_premium = disc * wang_premium(payouts, market.wang_lambda);
    res.expected_payout = e_pay;
    res.payout_std = sd_pay;
    res.trigger_probability = p_trig;
    res.expected_payout_if_triggered = e_if_trig;
    res.var_95 = var95;
    res.cvar_95 = cvar95;
    res.n_paths = N;
end

% ----------------------------------------------------------------------------
function H = wang_premium(payouts, lambda, n_grid)
% Undiscounted Wang-transform price of a non-negative loss variable.
    if nargin < 3, n_grid = 2000; end
    y = payouts(:);
    ymax = max(y);
    if ymax <= 0, H = 0; return; end
    grid = linspace(0, ymax, n_grid)';
    ys = sort(y);
    % empirical survival S(g) = P(Y > g)
    surv = 1 - (sum(ys' <= grid, 2) / numel(y));
    surv = min(max(surv, 1e-12), 1 - 1e-12);
    g = normcdf(norminv(surv) + lambda);
    H = trapz(grid, g);
end

% ----------------------------------------------------------------------------
function mc = default_mc()
    mc = struct('n_paths', 200000, 'antithetic', true, 'seed', 20260506, ...
                'ndvi_floor', -0.20, 'ndvi_cap', 0.95);
end
