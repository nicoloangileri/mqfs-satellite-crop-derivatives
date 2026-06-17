% ============================================================================
% plot_results.m - end-to-end MATLAB demo of the MQFS satellite crop pricer
% ----------------------------------------------------------------------------
% Self-contained: synthesises a dekadal NDVI series for a Sicilian durum-wheat
% zone, fits the harmonic climatology, calibrates the OU anomaly, prices the
% parametric drought put, runs a historical burn-cost backtest, and writes
% three figures to ../figures/. No Python or external data required.
%
% Mirrors the model in src/ (NumPy/C/C++) so all four languages agree.
%
% Run:  >> plot_results
%
% (c) Nicolo Angileri / MQFS - MIT Licence.
% ============================================================================
clear; close all; clc;
rng(7, 'twister');

%% --- 1. Synthetic dekadal NDVI series ------------------------------------
cadence = 10;                                  % days
dt = cadence / 365.0;
dates = (datetime(2016,1,1):caldays(cadence):datetime(2026,5,1))';
doy = day(dates, 'dayofyear');
yr = year(dates);
n = numel(dates);

% Mediterranean winter-cereal climatology (peak ~ DOY 105).
seasonal_true = mediter_seasonal(doy);

% OU anomaly.
kappa_true = 6.0; sigma_true = 0.12;
a = exp(-kappa_true*dt); b = sigma_true*sqrt((1-exp(-2*kappa_true*dt))/(2*kappa_true));
anom = zeros(n,1); s = 0;
for i = 1:n, s = a*s + b*randn; anom(i) = s; end

% Inject spring droughts.
drought = zeros(n,1);
spring = doy >= 60 & doy <= 150;
for y = [2017 2020 2023 2026]
    sel = spring & (yr == y);
    drought(sel) = drought(sel) - 0.14*(0.7 + 0.6*rand);
end

ndvi = min(max(seasonal_true + anom + drought + 0.01*randn(n,1), -0.2), 0.95);

%% --- 2. Harmonic climatology fit (2 harmonics) ---------------------------
H = 2;
D = harmonic_design(doy, H);
coef = D \ ndvi;
seasonal_fit = D * coef;
anomaly = ndvi - seasonal_fit;

%% --- 3. OU calibration ----------------------------------------------------
p = calibrate_ou(anomaly, dt);
fprintf('Calibrated OU: kappa=%.3f  sigma=%.3f  half-life=%.3f yr  x0=%.3f\n', ...
        p.kappa, p.sigma, p.half_life_years, p.x0);

%% --- 4. Seasonal grid + contract + pricing -------------------------------
grid_doy = 75:10:151;
seasonal_grid = (harmonic_design(grid_doy', H) * coef)';

contract = struct('strike', NaN, 'tick', 250000, 'limit', 50000, ...
                  'index', 'mean', 'style', 'put');
market = struct('risk_free', 0.03, 'horizon', 0.5, 'sd_theta', 0.25, 'wang_lambda', 0.15);
mc = struct('n_paths', 200000, 'antithetic', true, 'seed', 20260506, ...
            'ndvi_floor', -0.20, 'ndvi_cap', 0.95);

% Strike at the 20th percentile of the historical seasonal-mean index.
seas_idx = seasonal_index_series(dates, ndvi, grid_doy);
contract.strike = quantile(seas_idx, 0.20);
fprintf('Strike (20th pct seasonal index): %.4f\n', contract.strike);

res = price_parametric(seasonal_grid, p.kappa, p.sigma, dt, contract, market, mc);
fprintf('\nPricing sheet\n');
fprintf('  fair (burn) value    : EUR %12.2f\n', res.fair_value);
fprintf('  SD-principle premium : EUR %12.2f\n', res.sd_premium);
fprintf('  Wang-transform premium: EUR %12.2f\n', res.wang_premium);
fprintf('  trigger probability  : %12.2f %%\n', 100*res.trigger_probability);
fprintf('  writer CVaR 95%%      : EUR %12.2f\n', res.cvar_95);

%% --- 5. Burn analysis -----------------------------------------------------
burn_analysis(seas_idx, contract, res.sd_premium);

%% --- 6. Figures -----------------------------------------------------------
outdir = fullfile('..','figures');
if ~exist(outdir, 'dir'), mkdir(outdir); end

% (a) NDVI, climatology, anomaly.
f1 = figure('Position',[100 100 900 380],'Color','w');
plot(dates, ndvi, '.', 'Color',[0.6 0.6 0.6], 'MarkerSize',6); hold on;
plot(dates, seasonal_fit, '-', 'Color',[0.10 0.45 0.70], 'LineWidth',1.6);
ylabel('NDVI'); title('Durum-wheat NDVI and fitted harmonic climatology');
legend('Observed dekadal NDVI','Climatology','Location','southwest'); grid on;
exportgraphics(f1, fullfile(outdir,'matlab_ndvi_climatology.png'), 'Resolution',150);

% (b) Fan chart of simulated in-season NDVI paths.
[paths, qs] = simulate_fan(seasonal_grid, p.kappa, p.sigma, dt, mc);
f2 = figure('Position',[100 100 760 420],'Color','w');
xg = grid_doy;
fill([xg fliplr(xg)], [qs(1,:) fliplr(qs(5,:))], [0.85 0.90 0.97], 'EdgeColor','none'); hold on;
fill([xg fliplr(xg)], [qs(2,:) fliplr(qs(4,:))], [0.70 0.80 0.93], 'EdgeColor','none');
plot(xg, qs(3,:), '-', 'Color',[0.10 0.30 0.60], 'LineWidth',1.8);
yline(contract.strike, '--r', 'Strike', 'LineWidth',1.3);
xlabel('Day of year'); ylabel('NDVI'); grid on;
title('Simulated in-season NDVI - median, 50% and 90% bands');
legend('90% band','50% band','Median','Location','southwest');
exportgraphics(f2, fullfile(outdir,'matlab_fan_chart.png'), 'Resolution',150);

% (c) Payoff distribution (conditional on trigger).
[payouts, ~] = price_payouts(seasonal_grid, p.kappa, p.sigma, dt, contract, mc);
f3 = figure('Position',[100 100 760 380],'Color','w');
histogram(payouts(payouts>0), 50, 'FaceColor',[0.20 0.55 0.40], 'EdgeColor','none');
xlabel('Payout (EUR), triggered seasons'); ylabel('Frequency'); grid on;
title(sprintf('Payoff distribution | trigger prob = %.1f%%', 100*res.trigger_probability));
exportgraphics(f3, fullfile(outdir,'matlab_payoff_dist.png'), 'Resolution',150);

fprintf('\nFigures written to %s\n', outdir);

% ============================================================================
% Local functions
% ============================================================================
function s = mediter_seasonal(doy)
    phase = 2*pi*(doy - 105)/365.0;
    base = 0.42 + 0.30*cos(phase) + 0.06*cos(2*phase);
    s = min(max(base, 0.12), 0.80);
end

function D = harmonic_design(doy, H)
    w = 2*pi*doy/365.0;
    D = ones(numel(doy), 1 + 2*H);
    for k = 1:H
        D(:, 2*k)   = cos(k*w);
        D(:, 2*k+1) = sin(k*w);
    end
end

function si = seasonal_index_series(dates, ndvi, grid_doy)
    doy = day(dates,'dayofyear'); yr = year(dates);
    lo = min(grid_doy); hi = max(grid_doy);
    in = doy >= lo & doy <= hi;
    yrs = unique(yr(in));
    si = zeros(numel(yrs),1);
    for i = 1:numel(yrs)
        si(i) = mean(ndvi(in & yr==yrs(i)));
    end
end

function [paths, qs] = simulate_fan(seasonal, kappa, sigma, dt, mc)
    M = numel(seasonal); a = exp(-kappa*dt);
    b = sigma*sqrt((1-exp(-2*kappa*dt))/(2*kappa));
    N = min(mc.n_paths, 20000);
    Z = randn(N, M); X = zeros(N,M); st = zeros(N,1);
    for k=1:M, st = a*st + b*Z(:,k); X(:,k)=st; end
    paths = min(max(seasonal + X, mc.ndvi_floor), mc.ndvi_cap);
    qs = quantile(paths, [0.05 0.25 0.5 0.75 0.95]);
end

function [payouts, idx] = price_payouts(seasonal, kappa, sigma, dt, contract, mc)
    M = numel(seasonal); a = exp(-kappa*dt);
    b = sigma*sqrt((1-exp(-2*kappa*dt))/(2*kappa));
    N = mc.n_paths;
    if mc.antithetic
        half = ceil(N/2); Zc = randn(half,M); Z=[Zc;-Zc]; Z=Z(1:N,:);
    else, Z = randn(N,M); end
    X = zeros(N,M); st=zeros(N,1);
    for k=1:M, st=a*st+b*Z(:,k); X(:,k)=st; end
    ndvi = min(max(seasonal + X, mc.ndvi_floor), mc.ndvi_cap);
    idx = mean(ndvi,2);
    payouts = min(contract.limit, contract.tick*max(0, contract.strike - idx));
end
