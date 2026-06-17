function params = calibrate_ou(anomaly, dt)
% CALIBRATE_OU  Closed-form (OLS/MLE) calibration of the OU anomaly process.
%
%   params = CALIBRATE_OU(anomaly, dt) fits the discretised Ornstein-Uhlenbeck
%   process  x_{k+1} = a*x_k + c + eps,  a = exp(-kappa*dt), to the
%   de-seasonalised NDVI anomaly and returns a struct with fields:
%       .kappa            mean-reversion speed (per year)
%       .sigma            diffusion volatility (per sqrt-year)
%       .dt               sampling step (years)
%       .x0               last observed anomaly (initial state for MC)
%       .half_life_years  log(2)/kappa
%
%   This is the MATLAB twin of pricing_engine/ou_calibration.py:calibrate_ou,
%   producing the same estimator so cross-language calibration agrees.
%
%   (c) Nicolo Angileri / MQFS - MIT Licence.

    x = anomaly(:);
    x = x(isfinite(x));
    x0 = x(1:end-1);
    x1 = x(2:end);

    % OLS regression x1 = a*x0 + c
    A = [x0, ones(numel(x0), 1)];
    beta = A \ x1;
    a = beta(1);
    c = beta(2);
    a = min(max(a, 1e-6), 0.999999);      % keep stationary & invertible

    resid = x1 - (a * x0 + c);
    n = numel(resid);
    var_eps = sum(resid.^2) / (n - 2);     % unbiased (2 fitted params)

    kappa = -log(a) / dt;
    sigma = sqrt(var_eps * 2.0 * kappa / (1.0 - a^2));

    params = struct();
    params.kappa = kappa;
    params.sigma = sigma;
    params.dt = dt;
    params.x0 = x(end);
    params.half_life_years = log(2.0) / kappa;
end
