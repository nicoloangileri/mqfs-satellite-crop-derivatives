function out = burn_analysis(seasonal_index, contract, premium)
% BURN_ANALYSIS  Historical burn-cost backtest of a parametric drought contract.
%
%   out = BURN_ANALYSIS(seasonal_index, contract, premium) replays the contract
%   over the realised history of the seasonal settlement index and reports what
%   the grower would actually have collected. The "burn cost" is the simple
%   historical mean payout - the model-free anchor any quote is judged against.
%
%   Inputs
%     seasonal_index : vector of one index value per historical season
%     contract       : struct with .strike, .tick, .limit, .style
%     premium        : (optional) charged premium, for the loss ratio
%
%   Output struct
%     .payouts        per-season realised payout
%     .burn_cost      mean historical payout (undiscounted)
%     .trigger_rate   fraction of seasons that paid out
%     .worst          largest single-season payout
%     .loss_ratio     burn_cost / premium  (if premium given)
%
%   (c) Nicolo Angileri / MQFS - MIT Licence.

    I = seasonal_index(:);

    switch contract.style
        case 'put'
            payouts = min(contract.limit, contract.tick * max(0, contract.strike - I));
        case 'digital'
            payouts = contract.limit * double(I < contract.strike);
        otherwise, error('unknown payoff style %s', contract.style);
    end

    out = struct();
    out.payouts = payouts;
    out.burn_cost = mean(payouts);
    out.trigger_rate = mean(payouts > 0);
    out.worst = max(payouts);
    if nargin >= 3 && ~isempty(premium) && premium > 0
        out.loss_ratio = out.burn_cost / premium;
    else
        out.loss_ratio = NaN;
    end

    fprintf('Burn analysis over %d seasons\n', numel(I));
    fprintf('  burn cost (mean payout) : EUR %12.2f\n', out.burn_cost);
    fprintf('  trigger rate            : %12.1f %%\n', 100 * out.trigger_rate);
    fprintf('  worst season payout     : EUR %12.2f\n', out.worst);
    if ~isnan(out.loss_ratio)
        fprintf('  loss ratio @ premium    : %12.1f %%\n', 100 * out.loss_ratio);
    end
end
