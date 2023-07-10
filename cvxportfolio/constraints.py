"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from abc import ABCMeta, abstractmethod

import cvxpy as cvx
import numpy as np
import pandas as pd

from .utils import values_in_time


__all__ = [
    "LongOnly",
    "LeverageLimit",
    "LongCash",
    "MaxCash",
    "DollarNeutral",
    "MaxTrade",
    "MaxWeights",
    "MinWeights",
    "FactorMaxLimit",
    "FactorMinLimit",
    "FixedAlpha",
    "Cardinality",
    "TrackingErrorMax",
    "TurnoverLimit",
    "MinAlpha",
]


class BaseConstraint(object):
    __metaclass__ = ABCMeta

    def __init__(self, track_dual=False, **kwargs):
        self.w_bench = kwargs.pop("w_bench", 0.0)
        self.duals = [] if track_dual else None

    def weight_expr(self, t, w_plus, z, v):
        """Returns a list of trade constraints.

        Args:
          t: time
          w_plus: post-trade weights
          z: trade weights
          v: portfolio value
        """
        if w_plus is None:
            return self._track_duals(self._weight_expr(t, None, z, v))
        return self._track_duals(self._weight_expr(t, w_plus - self.w_bench, z, v))

    @abstractmethod
    def _weight_expr(self, t, w_plus, z, v):
        pass

    def _track_duals(self, expr):
        ## TODO: Test this code!
        if self.duals is not None:
            if isinstance(expr, list):
                self.duals += [con.dual_value for con in expr]
            else:
                self.duals.append(expr.dual_value)
        return expr


class FilterAssets:
    def __init__(self, asset_filter=True, **kwargs):
        self.asset_filter = asset_filter
        super().__init__(**kwargs)

    def filter(self, assets):
        # Returning self, allows for method chaining
        if not "asset_filter" in self.__dict__:
            raise ValueError("asset filtering not properly inherited")
        if not self.asset_filter:
            return self
        else:
            self.assets = assets
        return self


class MaxTrade(BaseConstraint):
    """A limit on maximum trading size."""

    def __init__(self, ADVs, max_fraction=0.05, **kwargs):
        self.ADVs = ADVs
        self.max_fraction = max_fraction
        super(MaxTrade, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of trade constraints.

        Args:
          t: time
          w_plus: post-trade weights
          z: trade weights
          v: portfolio value
        """
        return cvx.abs(z[:-1]) * v <= np.array(values_in_time(self.ADVs, t)) * self.max_fraction


class LongOnly(FilterAssets, BaseConstraint):
    """A long only constraint."""

    def __init__(self, **kwargs):
        super().__init__(asset_filter=False, **kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        return w_plus >= 0


class LeverageLimit(BaseConstraint):
    """A limit on leverage.

    Attributes:
      limit: A (time) series or scalar giving the leverage limit.
    """

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super(LeverageLimit, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        return cvx.norm(w_plus[:-1], 1) <= values_in_time(self.limit, t)


class LongCash(FilterAssets, BaseConstraint):
    """Requires that cash be non-negative."""

    def __init__(self, **kwargs):
        super().__init__(asset_filter=False, **kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        return w_plus[-1] >= 0


class MaxCash(FilterAssets, BaseConstraint):
    """A max limit on weight of cash.

    Attributes:
      limit: A series or number giving the weights limit.
    """

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super().__init__(asset_filter=False, **kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        return w_plus[-1] <= values_in_time(self.limit, t)


class DollarNeutral(BaseConstraint):
    """Long-short dollar neutral strategy."""

    def __init__(self, **kwargs):
        super(DollarNeutral, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        return sum(w_plus[:-1]) == 0


class MaxWeights(BaseConstraint):
    """A max limit on weights.

    Attributes:
      limit: A series or number giving the weights limit.
    """

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super(MaxWeights, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        return w_plus[:-1] <= values_in_time(self.limit, t)


class MinWeights(BaseConstraint):
    """A min limit on weights.

    Attributes:
      limit: A series or number giving the weights limit.
    """

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super(MinWeights, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        return w_plus[:-1] >= values_in_time(self.limit, t)


class FactorMaxLimit(BaseConstraint):
    """A max limit on portfolio-wide factor (e.g. beta) exposure.

    Attributes:
        factor_exposure: An (n * r) matrix giving the factor exposure per asset
        per factor, where n represents # of assets and r represents # of factors
        limit: A series of list or a single list giving the factor limits
    """

    def __init__(self, factor_exposure, limit, **kwargs):
        super(FactorMaxLimit, self).__init__(**kwargs)
        self.factor_exposure = factor_exposure
        self.limit = limit

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
            t: time
            w_plus: holdings
        """
        return values_in_time(self.factor_exposure, t).T @ w_plus[:-1] <= values_in_time(self.limit, t)


class FactorMinLimit(BaseConstraint):
    """A min limit on portfolio-wide factor (e.g. beta) exposure.

    Attributes:
        factor_exposure: An (n * r) matrix giving the factor exposure per asset
        per factor, where n represents # of assets and r represents # of factors
        limit: A series of list or a single list giving the factor limits
    """

    def __init__(self, factor_exposure, limit, **kwargs):
        super(FactorMinLimit, self).__init__(**kwargs)
        self.factor_exposure = factor_exposure
        self.limit = limit

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
            t: time
            w_plus: holdings
        """
        return values_in_time(self.factor_exposure, t).T @ w_plus[:-1] >= values_in_time(self.limit, t)


class FixedAlpha(BaseConstraint):
    """A constraint to fix portfolio-wide alpha

    Attributes:
        forecast_returns: An (n * 1) vector giving the return forecast on each
        asset
        alpha_target: A series or number giving the targeted portfolio return
    """

    def __init__(self, return_forecast, alpha_target, **kwargs):
        super(FixedAlpha, self).__init__(**kwargs)
        self.return_forecast = return_forecast
        self.alpha_target = alpha_target

    def _weight_expr(self, t, w_plus, z, v):
        return values_in_time(self.return_forecast, t).T @ w_plus[:-1] == values_in_time(self.alpha_target, t)


class MinAlpha(BaseConstraint):
    """A constraint to fix portfolio-wide alpha

    Attributes:
        forecast_returns: An (n * 1) vector giving the return forecast on each
        asset
        alpha_target: A series or number giving the targeted portfolio return
    """

    def __init__(self, return_forecast, alpha_target, **kwargs):
        super(FixedAlpha, self).__init__(**kwargs)
        self.return_forecast = return_forecast
        self.alpha_target = alpha_target

    def _weight_expr(self, t, w_plus, z, v):
        return values_in_time(self.return_forecast, t).T @ w_plus[:-1] >= values_in_time(self.alpha_target, t)


# class Alpha(BaseConstraint):
#     def __init__(self, return_forecast, alpha_target, **kwargs):
#         super(FixedAlpha, self).__init__(**kwargs)
#         self.return_return = return_forecast

#     def _weight_expr(self, t, w_plus, z, v):
#         ret = self.return_forecast.filter(self.assets).weight_expr(t, wplus=wplus, w_index=self.w_index)
#         return values_in_time(self.return_forecast, t).T @ w_plus[:-1] == values_in_time(self.alpha_target, t)


class Cardinality(FilterAssets, BaseConstraint):
    """A constraint to impose cardinality constraint on portfolio, this introduces MIP complexity"""

    def __init__(self, limit, include_cash=False, **kwargs):
        self.limit = limit
        self.include_cash = include_cash
        super().__init__(asset_filter=False, **kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        y = cvx.Variable(w_plus.shape[0], boolean=True)
        constr = []
        constr += [sum(y) <= self.limit]
        if not self.include_cash:
            for i in range(w_plus[:-1].shape[0]):
                constr += [w_plus[i] <= 1 * y[i]]
        else:
            for i in range(w_plus.shape[0]):
                constr += [w_plus[i] <= 1 * y[i]]
        return constr

        # return cvx.norm(w_plus[:-1], "nuc") <= self.limit


class IndexUpdater:
    """This class is not complete and does not work, the only part of it that works is the _get_index_weights function, where it returns index weights
    based off of a single vector of float shares. Essentially I assume that this vector would be properly updated IRL but that until I can find historical
    values of float shares, it makes no sense to try and estimate this...
    """

    def __init__(self, **kwargs):
        # Could potentially add a checker that makes sure that self has all the element this class will need outside of init updater function
        pass

    def _init_updater(self, index_ref, index_prices, budget, sigma):
        self.start_dt = sigma.index[0]
        self.market_returns = index_prices.pct_change().dropna()
        self.index_returns = index_ref.pct_change().dropna()
        self.update_freq = "q"
        self.current_quarter = index_prices.index[0].quarter
        self.current_year = index_prices.index[0].year
        self.budget = budget
        self.is_initiated = True

    def _update_required(self, t):
        # Update float_shares quarterly
        if self.update_freq == "q":
            if t.quarter != self.current_quarter or t.year != self.current_year:
                self.current_quarter = t.quarter
                self.current_year = t.year
                self._update_index_float_shares(t)

    def _update_index_float_shares(self, t):
        delta = cvx.Variable(shape=self.float_shares["float_shares"].shape)
        shares = cvx.Constant(self.float_shares["float_shares"].fillna(0))
        shares_plus = shares + delta

        w_index_plus = self._get_index_weights(t, shares=shares_plus)
        est_index_ret = w_index_plus @ self.market_returns.loc[self.start_dt : t].T
        est_index_val = (self.budget * (1 + est_index_ret).cumprod())[-1]

        est_index_val
        # # get individual sotck market cap
        # market_cap = cvx.multiply(shares_plus, values_in_time(self.index_prices, t))

        # # weight by market cap
        # weights = market_cap / cvx.sum(market_cap)

        # # Index value
        # index_val = weights @ values_in_time(self.index_prices, t)

        obj = cvx.norm(est_index_val, values_in_time(self.index_ref, t))
        constraints = [shares_plus >= 0]
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=cvx.CVXOPT)

        self.float_shares["float_shares"] = shares_plus.value

    def _get_index_weights(self, t, shares=None):
        # if shares is None:
        market_cap = self.float_shares["float_shares"].multiply(values_in_time(self.index_prices, t)).fillna(0)
        index_weights = market_cap / market_cap.sum()
        index_weights["Cash"] = 0
        # else:
        #     market_cap = cvx.multiply(shares, values_in_time(self.index_prices, t))
        #     index_weights = market_cap / cvx.sum(market_cap)
        return index_weights


class TrackingErrorMax(FilterAssets, BaseConstraint):
    def __init__(self, limit, **kwargs):
        self.limit = limit
        self.cond = 1
        super().__init__(**kwargs)

    def get_limit(self, t, track):
        return track <= values_in_time(self.limit, t)


class TurnoverLimit(FilterAssets, BaseConstraint):
    """Turnover limit as a fraction of the portfolio value.

    See page 37 of the book.

    :param delta: constant or changing in time turnover limit
    :type delta: float or pd.Series
    """

    def __init__(self, delta, half_spread=None, **kwargs):
        self.delta = delta
        self.half_spread = half_spread
        super().__init__(asset_filter=False, **kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        delta = values_in_time(self.delta, t)
        return 0.5 * cvx.norm1(z) * self.half_spread <= delta * self.half_spread
