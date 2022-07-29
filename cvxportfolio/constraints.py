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
    "DollarNeutral",
    "MaxTrade",
    "MaxWeights",
    "MinWeights",
    "FactorMaxLimit",
    "FactorMinLimit",
    "FixedAlpha",
    "Cardinality",
    "TrackingErrorMax",
]


class BaseConstraint(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.w_bench = kwargs.pop("w_bench", 0.0)

    def weight_expr(self, t, w_plus, z, v):
        """Returns a list of trade constraints.

        Args:
          t: time
          w_plus: post-trade weights
          z: trade weights
          v: portfolio value
        """
        if w_plus is None:
            return self._weight_expr(t, None, z, v)
        return self._weight_expr(t, w_plus - self.w_bench, z, v)

    @abstractmethod
    def _weight_expr(self, t, w_plus, z, v):
        pass


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


class LongOnly(BaseConstraint):
    """A long only constraint."""

    def __init__(self, **kwargs):
        super(LongOnly, self).__init__(**kwargs)

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


class LongCash(BaseConstraint):
    """Requires that cash be non-negative."""

    def __init__(self, **kwargs):
        super(LongCash, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        return w_plus[-1] >= 0


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


class Cardinality(BaseConstraint):
    """A constraint to impose cardinality constraint on portfolio, this introduces MIP complexity"""

    def __init__(self, limit, **kwargs):
        super(Cardinality, self).__init__(**kwargs)
        self.limit = limit

    def _weight_expr(self, t, w_plus, z, v):
        y = cvx.Variable(w_plus[:-1].shape[0], boolean=True)
        constr = []
        constr += [sum(y) <= self.limit]
        for i in range(w_plus[:-1].shape[0]):
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


class TrackingErrorMax(IndexUpdater, BaseConstraint):
    def __init__(self, Sigma, limit, float_shares, index_prices, use_updater=False, **kwargs):
        self.Sigma = Sigma  # Sigma is either a matrix or a pd.Panel
        self.limit = limit
        self.float_shares = float_shares
        self.index_prices = index_prices
        try:
            assert not pd.isnull(Sigma).values.any()
        except AttributeError:
            assert not pd.isnull(Sigma).any()
        super().__init__(**kwargs)

        # if use_updater:
        #     self._init_updater(index_prices=index_prices, sigma=Sigma, **kwargs)

    def weight_expr(self, t, w_plus, z, value):
        self.w_bench = self._get_index_weights(t)
        self.expression = self._estimate(t, w_plus - self.w_bench, z, value)
        return self.expression <= values_in_time(self.limit, t)

    def _estimate(self, t, wplus, z, value):
        try:
            self.expression = cvx.quad_form(wplus, values_in_time(self.Sigma, t))
        except TypeError:
            self.expression = cvx.quad_form(wplus, values_in_time(self.Sigma, t).values)
        return self.expression
