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
        return (
            cvx.abs(z[:-1]) * v
            <= np.array(values_in_time(self.ADVs, t)) * self.max_fraction
        )


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
        return values_in_time(self.factor_exposure, t).T @ w_plus[
            :-1
        ] <= values_in_time(self.limit, t)


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
        return values_in_time(self.factor_exposure, t).T @ w_plus[
            :-1
        ] >= values_in_time(self.limit, t)


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
        return values_in_time(self.return_forecast, t).T @ w_plus[
            :-1
        ] == values_in_time(self.alpha_target, t)


# TODO: Determine if it's better to implement these classes here or simply import baseconstraint class
# and implement in notebook
class Cardinality(BaseConstraint):
    """A constraint to impose cardinality constraint on portfolio
    TODO: Implement this
    """

    def __init__(self, limit, **kwargs):
        super(Cardinality, self).__init__(**kwargs)
        self.limit = limit

    def _weight_expr(self, t, w_plus, z, v):
        t = cvx.Variable(w_plus[:-1].shape[0], boolean=True)
        constr = []
        constr += [sum(t) == self.limit]
        constr += [w_plus[:-1] - t <= 0]
        return constr


class TrackingErrorMax(BaseConstraint):
    """A constraint to impose tracking error

    Attributes:
        sigma : covariance matrix
        w_index: tracked index weights
    """

    def __init__(self, Sigma, float_shares, index_prices, limit, **kwargs):
        self.Sigma = Sigma  # Sigma is either a matrix or a pd.Panel
        self.float_shares = float_shares
        self.index_prices = index_prices
        self.limit = limit
        try:
            assert not pd.isnull(Sigma).values.any()
        except AttributeError:
            assert not pd.isnull(Sigma).any()
        super(TrackingErrorMax, self).__init__(**kwargs)

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

    def _get_index_weights(self, t):
        market_cap = (
            self.float_shares["float_shares"]
            .multiply(values_in_time(self.index_prices, t))
            .fillna(0)
        )
        index_weights = market_cap / market_cap.sum()
        index_weights["Cash"] = 0
        return index_weights
