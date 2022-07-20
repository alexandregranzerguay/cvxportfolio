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


import cvxpy as cvx
import pandas as pd
import numpy as np
import functools
from .expression import Expression
from .utils import values_in_time, null_checker

__all__ = [
    "ReturnsForecast",
    "MPOReturnsForecast",
    "MultipleReturnsForecasts",
    "MPOIndexReturnsForecast",
]


class BaseReturnsModel(Expression):
    pass


class ReturnsForecast(BaseReturnsModel):
    """A single return forecast.

    Attributes:
      alpha_data: A dataframe of return estimates.
      delta_data: A confidence interval around the estimates.
      half_life: Number of days for alpha auto-correlation to halve.
    """

    def __init__(self, returns, delta=0.0, gamma_decay=None, name=None):
        null_checker(returns)
        self.returns = returns
        null_checker(delta)
        self.delta = delta
        self.gamma_decay = gamma_decay
        self.name = name

    def weight_expr(self, t, wplus=None, z=None, v=None):
        """Returns the estimated alpha.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        if wplus is None:
            alpha = values_in_time(self.returns, t)
            # assert (isinstance(alpha, float))
            return alpha
        else:
            alpha = cvx.multiply(values_in_time(self.returns, t), wplus)
            alpha -= cvx.multiply(values_in_time(self.delta, t), cvx.abs(wplus))
            return cvx.sum(alpha)

    def weight_expr_ahead(self, t, tau, wplus=None):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """

        alpha = self.weight_expr(t, wplus)
        if tau > t and self.gamma_decay is not None:
            alpha *= (tau - t).days ** (-self.gamma_decay)
        return alpha


class BlackLittermanModel:
    def __init__(self, covariance_matrix=None, rf_rate=None):
        self.covariance_matrix = covariance_matrix
        self.rf_rate = rf_rate

    # Calculates portfolio mean return
    def port_mean(self, W, R):
        return np.sum(R * W)

    # Calculates portfolio variance of returns
    def port_var(self, W, C):
        return np.dot(np.dot(W, C), W)

    # Combination of the two functions above - mean and variance of returns calculation
    def port_mean_var(self, W, R, C):
        return self.port_mean(W, R), self.port_var(W, C)

    def get_BL(self, returns, weights, t):
        """
        use historical returns, covariances and float shares weights to get
        excess returns (flaot share adjusted returns).
        NOTE: This is for each period

        NOTE: investor view are not implemented

        Returns:

        - excess returns
        """
        # index = returns.name
        cov = values_in_time(self.covariance_matrix, t)

        # Calculate portfolio historical return and variance
        mean, var = self.port_mean_var(weights, returns, cov)

        lmb = (mean - self.rf_rate) / var  # Calculate risk aversion
        Pi = np.dot(np.dot(lmb, cov), weights)  # Calculate equilibrium excess returns
        # excess_ret = Pi + rf #TODO: Check that this isn't double counting
        excess_ret = Pi
        # print(type(excess_ret))
        return excess_ret


class MPOReturnsForecast(BaseReturnsModel, BlackLittermanModel):
    """A single alpha estimation.

    Attributes:
      alpha_data: A dict of series of return estimates.
    """

    def __init__(self, alpha_data, **kwargs):
        self.alpha_data = alpha_data
        super(MPOReturnsForecast, self).__init__(**kwargs)

    def weight_expr_ahead(self, t, tau, wplus, w_index=None):
        """Returns the estimate at time t of alpha at time tau.

        Args:
            t: time estimate is made.
            wplus: An expression for holdings.
            tau: time of alpha being estimated.

        Returns:
            An expression for the alpha.
        """
        if self.covariance_matrix is not None:
            alpha = self.get_BL(self.alpha_data[(t, tau)], w_index, t)
            return alpha @ (wplus - w_index)
        if w_index is None:
            return self.alpha_data[(t, tau)].values.T @ wplus
        else:
            return self.alpha_data[(t, tau)].values.T @ (wplus - w_index)

class MultipleReturnsForecasts(BaseReturnsModel):
    """A weighted combination of alpha sources.

    Attributes:
      alpha_sources: a list of alpha sources.
      weights: An array of weights for the alpha sources.
    """

    def __init__(self, alpha_sources, weights):
        self.alpha_sources = alpha_sources
        self.weights = weights

    def weight_expr(self, t, wplus, z=None, v=None):
        """Returns the estimated alpha.

        Args:
            t: time estimate is made.
            wplus: An expression for holdings.
            tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        for idx, source in enumerate(self.alpha_sources):
            alpha += source.weight_expr(t, wplus) * self.weights[idx]
        return alpha

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        for idx, source in enumerate(self.alpha_sources):
            alpha += source.weight_expr_ahead(t, tau, wplus) * self.weights[idx]
        return alpha
