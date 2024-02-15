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
from .utils import values_in_time, null_checker, get_next_workday
from dateutil.relativedelta import relativedelta

__all__ = [
    "ReturnsForecast",
    "MPOReturnsForecast",
    "onlineMPOReturnsForecast",
    "MultipleReturnsForecasts",
]


class BaseReturnsModel(Expression):
    pass


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


class BlackLittermanModel(FilterAssets, BaseReturnsModel):
    def __init__(self, covariance_matrix=None, rf_rate=None, **kwargs):
        self.covariance_matrix = covariance_matrix
        self.rf_rate = rf_rate
        super().__init__(asset_filter=True, **kwargs)

    # Calculates portfolio mean return
    def port_mean(self, W, R):
        # return np.sum(R * W)
        return np.dot(W, R)

    # Calculates portfolio variance of returns
    def port_var(self, W, C):
        return np.dot(np.dot(W, C), W)

    # Combination of the two functions above - mean and variance of returns calculation
    def port_mean_var(self, W, R, C):
        return self.port_mean(W, R), self.port_var(W, C)

    def get_BL(self, returns, weights, t):
        """
        use historical returns, covariances and float shares weights to get
        excess returns (float share adjusted returns).
        NOTE: This is for each period

        NOTE: investor view are not implemented

        Returns:

        - excess returns
        """
        # index = returns.name
        cov = self.covariance_matrix.get_sigma(t)

        # Filter covariance matrix:
        # idx = cov.columns.get_indexer(self.assets)
        # cov = cov.loc[:, self.assets].iloc[idx]

        # Calculate portfolio historical return and variance
        mean, var = self.port_mean_var(weights, returns, cov)

        lmb = (mean - self.rf_rate) / var  # Calculate risk aversion
        Pi = np.dot(np.dot(lmb, cov), weights)  # Calculate equilibrium excess returns
        return pd.Series(data=Pi, index=returns.index)


class ReturnsForecast(BlackLittermanModel):
    """A single return forecast.

    Attributes:
      ret_est: A dataframe of return estimates.
      delta_data: A confidence interval around the estimates.
      half_life: Number of days for alpha auto-correlation to halve.
    """

    def __init__(self, returns, delta=0.0, gamma_decay=None, name=None, **kwargs):
        null_checker(returns)
        self.returns = returns
        null_checker(delta)
        self.delta = delta
        self.gamma_decay = gamma_decay
        self.name = name
        self.assets = returns.columns
        super().__init__(**kwargs)

    def weight_expr(self, t, wplus=None, z=None, v=None, w_index=None):
        """Returns the estimated alpha.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        returns = values_in_time(self.returns, t)[self.assets]

        if self.covariance_matrix is not None and w_index is not None:
            alpha = self.get_BL(returns, w_index, t)
            return alpha @ (wplus - w_index)
        elif w_index is None:
            alpha = returns
            return alpha @ wplus
        elif wplus is None:
            return returns
        else:
            alpha = cvx.multiply(returns, wplus)
            alpha -= cvx.multiply(values_in_time(self.delta, t), cvx.abs(wplus))
            return cvx.sum(alpha)

    def weight_expr_ahead(self, t, tau, wplus=None, w_index=None):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """

        alpha = self.weight_expr(t, wplus=wplus, w_index=w_index)
        if tau > t and self.gamma_decay is not None:
            alpha *= (tau - t).days ** (-self.gamma_decay)
        return alpha


class MPOReturnsForecast(BlackLittermanModel):
    """A single alpha estimation.

    Attributes:
      ret_est: A dict of series of return estimates.
    """

    def __init__(self, ret_est, **kwargs):
        self.ret_est = ret_est
        super().__init__(**kwargs)

    def weight_expr_ahead(self, t, tau, wplus=None, w_index=None, excess=False):
        """Returns the estimate at time t of alpha at time tau.

        Args:
            t: time estimate is made.
            wplus: An expression for holdings.
            tau: time of alpha being estimated.

        Returns:
            An expression for the alpha.
        """
        if excess:
            assert w_index is not None
            wplus = wplus - w_index
        if self.ret_est is None:
            raise ValueError("Return estimates were never generated or assigned")
        if wplus is None:
            return self.ret_est[(t, tau)][self.assets]
        elif w_index is None or self.covariance_matrix is None:
            return self.ret_est[(t, tau)][self.assets].values.T @ wplus
        else:
            alpha = self.get_BL(self.ret_est[(t, tau)][self.assets], w_index, t)
            return alpha.values.T @ (wplus)


class onlineMPOReturnsForecast(MPOReturnsForecast):
    def __init__(self, ret, lookahead_periods, trading_times, trading_frequency, **kwargs):
        self.lookahead_periods = lookahead_periods
        self.ret = ret
        self.trading_times = trading_times
        self.online = True
        self.trading_frequency = trading_frequency
        super().__init__(ret_est=None, **kwargs)

    def update(self, t):
        rng = np.random.default_rng()
        self.ret_est = {}
        sample_size = 1000

        idx = self.ret.index.get_indexer([t], method="pad")[0]
        end_dt = idx
        start_dt = max(idx - 252, 0)
        assert start_dt >= 0

        self.ret_obs = self.ret.iloc[start_dt:end_dt]
        # turn observed returns into log returns
        self.ret_obs = np.log(self.ret_obs + 1)
        # trading_frequency dictionary 
        freq = {"daily": "B", "weekly": "W", "monthly": "BMS", "quarterly": "BQS", "yearly": "BYS"}
        # add returns based on trading frequency
        self.ret_obs = self.ret_obs.resample(freq[self.trading_frequency]).sum()
        # undo log returns
        self.ret_obs = np.exp(self.ret_obs) - 1
        # numpy objects
        mu = np.mean(self.ret_obs, axis=0)
        cov = np.cov(self.ret_obs, rowvar=False)

        idx = self.trading_times.index(t)
        periods = self.trading_times[idx : idx + self.lookahead_periods]
        index_list = self.ret.loc[periods[0] : periods[-1]].index
        num_days = len(index_list)

        for i, tau in enumerate(periods):
            self.ret_est[(periods[0], tau)] = mu
    
    # def update(self, t):
    #     rng = np.random.default_rng()
    #     self.ret_est = {}
    #     sample_size = 1000

    #     idx = self.ret.index.get_indexer([t], method="pad")[0]
    #     end_dt = idx
    #     start_dt = max(idx - 252, 0)
    #     assert start_dt >= 0

    #     self.ret_obs = self.ret.iloc[start_dt:end_dt]
    #     # numpy objects
    #     mu = np.mean(self.ret_obs, axis=0)
    #     cov = np.cov(self.ret_obs, rowvar=False)

    #     idx = self.trading_times.index(t)
    #     periods = self.trading_times[idx : idx + self.lookahead_periods]
    #     index_list = self.ret.loc[periods[0] : periods[-1]].index
    #     num_days = len(index_list)

    #     # create planning matrix
    #     mvn = rng.multivariate_normal(mu, cov, size=(sample_size, num_days))
    #     self.mvn_avg = pd.DataFrame(index=index_list, data=mvn.mean(axis=0), columns=self.ret.columns)

    #     for i, tau in enumerate(periods):
    #         # cumulative return for each rebal period (previous tau to current tau)
    #         if i == 0:
    #             self.ret_est[(periods[0], tau)] = ((self.mvn_avg.loc[periods[0] : tau] + 1).cumprod() - 1).iloc[-1]
    #         else:
    #             self.ret_est[(periods[0], tau)] = ((self.mvn_avg.loc[periods[i - 1] : tau] + 1).cumprod() - 1).iloc[-1]


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
