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

import datetime as dt
import logging
from operator import index
import sys

# import numdifftools as nd
import traceback
from abc import ABCMeta, abstractmethod
from inspect import Parameter
from itertools import repeat

import cvxpy as cvx
import multiprocess
import ncvx as nc
import numpy as np
import pandas as pd

from .constraints import BaseConstraint, IndexUpdater, TrackingErrorMax
from .costs import BaseCost, TcostModel
from .returns import BaseReturnsModel
from .utils import null_checker, values_in_time

__all__ = [
    "Hold",
    "FixedTrade",
    "PeriodicRebalance",
    "AdaptiveRebalance",
    "SinglePeriodOpt",
    "MultiPeriodOpt",
    "ProportionalTrade",
    "RankAndLongShort",
    "PosTrackingSinglePeriodOpt",
    "QuadTrackingSinglePeriodOpt",
    "QuadTrackingMultiPeriodOpt",
    "PADMCardinalitySPO",
    "ADMCardinalitySPO",
    "NCVXCardinalitySPO",
    "PADMCardinalityMPO",
    "ADMCardinalityMPO",
]


class BasePolicy(object, metaclass=ABCMeta):
    """Base class for a trading policy."""

    def __init__(self, **kwargs):
        self.costs = []
        self.constraints = []
        # Add any other keyword args to the class dict
        self.__dict__.update(kwargs)

    @abstractmethod
    def get_trades(self, portfolio, t=dt.datetime.today()):
        """Trades list given current portfolio and time t."""
        return NotImplemented

    def _nulltrade(self, portfolio):
        return pd.Series(index=portfolio.index, data=0.0)

    def get_rounded_trades(self, portfolio, prices, t):
        """Get trades vector as number of shares, rounded to integers."""
        # trades = self.get_trades(portfolio, t)
        # temp2 = values_in_time(prices, t)[trades.index[:-1]]
        # trades[:-1] = np.round(trades[:-1] / values_in_time(prices, t)[trades.index[:-1]]).fillna(0)
        # # cash = portfolio[-1] + round_trades @ values_in_time(prices, t)
        # # return np.concatenate((round_trades, cash))
        # return trades
        trades = self.get_trades(portfolio, t)
        # Using floor for now for simplicity but this is sub-optimal obviously
        trades[:-1] = np.floor(trades[:-1] / values_in_time(prices, t)[trades.index[:-1]]).fillna(0)
        trades[:-1] = np.multiply(trades[:-1], values_in_time(prices, t)[trades.index[:-1]])
        return trades


class Hold(BasePolicy):
    """Hold initial portfolio."""

    def get_trades(self, portfolio, t=dt.datetime.today()):
        return self._nulltrade(portfolio)


class RankAndLongShort(BasePolicy):
    """Rank assets, long the best and short the worst (cash neutral)."""

    def __init__(self, return_forecast, num_long, num_short, target_turnover):
        self.target_turnover = target_turnover
        self.num_long = num_long
        self.num_short = num_short
        self.return_forecast = return_forecast
        super(RankAndLongShort, self).__init__()

    def get_trades(self, portfolio, t=dt.datetime.today()):
        prediction = values_in_time(self.return_forecast, t)
        sorted_ret = prediction.sort_values()

        short_trades = sorted_ret.index[: self.num_short]
        long_trades = sorted_ret.index[-self.num_long :]

        u = pd.Series(0.0, index=prediction.index)
        u[short_trades] = -1.0
        u[long_trades] = 1.0
        u /= sum(abs(u))
        u = sum(portfolio) * u * self.target_turnover

        # import pdb; pdb.set_trace()
        #
        # # ex-post cash neutrality
        # old_cash = portfolio[-1]
        # if old_cash > 0:
        #     u[short] = u[short] + old_cash/self.num_short
        # else:
        #     u[long] = u[long] + old_cash/self.num_long

        return u


class ProportionalTrade(BasePolicy):
    """Gets to target in given time steps."""

    def __init__(self, targetweight, time_steps):
        self.targetweight = targetweight
        self.time_steps = time_steps
        super(ProportionalTrade, self).__init__()

    def get_trades(self, portfolio, t=dt.datetime.today()):
        try:
            missing_time_steps = len(self.time_steps) - next(i for (i, x) in enumerate(self.time_steps) if x == t)
        except StopIteration:
            raise Exception("ProportionalTrade can only trade on the given time steps")
        deviation = self.targetweight - portfolio / sum(portfolio)
        return sum(portfolio) * deviation / missing_time_steps


class SellAll(BasePolicy):
    """Sell all non-cash assets."""

    def get_trades(self, portfolio, t=dt.datetime.today()):
        trade = -pd.Series(portfolio, copy=True)
        trade.ix[-1] = 0.0
        return trade


class FixedTrade(BasePolicy):
    """Trade a fixed trade vector."""

    def __init__(self, tradevec=None, tradeweight=None):
        """Trade the tradevec vector (dollars) or tradeweight weights."""
        if tradevec is not None and tradeweight is not None:
            raise Exception
        if tradevec is None and tradeweight is None:
            raise Exception
        self.tradevec = tradevec
        self.tradeweight = tradeweight
        assert self.tradevec is None or sum(self.tradevec) == 0.0
        assert self.tradeweight is None or sum(self.tradeweight) == 0.0
        super(FixedTrade, self).__init__()

    def get_trades(self, portfolio, t=dt.datetime.today()):
        if self.tradevec is not None:
            return self.tradevec
        return sum(portfolio) * self.tradeweight


class BaseRebalance(BasePolicy):
    def _rebalance(self, portfolio):
        return sum(portfolio) * self.target - portfolio


class PeriodicRebalance(BaseRebalance):
    """Track a target portfolio, rebalancing at given times."""

    def __init__(self, target, period, **kwargs):
        """
        Args:
            target: target weights, n+1 vector
            period: supported options are "day", "week", "month", "quarter",
                "year".
                rebalance on the first day of each new period
        """
        self.target = target
        self.period = period
        super(PeriodicRebalance, self).__init__()

    def is_start_period(self, t):
        result = not getattr(t, self.period) == getattr(self.last_t, self.period) if hasattr(self, "last_t") else True
        self.last_t = t
        return result

    def get_trades(self, portfolio, t=dt.datetime.today()):
        return self._rebalance(portfolio) if self.is_start_period(t) else self._nulltrade(portfolio)


class AdaptiveRebalance(BaseRebalance):
    """Rebalance portfolio when deviates too far from target."""

    def __init__(self, target, tracking_error):
        self.target = target
        self.tracking_error = tracking_error
        super(AdaptiveRebalance, self).__init__()

    def get_trades(self, portfolio, t=dt.datetime.today()):
        weights = portfolio / sum(portfolio)
        diff = (weights - self.target).values

        if np.linalg.norm(diff, 2) > self.tracking_error:
            return self._rebalance(portfolio)
        else:
            return self._nulltrade(portfolio)


class SinglePeriodOpt(BasePolicy):
    """Single-period optimization policy.

    Implements the model developed in chapter 4 of our paper
    https://stanford.edu/~boyd/papers/cvx_portfolio.html
    """

    def __init__(self, return_forecast, costs, constraints, solver=None, solver_opts=None):

        if not isinstance(return_forecast, BaseReturnsModel):
            null_checker(return_forecast)
        self.return_forecast = return_forecast

        super(SinglePeriodOpt, self).__init__()

        for cost in costs:
            assert isinstance(cost, BaseCost)
            self.costs.append(cost)

        for constraint in constraints:
            assert isinstance(constraint, BaseConstraint)
            self.constraints.append(constraint)

        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts

    def get_trades(self, portfolio, t=None):
        """
        Get optimal trade vector for given portfolio at time t.

        Parameters
        ----------
        portfolio : pd.Series
            Current portfolio vector.
        t : pd.timestamp
            Timestamp for the optimization.
        """

        if t is None:
            t = dt.datetime.today()

        value = sum(portfolio)
        w = portfolio / value
        z = cvx.Variable(w.size)  # TODO pass index
        wplus = w.values + z

        if isinstance(self.return_forecast, BaseReturnsModel):
            alpha_term = self.return_forecast.weight_expr(t, wplus)
        else:
            alpha_term = cvx.sum(cvx.multiply(values_in_time(self.return_forecast, t).values, wplus))

        assert alpha_term.is_concave()

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, wplus, z, value)
            costs.append(cost_expr)
            constraints += const_expr

        constraints += [item for item in (con.weight_expr(t, wplus, z, value) for con in self.constraints)]

        for el in costs:
            assert el.is_convex()

        for el in constraints:
            assert el.is_dcp()

        self.prob = cvx.Problem(cvx.Maximize(alpha_term - sum(costs)), [cvx.sum(z) == 0] + constraints)
        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)

            if self.prob.status == "unbounded":
                logging.error("The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(portfolio)

            if self.prob.status == "infeasible":
                logging.error("The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(portfolio)

            return pd.Series(index=portfolio.index, data=(z.value * value))
        except (cvx.SolverError, TypeError) as e:
            logging.error("The solver %s failed. Defaulting to no trades" % self.solver)
            return self._nulltrade(portfolio)


# class LookaheadModel():
#     """Returns the planning periods for multi-period.
#     """
#     def __init__(self, trading_times, period_lens):
#         self.trading_times = trading_times
#         self.period_lens = period_lens
#
#     def get_periods(self, t):
#         """Returns planning periods.
#         """
#         periods = []
#         tau = t
#         for length in self.period_lens:
#             incr = length*pd.Timedelta('1 days')
#             periods.append((tau, tau + incr))
#             tau += incr
#         return periods


class MultiPeriodOpt(SinglePeriodOpt):
    def __init__(self, trading_times, terminal_weights, lookahead_periods=None, *args, **kwargs):
        """
        trading_times: list, all times at which get_trades will be called
        lookahead_periods: int or None. if None uses all remaining periods
        """
        # Number of periods to look ahead.
        self.lookahead_periods = lookahead_periods
        self.trading_times = trading_times
        # Should there be a constraint that the final portfolio is the bmark?
        self.terminal_weights = terminal_weights
        super(MultiPeriodOpt, self).__init__(*args, **kwargs)

    def get_trades(self, portfolio, t=dt.datetime.today()):

        value = sum(portfolio)
        assert value > 0.0
        w = cvx.Constant(portfolio.values / value)

        prob_arr = []
        z_vars = []

        # planning_periods = self.lookahead_model.get_periods(t)
        for tau in self.trading_times[
            self.trading_times.index(t) : self.trading_times.index(t) + self.lookahead_periods
        ]:
            # delta_t in [pd.Timedelta('%d days' % i) for i in
            # range(self.lookahead_periods)]:

            #            tau = t + delta_t
            z = cvx.Variable(*w.shape)
            wplus = w + z
            obj = self.return_forecast.weight_expr_ahead(t, tau, wplus)

            costs, constr = [], []
            for cost in self.costs:
                cost_expr, const_expr = cost.weight_expr_ahead(t, tau, wplus, z, value)
                costs.append(cost_expr)
                constr += const_expr

            obj -= sum(costs)
            constr += [cvx.sum(z) == 0]
            constr += [con.weight_expr(t, wplus, z, value) for con in self.constraints]

            prob = cvx.Problem(cvx.Maximize(obj), constr)
            prob_arr.append(prob)
            z_vars.append(z)
            w = wplus

        # Terminal constraint.
        if self.terminal_weights is not None:
            prob_arr[-1].constraints += [wplus == self.terminal_weights.values]

        sum(prob_arr).solve(solver=self.solver)
        return pd.Series(index=portfolio.index, data=(z_vars[0].value * value))


class PosTrackingSinglePeriodOpt(BasePolicy):
    """Single-period optimization policy.

    Implements the model developed in chapter 4 of our paper
    https://stanford.edu/~boyd/papers/cvx_portfolio.html
    """

    def __init__(
        self,
        return_forecast,
        returns_index,
        costs,
        constraints,
        solver=None,
        solver_opts=None,
    ):

        if not isinstance(return_forecast, BaseReturnsModel):
            null_checker(return_forecast)
        self.return_forecast = return_forecast
        self.returns_index = returns_index

        super(PosTrackingSinglePeriodOpt, self).__init__()

        for cost in costs:
            assert isinstance(cost, BaseCost)
            self.costs.append(cost)

        for constraint in constraints:
            assert isinstance(constraint, BaseConstraint)
            self.constraints.append(constraint)

        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts

    def get_trades(self, portfolio, t=None):
        """
        Get optimal trade vector for given portfolio at time t.

        Parameters
        ----------
        portfolio : pd.Series
            Current portfolio vector.
        t : pd.timestamp
            Timestamp for the optimization.
        """

        if t is None:
            t = dt.datetime.today()

        value = sum(portfolio)
        w = portfolio / value
        z = cvx.Variable(w.size)  # TODO pass index
        wplus = w.values + z

        if isinstance(self.return_forecast, BaseReturnsModel) and isinstance(self.returns_index, BaseReturnsModel):
            diff = cvx.pos(self.returns_index.weight_expr(t) - self.return_forecast.weight_expr(t, wplus))
            tracking_term = cvx.huber(diff, 0.1)
        else:
            # TODO: Properly implement this if I want
            # diff = self.returns_index[t] - cvx.multiply(self.return_forecast[t], wplus)
            # tracking_term = cvx.sum(cvx.multiply(
            #     values_in_time(self.return_forecast, t).values,
            #     wplus))
            logging.warning("Not implemented see TrackingSinglePeriodOpt.get_trades()")

        assert tracking_term.is_convex()

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, wplus, z, value)
            costs.append(cost_expr)
            constraints += const_expr

        for item in (con.weight_expr(t, wplus, z, value) for con in self.constraints):
            if isinstance(item, list):
                constraints += item
            else:
                constraints += [item]

        for el in costs:
            assert el.is_convex()

        for el in constraints:
            assert el.is_dcp()

        obj = cvx.Minimize(tracking_term + sum(costs))
        self.prob = cvx.Problem(obj, [cvx.sum(z) == 0] + constraints)
        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)
            if self.prob.status == "unbounded":
                logging.error("The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(portfolio)

            if self.prob.status == "infeasible":
                logging.error("The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(portfolio)
            return pd.Series(index=portfolio.index, data=(z.value * value))
        except (cvx.SolverError, TypeError) as e:
            logging.error(e)
            logging.error("The solver %s failed. Defaulting to no trades" % self.solver)
            return self._nulltrade(portfolio)


class QuadTrackingSinglePeriodOpt(BasePolicy):
    """Single-period optimization policy.

    Implements the model developed in chapter 4 of our paper
    https://stanford.edu/~boyd/papers/cvx_portfolio.html
    """

    def __init__(
        self,
        return_forecast,
        index_prices,
        float_shares,
        costs,
        constraints,
        gamma_te,
        Sigma,
        index_weights=None,
        solver=None,
        solver_opts=None,
        index_value=None,
        **kwargs,
    ):

        if not isinstance(return_forecast, BaseReturnsModel):
            null_checker(return_forecast)
        self.return_forecast = return_forecast
        self.index_prices = index_prices
        self.index_value = index_value
        self.float_shares = float_shares
        self.index_weights = index_weights
        self.gamma_te = gamma_te
        self.Sigma = Sigma
        # self.returns_index = returns_index
        # self.index_weights = index_weights
        # self.TE = TE
        # self.Q = Q

        # Add any other keyword args to the class dict
        self.__dict__.update(kwargs)

        super(QuadTrackingSinglePeriodOpt, self).__init__()

        for cost in costs:
            assert isinstance(cost, BaseCost)
            self.costs.append(cost)

        for constraint in constraints:
            assert isinstance(constraint, BaseConstraint)
            self.constraints.append(constraint)

        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts

    def get_trades(self, portfolio, t=None):
        """
        Get optimal trade vector for given portfolio at time t.

        Parameters
        ----------
        portfolio : pd.Series
            Current portfolio vector.
        t : pd.timestamp
            Timestamp for the optimization.
        """

        if t is None:
            t = dt.datetime.today()

        value = sum(portfolio)
        assert value > 0.0
        w = cvx.Constant(portfolio.values / value)
        z = cvx.Variable(w.size)  # TODO pass index
        wplus = w + z
        w_index = self._get_index_weights(t)
        self.w_index = w_index.copy()

        if isinstance(self.return_forecast, BaseReturnsModel):
            #  max mu^T @ (w - w_index)
            # ret = self.gamma_excess * self.return_forecast.weight_expr_ahead(t, tau, wplus, w_index)
            ret = self.gamma_te * cvx.quad_form((wplus - w_index), values_in_time(self.Sigma, t))

        assert ret.is_convex()
        # assert tracking_term.is_convex()

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, wplus, z, value)
            costs.append(cost_expr)
            constraints += const_expr

        for item in (con.weight_expr(t, wplus, z, value) for con in self.constraints):
            if isinstance(item, list):
                constraints += item
            else:
                constraints += [item]

        for el in costs:
            assert el.is_convex()

        for el in constraints:
            assert el.is_dcp()

        obj = cvx.Minimize(ret + sum(costs))
        self.prob = cvx.Problem(obj, [cvx.sum(z) == 0] + constraints)
        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)
            if self.prob.status == "unbounded":
                logging.error("The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(portfolio)

            if self.prob.status == "infeasible":
                logging.error("The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(portfolio)
            return pd.Series(index=portfolio.index, data=(z.value * value))
        except (cvx.SolverError, TypeError) as e:
            logging.error(e)
            logging.error("The solver %s failed. Defaulting to no trades" % self.solver)
            return self._nulltrade(portfolio)

    def _get_index_weights(self, t):
        if self.index_weights is not None:
            try:
                index_weights = self.index_weights.loc[t]
                index_weights["Cash"] = 0
                return index_weights
            except:
                idx = self.index_weights.index.get_loc(t, method="pad")
                index_weights = self.index_weights.iloc[idx]
                t = index_weights.sum()
                index_weights["Cash"] = 0
                return index_weights
        else:
            idx = self.float_shares.index.get_loc(t, method="pad")
            market_cap = self.float_shares.iloc[idx].multiply(values_in_time(self.index_prices, t)).fillna(0)
            if market_cap.ndim > 1:
                raise KeyError(f"Missing index data at {t}")
            index_weights = market_cap / market_cap.sum()
            index_weights["Cash"] = 0
            return index_weights


class QuadTrackingMultiPeriodOpt(QuadTrackingSinglePeriodOpt):
    def __init__(
        self, trading_times: list, terminal_weights: pd.DataFrame, lookahead_periods: int = None, *args, **kwargs
    ):
        """
        trading_times: list, all times at which get_trades will be called
        lookahead_periods: int or None. if None uses all remaining periods
        index_weights: Dataframe or array of weights at each time t, or constant over time
        Q: list of Q matrix for each time step
        """
        # Number of periods to look ahead.
        self.lookahead_periods = lookahead_periods
        self.trading_times = trading_times
        self.terminal_weights = terminal_weights
        self.estimated_index_w = False
        super(QuadTrackingMultiPeriodOpt, self).__init__(*args, **kwargs)

    def get_trades(self, portfolio, t=dt.datetime.today()):

        value = sum(portfolio)
        assert value > 0.0
        w = cvx.Constant(portfolio.values / value)
        try:
            w_index = self._get_index_weights(t)
        except:
            return self._nulltrade(portfolio)

        prob_arr = []
        z_vars = []

        rng = np.random.default_rng()

        # for con in self.constraints:
        #     if isinstance(con, IndexUpdater) and hasattr(con, "is_initiated"):
        #         con._update_required(t)
        index_track = list()
        w_arr = list()
        # planning_periods = self.lookahead_model.get_periods(t)
        for tau in self.trading_times[
            self.trading_times.index(t) : self.trading_times.index(t) + self.lookahead_periods
        ]:
            if tau != t and self.estimated_index_w:
                w_index[:-1] = np.abs(w_index[:-1] + rng.normal(0, 0.01, w_index[:-1].shape))
                w_index = w_index / np.sum(w_index)
            else:
                w_index = self._get_index_weights(tau)
            index_track.append(w_index.copy())
            z = cvx.Variable(*w.shape)
            wplus = w + z

            if isinstance(self.return_forecast, BaseReturnsModel):
                #  max mu^T @ (w - w_index)
                # ret = self.gamma_excess * self.return_forecast.weight_expr_ahead(t, tau, wplus, w_index)
                ret = self.gamma_te * cvx.quad_form((wplus - w_index), values_in_time(self.Sigma, t))

            assert ret.is_convex()
            # assert tracking_term.is_convex()

            costs, constraints = [], []

            for cost in self.costs:
                cost_expr, const_expr = cost.weight_expr_ahead(t, tau, wplus, z, value)
                costs.append(cost_expr)
                constraints += const_expr

            for item in (con.weight_expr(t, wplus, z, value) for con in self.constraints):
                if isinstance(item, list):
                    constraints += item
                else:
                    constraints += [item]

            for el in costs:
                assert el.is_convex()

            for el in constraints:
                assert el.is_dcp()

            obj = cvx.Minimize(ret + sum(costs))
            prob = cvx.Problem(obj, [cvx.sum(z) == 0] + constraints)
            prob_arr.append(prob)
            z_vars.append(z)
            w_arr.append(wplus)
            w = wplus

        # Terminal constraint.
        if self.terminal_weights is not None:
            prob_arr[-1].constraints += [wplus == self.terminal_weights.values]

        # We are summing all problems in order to obtain overall objective
        self.prob = sum(prob_arr)
        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)
            # temp3 = self.index_weights.loc[t]
            # temp2 = np.abs(index_track[0] - self.index_weights.loc[t])
            temp = np.abs(self.optToArr(w_arr) - index_track)
            if self.prob.status == "unbounded":
                logging.error("The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(portfolio)

            if self.prob.status == "infeasible":
                logging.error("The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(portfolio)

            # for con in prob_arr[0].constraints:
            #     if con.id == self.te_const_id:

            return pd.Series(index=portfolio.index, data=(z_vars[0].value * value))
        except (cvx.SolverError, TypeError) as e:
            logging.error(e)
            logging.error("The solver %s failed. Defaulting to no trades" % self.solver)
            return self._nulltrade(portfolio)

    def optToArr(self, l):
        arr_l = list()
        for el in l:
            arr_l.append(el.value)
        return np.array(arr_l)


class CardinalitySPO(BasePolicy):
    def __init__(
        self,
        return_forecast,
        index_prices,
        float_shares,
        costs,
        constraints,
        gamma_excess,
        card,
        Sigma,
        index_weights=None,
        gamma_te=1,
        thresh=1e-5,
        max_iter=10,
        solver=None,
        solver_opts=None,
        index_value=None,
        **kwargs,
    ):

        if not isinstance(return_forecast, BaseReturnsModel):
            null_checker(return_forecast)
        self.return_forecast = return_forecast
        self.index_prices = index_prices
        self.index_value = index_value
        self.float_shares = float_shares
        self.gamma_excess = gamma_excess
        self.index_weights = index_weights
        self.card = card
        self.THRESH = thresh
        self.MAX_ITER = max_iter
        self.gamma_te = gamma_te
        self.Sigma = Sigma
        super().__init__()

        for cost in costs:
            assert isinstance(cost, BaseCost)
            self.costs.append(cost)

        for constraint in constraints:
            assert isinstance(constraint, BaseConstraint)
            self.constraints.append(constraint)

        assert isinstance(self.return_forecast, BaseReturnsModel), "return forecast must be of type BaseReturnModel"

        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts

    def get_trades(self, portfolio, t=None):
        """
        Get optimal trade vector for given portfolio at time t.

        Parameters
        ----------
        portfolio : pd.Series
            Current portfolio vector.
        t : pd.timestamp
            Timestamp for the optimization.
        """

        if t is None:
            t = dt.datetime.today()

        self.portfolio = portfolio
        self.value = sum(self.portfolio)
        assert self.value > 0.0
        self.w = cvx.Constant(self.portfolio.values / self.value)
        self.w_index = self._get_index_weights(t)
        if self.method == "PADM":
            z = self.PADM(t)
        elif self.method == "ADM":
            z = self.ADM(t)
        elif self.method == "NCVX":
            z = self.NCVX(t)

        return pd.Series(index=portfolio.index, data=(z * self.value))

    def f():
        return NotImplemented

    def g():
        return NotImplemented

    def _get_index_weights(self, t):
        if self.index_weights is not None:
            try:
                index_weights = self.index_weights.loc[t]
                index_weights["Cash"] = 0
                return index_weights
            except:
                idx = self.index_weights.index.get_loc(t, method="pad")
                index_weights = self.index_weights.iloc[idx]
                t = index_weights.sum()
                index_weights["Cash"] = 0
                return index_weights
        else:
            idx = self.float_shares.index.get_loc(t, method="pad")
            market_cap = self.float_shares.iloc[idx].multiply(values_in_time(self.index_prices, t)).fillna(0)
            if market_cap.ndim > 1:
                raise KeyError(f"Missing index data at {t}")
            index_weights = market_cap / market_cap.sum()
            index_weights["Cash"] = 0
            return index_weights

    # def _get_index_weights(self, t):
    #     market_cap = self.float_shares["float_shares"].multiply(values_in_time(self.index_prices, t)).fillna(0)
    #     index_weights = market_cap / market_cap.sum()
    #     index_weights["Cash"] = 0
    #     return index_weights

    def _rand_gen(self, size: tuple):
        # generator
        rng = np.random.default_rng()
        arr = rng.random(size)
        # Set the first (total - card) elements to 0
        arr[: -self.card] = 0
        # Normalize so it sums to 1
        arr = arr / arr.sum()
        # Shuffle vector around
        rng.shuffle(arr)
        return arr

    def _hasImproved(self, prev, curr, thresh):
        print(np.linalg.norm((prev - curr), 2) ** 2)
        return np.linalg.norm((prev - curr), 2) ** 2 < thresh

    def _distance(self, prev, curr):
        return np.linalg.norm((prev - curr), 2) ** 2


class PADMCardinalitySPO(CardinalitySPO):
    """This is a crude implementation of PADM - see PADM-Cardinality Constrained Portfolio[73]"""

    def __init__(self, mu=1e-4, **kwargs):
        self.method = "PADM"
        self.mu_start = mu
        super().__init__(**kwargs)

    def PADM(self, t):
        # y_next = self.w
        y_next = self._rand_gen(self.w.size)
        iteration = 0
        mu = self.mu_start
        while True:
            w_next = self.f(y_next, mu, t)
            y_next = self.g(w_next)

            # Check stopping criteria
            if iteration == 0:
                # diff = np.linalg.norm((w_next - y_next), 1)
                diff = self._distance(w_next, y_next)
            else:
                diff_last = diff
                # diff = np.linalg.norm((w_next - y_next), 1)
                diff = self._distance(w_next, y_next)
                diff_calc = np.abs(diff_last - diff)
                # if self._distance(diff_last, diff) < self.THRESH:
                if diff_calc < self.THRESH:
                    z = y_next - self.w.value
                    return z

                if iteration >= self.MAX_ITER:
                    print("Max iter reached - not optimal!")
                    z = y_next - self.w.value
                    return z

            mu *= 10
            iteration += 1

    def f(self, y, mu, t):
        z = cvx.Variable(self.w.size)
        wplus = self.w + z

        # Objective Function
        # ret = -self.gamma_excess * self.return_forecast.weight_expr(t, wplus=w_next, w_index=self.w_index)
        ret = self.gamma_te * cvx.quad_form((wplus - self.w_index), values_in_time(self.Sigma, t))

        # l1 penalty term
        ret += mu * cvx.norm(wplus - y, 1)

        # assert tracking_term.is_convex()
        assert ret.is_convex()

        # Additional Costs & Constraints
        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, wplus, z, self.value)
            costs.append(cost_expr)
            constraints += const_expr

        for item in (con.weight_expr(t, wplus, z, self.value) for con in self.constraints):
            if isinstance(item, list):
                constraints += item
            else:
                constraints += [item]

        for el in costs:
            assert el.is_convex()

        for el in constraints:
            assert el.is_dcp()

        obj = cvx.Minimize(ret + sum(costs))
        prob = cvx.Problem(obj, [cvx.sum(z) == 0] + constraints)
        try:
            prob.solve(solver=self.solver, **self.solver_opts)
            if prob.status == "unbounded":
                logging.error("The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(self.portfolio)

            if prob.status == "infeasible":
                logging.error("The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(self.portfolio)
            return wplus.value
        except Exception as e:
            print(e)
            print("Error solving f() part of the problem")

    def g(self, w_next):
        i_s = np.argsort(w_next)[::-1][: self.card]
        w_s = w_next[i_s]
        return np.where(np.isin(w_next, w_s), w_next / np.sum(w_s), 0)


class PADMCardinalityMPO(PADMCardinalitySPO):
    def __init__(
        self, trading_times: list, terminal_weights: pd.DataFrame, lookahead_periods: int = None, *args, **kwargs
    ):
        """
        trading_times: list, all times at which get_trades will be called
        lookahead_periods: int or None. if None uses all remaining periods
        index_weights: Dataframe or array of weights at each time t, or constant over time
        Q: list of Q matrix for each time step
        """
        # Number of periods to look ahead.
        self.lookahead_periods = lookahead_periods
        self.trading_times = trading_times
        self.terminal_weights = terminal_weights
        self.estimated_index_w = True
        super().__init__(*args, **kwargs)

    def get_trades(self, portfolio, t=dt.datetime.today()):

        self.portfolio = portfolio
        self.value = sum(self.portfolio)
        self.w_index = self._get_index_weights(t)
        assert self.value > 0.0
        self.w = self.portfolio.values / self.value

        # planning_periods = self.lookahead_model.get_periods(t)
        tau_times = self.trading_times[
            self.trading_times.index(t) : self.trading_times.index(t) + self.lookahead_periods
        ]

        iteration = 0
        mu = self.mu_start
        y = np.array([self.w] * 3)
        while True:
            # W incorporates return and TE and outter w_plus
            w = self.f(self.w, y, mu, t, tau_times)
            # y incorporates cardinality constraint
            y = self.g(w)

            # Check stopping criteria
            if iteration == 0:
                diff = self._distance(w, y)
            else:
                diff_last = diff
                diff = self._distance(w, y)
                diff_calc = np.abs(diff_last - diff)
                if diff_calc < self.THRESH:
                    break

                if iteration >= self.MAX_ITER:
                    logging.info("Max iter reached - not optimal!")
                    break

            mu *= 10
            iteration += 1

        z = y[0] - self.w
        return pd.Series(index=portfolio.index, data=(z * self.value))

    def f(self, w_init, y, mu, t, tau_times):
        prob_arr = []
        z_vars = []
        w_vars = []
        rng = np.random.default_rng()

        for i, tau in enumerate(tau_times):
            if tau != t and self.estimated_index_w:
                w_index = np.abs(self.w_index + rng.normal(0, 1, self.w_index.shape))
                w_index = w_index / np.sum(w_index)
            else:
                w_index = self.w_index
            z = cvx.Variable(w_init.size)
            wplus = w_init + z

            # Objective Function
            # ret = -self.gamma_excess * self.return_forecast.weight_expr_ahead(
            #     t, tau, wplus=wplus, w_index=self.w_index
            # )

            # ret = -self.gamma_excess * self.return_forecast.weight_expr_ahead(t, tau, wplus=wplus, w_index=w_index)
            ret = self.gamma_te * cvx.quad_form((wplus - w_index), values_in_time(self.Sigma, t))

            # l1 penalty term
            ret += mu * cvx.norm(wplus - y[i], 1)

            # assert tracking_term.is_convex()
            assert ret.is_convex()

            # Additional Costs & Constraints
            costs, constraints = [], []

            for cost in self.costs:
                cost_expr, const_expr = cost.weight_expr(t, wplus, z, self.value)
                costs.append(cost_expr)
                constraints += const_expr

            for item in (con.weight_expr(t, wplus, z, self.value) for con in self.constraints):
                if isinstance(item, list):
                    constraints += item
                else:
                    constraints += [item]

            for el in costs:
                assert el.is_convex()

            for el in constraints:
                assert el.is_dcp()

            # obj = cvx.Minimize(tracking_term + sum(costs))
            obj = cvx.Minimize(ret + sum(costs))
            prob = cvx.Problem(obj, [cvx.sum(z) == 0] + constraints)
            prob_arr.append(prob)
            z_vars.append(z)
            w_vars.append(wplus)
            w = wplus

        # Terminal constraint.
        if self.terminal_weights is not None:
            prob_arr[-1].constraints += [wplus == self.terminal_weights.values]

        # We are summing all problems in order to obtain overall objective
        self.prob = sum(prob_arr)
        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)
            if self.prob.status == "unbounded":
                logging.error("The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(self.portfolio)

            if self.prob.status == "infeasible":
                logging.error("The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(self.portfolio)

            return self.optToArr(w_vars)
        except (cvx.SolverError, TypeError) as e:
            logging.error(e)
            logging.error("The solver %s failed. Defaulting to no trades" % self.solver)
            return self._nulltrade(self.portfolio)

    def g(self, w_arr):
        y_arr = np.zeros(w_arr.shape)

        def get_prox(w):
            i_s = np.argsort(w)[::-1][: self.card]
            w_s = w[i_s]
            return np.where(np.isin(w, w_s), w / np.sum(w_s), 0)

        for i, row in enumerate(w_arr):
            y_arr[i] = get_prox(row)

        return y_arr

    def optToArr(self, l):
        arr_l = list()
        for el in l:
            arr_l.append(el.value)
        return np.array(arr_l)


class ADMCardinalitySPO(CardinalitySPO):
    """This is the implementation of ADMM - see A Novel Approach for Solving Convex Problems with Cardinality Constraints,
    prox_algs (boyd).
    """

    def __init__(self, mu=1e-4, **kwargs):
        self.method = "ADM"
        self.mu_start = mu
        self.h = 1e-4
        self.hard = False
        super().__init__(**kwargs)

    def ADM(self, t):
        # Initialize y_next
        y_next = self._rand_gen(self.w.size)
        iteration = 0
        u = 0
        gamma = 1

        while True:
            w_next, z_w = self.f(y_next - u, t)
            y_next, z_y = self.g(w_next + u, gamma)

            # Check stopping criteria
            if iteration == 0:
                # diff = np.linalg.norm((w_next - y_next), 1)
                diff = self._distance(w_next, y_next)
            else:
                diff_last = diff
                # diff = np.linalg.norm((w_next - y_next), 1)
                diff = self._distance(w_next, y_next)
                diff_calc = np.abs(diff_last - diff)
                # if self._distance(diff_last, diff) < self.THRESH:
                if diff_calc < self.THRESH:
                    # z = y_next - self.w.value
                    return z_y

                if iteration >= self.MAX_ITER:
                    print("Max iter reached - not optimal!")
                    # z = y_next - self.w.value
                    return z_y

            # Update
            gamma = self._max_grad_back(w_next, z_w, t, self.h)
            u = u + w_next - y_next
            iteration += 1

    def eval_f(self, w, z, t):
        ret = -self.gamma_excess * self.return_forecast.weight_expr(t, wplus=w, w_index=self.w_index)
        costs = []
        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, w, z, self.value)
            costs.append(cost_expr)

        temp = ret + sum(costs)
        return ret + sum(costs)

    def f(self, y, t):
        z = cvx.Variable(self.w.size)
        w_next = self.w + z

        # Objective Function
        ret = -self.gamma_excess * self.return_forecast.weight_expr(t, wplus=w_next, w_index=self.w_index)

        # # l1 penalty term
        # ret += mu * cvx.norm(w_next - y, 1)

        # Proximality term 1/2||x-z||^2
        gamma = 1  # TBD if I want to change this
        ret += (1 / (2 * gamma)) * cvx.square(cvx.norm(w_next - y, 2))

        # assert tracking_term.is_convex()
        assert ret.is_convex()

        # Additional Costs & Constraints
        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, w_next, z, self.value)
            costs.append(cost_expr)
            constraints += const_expr

        for item in (con.weight_expr(t, w_next, z, self.value) for con in self.constraints):
            if isinstance(item, list):
                constraints += item
            else:
                constraints += [item]

        for el in costs:
            assert el.is_convex()

        for el in constraints:
            assert el.is_dcp()

        obj = cvx.Minimize(ret + sum(costs))
        prob = cvx.Problem(obj, [cvx.sum(z) == 0] + constraints)
        try:
            prob.solve(solver=self.solver, **self.solver_opts)
            if prob.status == "unbounded":
                logging.error("The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(self.portfolio)

            if prob.status == "infeasible":
                logging.error("The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(self.portfolio)
            return w_next.value, z.value
        except Exception as e:
            print(e)
            print("Error solving f() part of the problem")

    def g(self, w_next, gamma):
        """Proximal operator for I_B(x) + gamma * phi_k(x)
        This function requires the calculation of the gradient TODO: Will implement at a later time to compare

        phi_k(x) : ||x||_1 - ||x||_k, i.e., the difference between the sum of elements of x and the sum of the k largest elements
        """
        # Set all negative elements to 0.
        # w_next = np.where(w_next <0, 0, w_next)
        # Sorting incoming vector
        i_s = np.argsort(w_next)[::-1]

        # kappa = 1  # Can change this later to be = max(gradient(f)) - see paper by gonjac

        def prox_op(val, idx):
            if self.hard == True:
                # A version of hard thresholding where sqrt(2rho) = first el that exceeds card
                # Note this is effectively the same as PADM method
                for init_i, sorted_i in enumerate(idx):
                    if init_i > self.card:
                        val[sorted_i] = 0
                    else:
                        continue
            else:
                for init_i, sorted_i in enumerate(idx):
                    if init_i > self.card:
                        val[sorted_i] = self._saturation(self._soft_thresh(val[sorted_i], gamma), 0, 1)
                    else:
                        val[sorted_i] = self._saturation(val[sorted_i], 0, 1)

        prox_op(w_next, i_s)
        # re-weight
        w_next = w_next / w_next.sum()
        z = w_next - self.w.value
        return w_next, z

    # def _hard_thresh(self, val, thresh):
    #     # Not implemented
    #     if val < thresh:
    #         return 0
    #     else:
    #         return val

    def _soft_thresh(self, val, kappa):
        if val < -kappa:
            return val + kappa
        elif val > kappa:
            return val - kappa
        else:
            return 0

    def _saturation(self, val, l, u):
        if val < l:
            return l
        elif val > u:
            return u
        else:
            return val

    # a function that returns one standard basis function
    def _basis_vec(self, ndim, index):
        v = np.zeros(ndim)
        v[index] = 1.0
        return v

    def _max_grad_back(self, w, z, t, h):
        h_ndim = w.size
        return np.max(
            np.abs(
                np.array(
                    [
                        (self.eval_f(w, z, t).value - self.eval_f(w - h * self._basis_vec(h_ndim, i), z, t).value) / h
                        for i in range(h_ndim)
                    ]
                )
            )
        )


class ADMCardinalityMPO(ADMCardinalitySPO):
    def __init__(
        self, trading_times: list, terminal_weights: pd.DataFrame, lookahead_periods: int = None, *args, **kwargs
    ):
        """
        trading_times: list, all times at which get_trades will be called
        lookahead_periods: int or None. if None uses all remaining periods
        index_weights: Dataframe or array of weights at each time t, or constant over time
        Q: list of Q matrix for each time step
        """
        # Number of periods to look ahead.
        self.lookahead_periods = lookahead_periods
        self.trading_times = trading_times
        self.terminal_weights = terminal_weights
        super().__init__(*args, **kwargs)

    def get_trades(self, portfolio, t=dt.datetime.today()):

        self.portfolio = portfolio
        self.value = sum(self.portfolio)
        self.w_index = self._get_index_weights(t)
        assert self.value > 0.0
        self.w = self.portfolio.values / self.value

        # planning_periods = self.lookahead_model.get_periods(t)
        tau_times = self.trading_times[
            self.trading_times.index(t) : self.trading_times.index(t) + self.lookahead_periods
        ]
        # Skeleton
        # n: number of assets
        # T: lookaheads periods
        # z_next = cvx.Variable((len(tau_times), self.w.size), value=np.zeros((len(tau_times), self.w.size))).value
        z_next = np.zeros((len(tau_times), self.w.size))
        w_next_cost = np.zeros((len(tau_times), self.w.size))
        gamma = 1
        u = np.zeros((len(tau_times), self.w.size))
        while True:
            # w_prev = w_next.copy()
            # z_prev = z_next.copy()

            # w_next opt:
            num_workers = min(multiprocess.cpu_count(), len(tau_times))
            num_workers = 1
            workers = multiprocess.Pool(num_workers)
            w_next_cost = w_next_cost - u
            w_rows = [self.w] + [row for row in w_next_cost[:-1]]
            z_rows = [row for row in z_next]
            zipped_args = zip(w_rows, z_rows, repeat(gamma), repeat(t), tau_times)
            w_next_stage = np.array(workers.starmap(self.PADM, zipped_args))

            # z_next opt:
            w_next_stage = w_next_stage + u
            num_workers = min(multiprocess.cpu_count(), w_next_stage.shape[1])
            workers = multiprocess.Pool(num_workers)
            w_init_cols = [asset for asset in self.w]
            w_next_cols = [row for row in w_next_stage.T]
            zipped_args = zip(w_init_cols, w_next_cols, repeat(gamma), repeat(tau_times))
            for i, el in enumerate(zipped_args):
                w_next_cost[:, i] = self.prox_schedule(*el)
            # w_next_cost = np.array(workers.starmap(self.prox_schedule, zipped_args)).T

            if self._hasImproved(w_next_stage, w_next_cost, self.THRESH):
                break

            # Update error term
            # u = u + w_next_stage - w_next_cost

        w = np.reshape(self.w, (-1, self.w.size))
        z_next = np.diff(w_next_stage, axis=0, prepend=w)
        return pd.Series(index=portfolio.index, data=(z_next[0, :] * self.value))

        # z = cvx.Variable(*w.shape)
        # wplus = w + z

    def PADM(self, w_prev, z_next, gamma, t, tau):
        if not isinstance(w_prev, np.ndarray):
            try:
                w_prev = np.array(w_prev)
            except:
                raise ValueError("Did not pass ndarray or list for w_prev")
        if not isinstance(z_next, np.ndarray):
            try:
                z_next = np.array(z_next)
            except:
                raise ValueError("Did not pass ndarray or list for z_next")

        # I cannot use tau here unless I use my return estimates to determine weights.. will add noise
        w_index = self._get_index_weights(tau)

        # z comes from outter ADMM
        w_outter = w_prev + z_next
        iteration = 0
        mu = self.mu_start
        y = w_prev
        while True:
            # W incorporates return and TE and outter w_plus
            w = self.f(y, w_outter, mu, w_index, gamma, t, tau)
            # y incorporates cardinality constraint
            y = self.g(w)

            # Check stopping criteria
            if iteration == 0:
                diff = self._distance(w, y)
            else:
                diff_last = diff
                diff = self._distance(w, y)
                diff_calc = np.abs(diff_last - diff)
                if diff_calc < self.THRESH:
                    # z = y_next - self.w.value
                    return y

                if iteration >= self.MAX_ITER:
                    logging.info("Max iter reached - not optimal!")
                    # z = y_next - self.w.value
                    return y

            mu *= 10
            iteration += 1

    def f(self, y, w_outter, mu, w_index, gamma, t, tau):
        # z = cvx.Variable(self.w.size)
        # w_next = self.w + z
        w = cvx.Variable(self.w.size)

        # Objective Function
        ret = -self.gamma_excess * self.return_forecast.weight_expr_ahead(t, tau, wplus=w, w_index=w_index)

        # l1 penalty term
        ret += mu * cvx.norm(w - y, 1)

        # Proximal regularization:
        ret += (1 / (2 * gamma)) * cvx.square(cvx.norm(w - w_outter, 2))

        # assert tracking_term.is_convex()
        assert ret.is_convex()

        # Additional Costs & Constraints
        costs, constraints = [], []

        for cost in self.costs:
            if isinstance(cost, TcostModel):
                continue
            cost_expr, const_expr = cost.weight_expr(t, w, z=None, value=self.value)
            costs.append(cost_expr)
            constraints += const_expr

        for item in (con.weight_expr(t, w, None, self.value) for con in self.constraints):
            if isinstance(item, list):
                constraints += item
            else:
                constraints += [item]

        for el in costs:
            assert el.is_convex()

        for el in constraints:
            assert el.is_dcp()

        obj = cvx.Minimize(ret + sum(costs))
        prob = cvx.Problem(obj, [sum(w) == 1] + constraints)
        try:
            prob.solve(solver=self.solver, **self.solver_opts)
            if prob.status == "unbounded":
                logging.error("The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(self.portfolio)

            if prob.status == "infeasible":
                logging.error("The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(self.portfolio)
            return w.value
        except Exception as e:
            print(e)
            print("Error solving f() part of the problem")

    def g(self, w):
        i_s = np.argsort(w)[::-1][: self.card]
        w_s = w[i_s]
        return np.where(np.isin(w, w_s), w / np.sum(w_s), 0)

    def prox_schedule(self, w_init, w_asset_i, gamma, tau_times):
        w_z = cvx.Variable(len(w_asset_i))
        for cost in self.costs:
            if isinstance(cost, TcostModel):
                func = cost
                # obj = cost_expr
        obj = 0
        # sum over time periods t = 0 to t + T
        for i, t in enumerate(tau_times):

            # g(x(t) - x(t-1))
            if i == 0:
                obj += func.weight_expr(t=t, w_plus=w_z[i], z=w_z[i] - w_init, value=self.value)[0]
            else:
                obj += func.weight_expr(t=t, w_plus=w_z[i], z=w_z[i] - w_z[i - 1], value=self.value)[0]

            # proximal regularization:
            gamma = 1  # TBD if I want to change this
            obj += (1 / (2 * gamma)) * cvx.square(cvx.norm(w_z[i] - w_asset_i[i], 2))

        prob = cvx.Problem(cvx.Minimize(obj))
        prob.solve(solver=self.solver, **self.solver_opts)

        return w_z.value


class NCVXCardinalitySPO(CardinalitySPO):
    # Does not work at the moment
    def __init__(self, **kwargs):
        self.method = "NCVX"
        super().__init__(**kwargs)

    def NCVX(self, t):
        # z = cvx.Variable(self.w.size)
        w_next = nc.Card(self.w.size, self.card, 1).flatten()
        z = w_next - self.w

        # Objective Function
        ret = -self.gamma_excess * self.return_forecast.weight_expr(t, wplus=w_next, w_index=self.w_index)

        # assert tracking_term.is_convex()
        assert ret.is_convex()

        # Additional Costs & Constraints
        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, w_next, z, self.value)
            costs.append(cost_expr)
            constraints += const_expr

        for item in (con.weight_expr(t, w_next, z, self.value) for con in self.constraints):
            if isinstance(item, list):
                constraints += item
            else:
                constraints += [item]

        for el in costs:
            assert el.is_convex()

        for el in constraints:
            assert el.is_dcp()

        obj = cvx.Minimize(ret + sum(costs))
        prob = cvx.Problem(obj, [cvx.sum(w_next) == 1] + constraints)
        try:
            # prob.solve(solver=self.solver, **self.solver_opts)
            prob.solve(method="NC-ADMM")
            if prob.status == "unbounded":
                logging.error("The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(self.portfolio)

            if prob.status == "infeasible":
                logging.error("The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(self.portfolio)
            return z.value
        except Exception as e:
            print(e)
            print("Error solving f() part of the problem")
