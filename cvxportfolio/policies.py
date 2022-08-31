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
import datetime as dt
from inspect import Parameter
import pandas as pd
import numpy as np
import logging
import cvxpy as cvx
import ncvx as nc

# import numdifftools as nd
import traceback
import sys

from .costs import BaseCost
from .returns import BaseReturnsModel
from .constraints import BaseConstraint, TrackingErrorMax, IndexUpdater
from .utils import values_in_time, null_checker


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
    "FastADMCardinalitySPO",
    "NCVXCardinalitySPO",
]


class BasePolicy(object, metaclass=ABCMeta):
    """Base class for a trading policy."""

    def __init__(self):
        self.costs = []
        self.constraints = []

    @abstractmethod
    def get_trades(self, portfolio, t=dt.datetime.today()):
        """Trades list given current portfolio and time t."""
        return NotImplemented

    def _nulltrade(self, portfolio):
        return pd.Series(index=portfolio.index, data=0.0)

    def get_rounded_trades(self, portfolio, prices, t):
        """Get trades vector as number of shares, rounded to integers."""
        return np.round(self.get_trades(portfolio, t) / values_in_time(prices, t))[:-1]


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
        # returns_index,
        # index_weights,
        # Q,
        # TE,
        index_prices,
        float_shares,
        costs,
        constraints,
        gamma_excess,
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
        self.w_index = w_index

        if isinstance(self.return_forecast, BaseReturnsModel):
            # diff = cvx.norm(
            #     self.returns_index.weight_expr(t)
            #     - self.return_forecast.weight_expr(t, wplus),
            #     2,
            # )
            # tracking_term = cvx.huber(diff, 0.1)
            # tracking_term = diff
            ret = self.gamma_excess * self.return_forecast.weight_expr(t, wplus=wplus, w_index=w_index)
        else:
            # TODO: Properly implement this if I want
            # diff = self.returns_index[t] - cvx.multiply(self.return_forecast[t], wplus)
            # tracking_term = cvx.sum(cvx.multiply(
            #     values_in_time(self.return_forecast, t).values,
            #     wplus))
            logging.warning("Not implemented see TrackingSinglePeriodOpt.get_trades()")

        # assert tracking_term.is_convex()
        assert ret.is_convex()

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

        # obj = cvx.Minimize(tracking_term + sum(costs))
        obj = cvx.Minimize(-ret + sum(costs))
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
        market_cap = self.float_shares["float_shares"].multiply(values_in_time(self.index_prices, t)).fillna(0)
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
        super(QuadTrackingMultiPeriodOpt, self).__init__(*args, **kwargs)

    # TODO: Check if general function is better
    # def add_robust(self, box:bool=False, elipsoidal:bool=False, *args, **kwargs):
    #     if box:
    #         # check presence of necesary args
    #         kwords = ["ret_hat", "lambda_robust"]
    #         for el in kwords:
    #             if el not in kwargs:
    #                 print(f"Missing argument {el}")

    def add_robust(self, sigma, gamma):
        self.sigma = sigma
        self.gamma_robust = gamma

    def get_trades(self, portfolio, t=dt.datetime.today()):

        value = sum(portfolio)
        assert value > 0.0
        w = cvx.Constant(portfolio.values / value)
        w_index = self._get_index_weights(t)
        self.w_index = w_index  # used for reporting

        prob_arr = []
        z_vars = []

        # for con in self.constraints:
        #     if isinstance(con, IndexUpdater) and hasattr(con, "is_initiated"):
        #         con._update_required(t)

        # planning_periods = self.lookahead_model.get_periods(t)
        for tau in self.trading_times[
            self.trading_times.index(t) : self.trading_times.index(t) + self.lookahead_periods
        ]:
            # delta_t in [pd.Timedelta('%d days' % i) for i in
            # range(self.lookahead_periods)]:
            #            tau = t + delta_t
            z = cvx.Variable(*w.shape)
            wplus = w + z

            if isinstance(self.return_forecast, BaseReturnsModel):
                # diff = cvx.norm(
                #     self.returns_index.weight_expr(t)
                #     - self.return_forecast.weight_expr_ahead(t, tau, wplus),
                #     2,
                # )

                #  max mu^T @ (w - w_index)
                ret = self.gamma_excess * self.return_forecast.weight_expr_ahead(t, tau, wplus, w_index)
                # tracking_term = cvx.huber(diff, 0.1)
                # tracking_term = self.gamma_excess * diff
                # tracking_error = (wplus - self.index_weights) @ self.Q.weight_expr_ahead(t, tau) @ (wplus - self.index_weights)
            else:
                # TODO: Properly implement this if I want
                # diff = self.returns_index[t] - cvx.multiply(self.return_forecast[t], wplus)
                # tracking_term = cvx.sum(cvx.multiply(
                #     values_in_time(self.return_forecast, t).values,
                #     wplus))
                logging.warning("Not implemented see TrackingSinglePeriodOpt.get_trades()")

            if hasattr(self, "gamma_robust"):
                # Required portfolio robustness
                var_matr = np.diag(np.diag(values_in_time(self.sigma, tau)))
                # # Target portfolio return estimation error (r.e.e. bound)
                # # rob_bnd = np.dot(w0, np.dot(var_matr, w0))
                # np.dot(w0, np.dot(var_matr, w0))
                # var_minVar = np.dot(w_minVar, np.dot(Q, w_minVar))
                # ret_minVar = np.dot(mu, w_minVar)
                # rob_minVar = np.dot(w_minVar, np.dot(var_matr, w_minVar))

                # Portf_Retn = np.dot(mu, w1.value)  # Estimated returns from minVar
                # Portf_Retn = ret_minVar * 10
                # Qq_rMV = var_matr
                temp = self.gamma_robust * cvx.quad_form(wplus, var_matr)
                ret = ret - temp
            else:
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

            # obj = cvx.Minimize(tracking_term + sum(costs))
            obj = cvx.Minimize(-ret + sum(costs))
            prob = cvx.Problem(obj, [cvx.sum(z) == 0] + constraints)
            prob_arr.append(prob)
            z_vars.append(z)
            w = wplus

            # Using this for troubleshooting/logging
            # self.index_obj = self.returns_index.weight_expr(t)
            # self.portfolio_obj = self.return_forecast.weight_expr_ahead(t, tau, wplus)
            # self.diff_obj = tracking_term
            # try:
            #     self.te = costs[2].args[1]
            # except:
            #     self.te = constraints[2].args[0]
        # Terminal constraint.
        if self.terminal_weights is not None:
            prob_arr[-1].constraints += [wplus == self.terminal_weights.values]

        # We are summing all problems in order to obtain overall objective
        self.prob = sum(prob_arr)
        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)
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
        thresh=1e-6,
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
        self.card = card
        self.THRESH = thresh
        self.MAX_ITER = max_iter

        # Add any other keyword args to the class dict
        self.__dict__.update(kwargs)

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
        elif self.method == "FastADM":
            z = self.FastADM(t)
        elif self.method == "NCVX":
            z = self.NCVX(t)

        return pd.Series(index=portfolio.index, data=(z * self.value))

    def f():
        return NotImplemented

    def g():
        return NotImplemented

    def _get_index_weights(self, t):
        market_cap = self.float_shares["float_shares"].multiply(values_in_time(self.index_prices, t)).fillna(0)
        index_weights = market_cap / market_cap.sum()
        index_weights["Cash"] = 0
        return index_weights


class PADMCardinalitySPO(CardinalitySPO):
    """This is a crude implementation of PADM - see PADM-Cardinality Constrained Portfolio[73]"""

    def __init__(self, mu=1e-4, **kwargs):
        self.method = "PADM"
        self.mu_start = mu
        super().__init__(**kwargs)

    def PADM(self, t):
        y_next = self.w
        iteration = 0
        mu = self.mu_start
        while True:
            w_next = self.f(y_next, mu, t)
            y_next = self.g(w_next)

            # Check stopping criteria
            if iteration == 0:
                diff = np.linalg.norm((w_next - y_next), 1)
            else:
                diff_last = diff
                diff = np.linalg.norm((w_next - y_next), 1)

                if (diff_last - diff) ** 2 < self.THRESH:
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
        w_next = self.w + z

        # Objective Function
        ret = -self.gamma_excess * self.return_forecast.weight_expr(t, wplus=w_next, w_index=self.w_index)

        # l1 penalty term
        ret += mu * cvx.norm(w_next - y, 1)

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
            return w_next.value
        except Exception as e:
            print(e)
            print("Error solving f() part of the problem")

    def g(self, w_next):
        i_s = np.argsort(w_next)[::-1][: self.card]
        w_s = w_next[i_s]
        return np.where(np.isin(w_next, w_s), w_next / np.sum(w_s), 0)


class FastADMCardinalitySPO(CardinalitySPO):
    """This is the implementation of ADMM - see A Novel Approach for Solving Convex Problems with Cardinality Constraints,
    prox_algs (boyd).
    """

    def __init__(self, mu=1e-4, **kwargs):
        self.method = "FastADM"
        self.mu_start = mu
        super().__init__(**kwargs)

    def FastADM(self, t):
        # Initialize y_next
        y_next = self._rand_gen(self.w.size)
        iteration = 0
        u = 0
        mu = self.mu_start
        while True:
            w_next = self.f(y_next - u, mu, t)
            y_next = self.g(w_next + u)

            # Check stopping criteria
            if iteration == 0:
                diff = np.linalg.norm((w_next - y_next), 1)
            else:
                diff_last = diff
                diff = np.linalg.norm((w_next - y_next), 1)

                if (diff_last - diff) ** 2 < self.THRESH:
                    z = y_next - self.w.value
                    return z

                if iteration >= self.MAX_ITER:
                    print("Max iter reached - not optimal!")
                    z = y_next - self.w.value
                    return z

            # Update
            mu *= 1.1
            u = u + w_next - y_next
            iteration += 1

    def f(self, y, mu, t):
        z = cvx.Variable(self.w.size)
        w_next = self.w + z

        # Objective Function
        ret = -self.gamma_excess * self.return_forecast.weight_expr(t, wplus=w_next, w_index=self.w_index)

        # # l1 penalty term
        # ret += mu * cvx.norm(w_next - y, 1)

        # Proximality term 1/2||x-z||^2
        gamma = 1  # TBD if I want to change this
        ret += 1 / (2 * gamma) * cvx.square(cvx.norm(w_next - y, 2))

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
            return w_next.value
        except Exception as e:
            print(e)
            print("Error solving f() part of the problem")

    def g(self, w_next):
        """Proximal operator for I_B(x) + gamma * phi_k(x)
        This function requires the calculation of the gradient TODO: Will implement at a later time to compare

        phi_k(x) : ||x||_1 - ||x||_k, i.e., the difference between the sum of elements of x and the sum of the k largest elements
        """
        # Set all negative elements to 0.
        # w_next = np.where(w_next <0, 0, w_next)
        # Sorting incoming vector
        i_s = np.argsort(w_next)[::-1]

        kappa = 1  # Can change this later to be = max(gradient(f)) - see paper by gonjac

        def prox_op(val, idx):
            for init_i, sorted_i in enumerate(idx):
                if init_i > self.card:
                    val[sorted_i] = self._saturation(self._soft_thresh(val[sorted_i], kappa), 0, 1)
                else:
                    val[sorted_i] = self._saturation(val[sorted_i], 0, 1)

        prox_op(w_next, i_s)
        # re-weight
        w_next = w_next / w_next.sum()
        return w_next

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

    def _hard_thresh(self, val, rho):
        # Not implemented
        if (1 / 2) * np.abs(val) ** 2 < rho:
            return 0
        else:
            return val

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


class ADMCardinalitySPO(CardinalitySPO):
    def __init__(self, mu=1e-4, **kwargs):
        self.method = "ADM"
        super().__init__(**kwargs)

    def ADM(self, t):
        y_next = self.w
        iteration = 0
        u = 0
        while True:
            w_next = self.f(y_next - u, t)
            y_next = self.g(w_next + u)

            # Check stopping criteria
            if iteration == 0:
                diff = np.linalg.norm((w_next - y_next), 1)
            else:
                diff_last = diff
                diff = np.linalg.norm((w_next - y_next), 1)

                if (diff_last - diff) ** 2 < self.THRESH:
                    z = y_next - self.w.value
                    return z

                if iteration >= self.MAX_ITER:
                    print("Max iter reached - not optimal!")
                    z = y_next - self.w.value
                    return z

            iteration += 1
            u = u + w_next - y_next

    def f(self, y, t):
        z = cvx.Variable(self.w.size)
        w_next = self.w + z

        # Objective Function
        ret = -self.gamma_excess * self.return_forecast.weight_expr(t, wplus=w_next, w_index=self.w_index)

        # Proximality term 1/2||x-z||^2
        gamma = 1  # TBD if I want to change this
        ret += 1 / (2 * gamma) * cvx.square(cvx.norm(w_next - y, 2))

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
            return w_next.value
        except Exception as e:
            print(e)
            print("Error solving f() part of the problem")

    def g(self, w_next):
        z = cvx.Variable(self.w.size)  # TODO pass index
        y_next = self.w + z

        # Proximality term 1/2||x-z||^2
        gamma = 1  # TBD if I want to change this
        prox = 1 / (2 * gamma) * cvx.square(cvx.norm(y_next - w_next, 2))

        obj = cvx.Minimize(prox)

        constraints = [cvx.sum(z) == 0]
        y_bool = cvx.Variable(w_next[:-1].shape[0], boolean=True)
        constraints += [sum(y_bool) <= self.card]
        for i in range(y_next[:-1].shape[0]):
            constraints += [y_next[i] <= 1 * y_bool[i]]

        prob = cvx.Problem(obj, constraints)
        try:
            prob.solve(solver=self.solver, **self.solver_opts)
            if prob.status == "unbounded":
                logging.error("The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(self.portfolio)

            if prob.status == "infeasible":
                logging.error("The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(self.portfolio)
            # Check that return is correct
            return y_next.value
        except Exception as e:
            print(e)
            print("Error solving g() part of the problem")


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


class badCardTrackingSinglePeriodOpt(BasePolicy):
    """This is old code. keeping for archiving purposes"""

    def __init__(
        self,
        return_forecast,
        returns_index,
        costs,
        constraints,
        cardinality,
        epsilon=10**-1,
        max_iter=100,
        h=0.1,
        penalty=1,
        approx_gradient=False,
        solver=None,
        solver_opts=None,
    ):

        if not isinstance(return_forecast, BaseReturnsModel):
            null_checker(return_forecast)

        self.return_forecast = return_forecast
        self.returns_index = returns_index
        self.cardinality = cardinality

        # Upper and Lower bound for saturation function
        self.upper = 1
        self.lower = 0

        # Start soft-thresholding parameter (also represents weighting param)
        self.gamma = 1

        self.max_iter = max_iter
        self.epsilon = epsilon
        self.h = h
        self.penalty = penalty
        super(CardTrackingSinglePeriodOpt, self).__init__()

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

        # Initialize error term
        u = np.zeros(w.size)

        # Initialization Variables
        self.z = cvx.Variable(w.size)
        self.wplus = w.values + self.z
        # Initial cardinality compliant array
        arr = np.array([value / self.cardinality] * self.cardinality + [0] * (w.size - self.cardinality))
        self.y = cvx.Parameter(shape=w.size, value=arr)

        iter = 0
        delta = self.epsilon

        while not self._endloop(iter, delta):
            # Store values from x(t-1)
            if not iter == 0:
                prev_wplus = self.wplus.value
                prev_f = self.prob.value
            # TODO: See if it makes sense to include opt in f(x)
            obj, constraints = self.f(value, t, u)
            self.prob = cvx.Problem(cvx.Minimize(obj), [cvx.sum(self.z) == 0] + constraints)

            # Attempt to solve f(x)
            try:
                self.prob.solve(solver=self.solver, **self.solver_opts)

                # Check if unbounded
                if self.prob.status == "unbounded":
                    logging.error("The problem is unbounded. Defaulting to no trades")
                    return self._nulltrade(portfolio)

                # Check if unfeasible
                if self.prob.status == "infeasible":
                    logging.error("The problem is infeasible. Defaulting to no trades")
                    return self._nulltrade(portfolio)
            except (cvx.SolverError, TypeError) as e:
                logging.error(e)
                logging.error("The solver %s failed. Defaulting to no trades" % self.solver)
                return self._nulltrade(portfolio)

            # Optimal weights in g(x)
            self.y.value = self.g(u)

            # Update error term
            u = u + self.wplus.value - self.y.value

            # Update gamma
            self.gamma = self._max_grad_back(self.wplus.value, self.z.value, self.y.value, u, value, t, self.h)

            # Update stopping criteria stuff

            # Currently delta is only f(xt+1)-f(xt)/f(xt)
            # TODO: Include g(x) but g(x) is a vector so not sure how to include it
            if iter == 0:
                delta = self.epsilon
            else:
                delta = (prev_f - self.prob.value) / prev_f

            iter += 1
        print(iter)
        return pd.Series(index=portfolio.index, data=(self.z.value * value))

    def f(self, portf_value, t, u):
        # y.value is the current optimal weights from g(x)
        if isinstance(self.return_forecast, BaseReturnsModel) and isinstance(self.returns_index, BaseReturnsModel):
            diff = cvx.square(self.returns_index.weight_expr(t) - self.return_forecast.weight_expr(t, self.wplus))
            tracking_term = cvx.huber(diff, 0.1)
        else:
            # TODO: Properly implement this if I want
            # diff = self.returns_index[t] - cvx.multiply(self.return_forecast[t], wplus)
            # tracking_term = cvx.sum(cvx.multiply(
            #     values_in_time(self.return_forecast, t).values,
            #     wplus))
            logging.warning("Not implemented see TrackingSinglePeriodOpt.get_trades()")

        # Penalty for distance from sparse vector
        # TODO: Understand the penalty factor rho
        pen = self.penalty * cvx.sum_squares(self.wplus - (self.y - u))
        # pen = 0
        assert tracking_term.is_convex()

        # Initialize Constraints
        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, self.wplus, self.z, portf_value)
            costs.append(cost_expr)
            constraints += const_expr

        for item in (con.weight_expr(t, self.wplus, self.z, portf_value) for con in self.constraints):
            if isinstance(item, list):
                constraints += item
            else:
                constraints += [item]

        for el in costs:
            assert el.is_convex()

        for el in constraints:
            assert el.is_dcp()

        # Return f(x)
        return tracking_term + pen + sum(costs), constraints

    def F(self, w, z, y, u, portf_value, t):
        # y.value is the current optimal weights from g(x)
        if isinstance(self.return_forecast, BaseReturnsModel) and isinstance(self.returns_index, BaseReturnsModel):
            diff = cvx.square(self.returns_index.weight_expr(t) - self.return_forecast.weight_expr(t, w))
            tracking_term = cvx.huber(diff, 0.1)
        else:
            # TODO: Properly implement this if I want
            # diff = self.returns_index[t] - cvx.multiply(self.return_forecast[t], wplus)
            # tracking_term = cvx.sum(cvx.multiply(
            #     values_in_time(self.return_forecast, t).values,
            #     wplus))
            logging.warning("Not implemented see TrackingSinglePeriodOpt.get_trades()")

        # Penalty for distance from sparse vector
        pen = self.penalty * cvx.norm(w - (y - u), 1)
        # pen = 0
        # Initialize Constraints

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, w, z, portf_value)
            costs.append(cost_expr)
            constraints += const_expr

        for item in (con.weight_expr(t, w, z, portf_value) for con in self.constraints):
            if isinstance(item, list):
                constraints += item
            else:
                constraints += [item]

        for el in costs:
            assert el.is_convex()

        # for el in constraints:
        #     assert el.is_dcp()

        # Return f(x)
        return tracking_term + pen + sum(costs), constraints

    def g(self, u):
        # wplus.value is the current optimal weights from f(x)
        w = self.wplus.value + u
        wcopy = w.copy()
        idx = np.argsort(-w)
        for i, j in enumerate(idx):
            wcopy[j] = self._proj_map(w[j], i)
        return wcopy

    def _proj_map(self, val, idx):
        if idx > self.cardinality:
            return self._saturation(self._soft_thresh(val, self.gamma), self.lower, self.upper)
        else:
            return self._saturation(val, self.lower, self.upper)

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

    # def _max_grad(self, f, x):
    #   this uses numdifftools
    #     return np.max(np.abs(nd.Gradient(f)(x)))

    # a function that return one standard basis function
    def _basis_vec(self, ndim, index):
        v = np.zeros(ndim)
        v[index] = 1.0
        return v

    # This does not work because costs are cvx expr.
    # def _grad_f(self, w, z, t, portf_value):
    #     diff = np.square(self.returns_index.weight_expr(t)
    #             - self.return_forecast.weight_expr(t, w)
    #         )
    #     tracking_term = self._huber(diff, 0.1)

    #     costs, constraints = [], []

    #     for cost in self.costs:
    #         cost_expr, const_expr = cost.weight_expr(t, w, z, portf_value)
    #         costs.append(cost_expr)

    # def _huber(value, M):
    #     thresh = np.abs(value) < M
    #     squared_loss = np.square(value) / 2
    #     linear_loss  = np.abs(value) - 0.5
    #     return np.where(thresh, squared_loss, linear_loss)

    # compute gradient by backward finite difference method
    def _max_grad_back(self, w, z, y, u, portf_value, t, h):
        h_ndim = w.size
        return np.max(
            np.abs(
                np.array(
                    [
                        (
                            self.F(w, z, y, u, portf_value, t)[0].value
                            - self.F(w - h * self._basis_vec(h_ndim, i), z, y, u, portf_value, t)[0].value
                        )
                        / h
                        for i in range(h_ndim)
                    ]
                )
            )
        )

    def _endloop(self, iter, delta):
        if iter == self.max_iter:
            return True
        elif delta < self.epsilon:
            return True
        else:
            return False
