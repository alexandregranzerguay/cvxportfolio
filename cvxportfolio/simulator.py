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

import copy
from datetime import datetime
import logging
import time

# from pathos import multiprocessing
import multiprocess
import numpy as np
import pandas as pd
import cvxpy as cvx

from tqdm import tqdm

from .returns import MultipleReturnsForecasts

from .result import SimulationResult
from .costs import BaseCost

from .utils import null_checker, values_in_time

# TODO update benchmark weights (?)
# Also could try jitting with numba.


class MarketSimulator:
    logger = None

    def __init__(self, market_returns, costs, market_volumes=None, prices=None, cash_key="cash"):
        """Provide market returns object and cost objects."""
        self.market_returns = market_returns
        if market_volumes is not None:
            self.market_volumes = market_volumes[market_volumes.columns.difference([cash_key])]
        else:
            self.market_volumes = None
        self.prices = prices
        self.costs = costs
        for cost in self.costs:
            assert isinstance(cost, BaseCost)

        self.cash_key = cash_key

    def propagate(self, h, u, t):
        """Propagates the portfolio forward over time period t, given trades u.

        Args:
            h: pandas Series object describing current portfolio
            u: n vector with the stock trades (not cash)
            t: current time

        Returns:
            h_next: portfolio after returns propagation
            u: trades vector with simulated cash balance
        """
        if h.index.size > u.index.size:
            missing_idx = h.index.difference(u.index)
            s = pd.Series(data=(0.0), index=missing_idx)
            u = pd.concat([u, s])[h.index]
        assert u.index.equals(h.index)

        if self.market_volumes is not None:
            # don't trade if volume is null
            null_trades = self.market_volumes.columns[self.market_volumes.loc[t] == 0]
            if len(null_trades):
                logging.info("No trade condition for stocks %s on %s" % (null_trades, t))
                u.loc[null_trades] = 0.0

        hplus = h + u
        self.w_plus = hplus / hplus.sum()
        hplus_old = hplus.copy()
        costs = [cost.value_expr(t, h_plus=hplus, u=u) for cost in self.costs]
        for cost in costs:
            try:
                assert not pd.isnull(cost)
            except:
                cost = 0.0
            assert not np.isinf(cost)

        # if self.prices is not None:
        #     traded_value = u[u.index != self.cash_key] @ values_in_time(self.prices, t)[u.index[:-1]]
        #     u[self.cash_key] = -traded_value - sum(costs)
        # else:
        #     u[self.cash_key] = -sum(u[u.index != self.cash_key]) - sum(costs)
        u[self.cash_key] = -sum(u[u.index != self.cash_key]) - sum(costs)
        hplus[self.cash_key] = h[self.cash_key] + u[self.cash_key]

        # print((hplus_old[:-1] != hplus[:-1]).any())
        # print(hplus[self.cash_key])
        # print(u[self.cash_key])
        # logging.info(hplus.index.sort_values())
        # logging.info(self.market_returns.columns.sort_values())

        # assert hplus.index.sort_values().equals(self.market_returns.columns.sort_values())
        h_next = self.market_returns.loc[t] * hplus + hplus

        assert not h_next.isnull().values.all()
        assert not u.isnull().values.all()
        return h_next, u

    def run_backtest(self, initial_portfolio, start_time, end_time, policy, rebalance_on=None, tqdm_pos=0):
        """Backtest a single policy.

        rebalance_on: Dataframe of dates to execute trades on
        """
        # if do_not_log:
        #     logging.basicConfig(level=loglevel)
        # else:
        #     logging.basicConfig(filename=logfile_name, level=loglevel, filemode="w")

        results = SimulationResult(
            initial_portfolio=copy.copy(initial_portfolio),
            policy=policy,
            cash_key=self.cash_key,
            simulator=self,
        )
        h = initial_portfolio

        simulation_times = self.market_returns.index[
            (self.market_returns.index >= start_time) & (self.market_returns.index <= end_time)
        ]
        logging.info("Backtest started, from %s to %s" % (simulation_times[0], simulation_times[-1]))

        if rebalance_on is None:
            rebalance_on = simulation_times

        for t in tqdm(simulation_times, position=tqdm_pos):
            # # Used for debugging
            # if t == datetime.strptime("2020-06-25", "%Y-%m-%d"):
            #     print("check time")
            if t in rebalance_on:
                logging.info("Getting trades at time %s" % t)
                start = time.time()
                # try:
                if self.prices is not None:
                    u = policy.get_rounded_trades(h, self.prices, t)
                else:
                    u = policy.get_trades(h, t)
                # except Exception as e:
                #     logging.warning("Solver failed on timestamp %s. Default to no trades." % t)
                #     print(e)
                #     u = pd.Series(index=h.index, data=0.0)
                end = time.time()
                results.log_policy(t, end - start)
            else:
                logging.info(f"{t} is not a rebalancing date")
                u = pd.Series(index=h.index, data=0.0)

            if "index_weights" in policy.__dict__:
                # idx = policy.index_weights.index.get_loc(t, method="pad")
                # temp = policy.index_weights.iloc[idx]
                # results.log_data("w_index", t, policy.index_weights.iloc[idx])
                results.log_data("w_index", t, policy.index_weights.loc[t])

            # Index returns are only monthly... so this won't work
            # if "index_ret" in policy.__dict__:
            # results.log_data("index_ret", t, policy.index_ret.loc[t])
            # idx = policy.index_weights.index.get_loc(t, method="pad")
            # results.log_data("w_index", t, policy.index_weights.iloc[idx])
            assert not pd.isnull(u).any()
            logging.info("Propagating portfolio at time %s" % t)
            start = time.time()
            h, u = self.propagate(h, u, t)
            end = time.time()
            # assert (not h.isnull().values.any())
            assert not h.isnull().values.all()
            results.log_simulation(
                t=t,
                u=u,
                h_next=h,
                risk_free_return=self.market_returns.loc[t, self.cash_key],
                exec_time=end - start,
            )

            # Reporting
            try:
                results.log_data("te", t, policy.te)
            except:
                results.log_data("te", t, 0)
            results.log_data("w_dist", t, np.abs(self.w_plus - policy.index_weights.loc[t]))
            results.log_data("w_plus", t, self.w_plus)
            results.log_data("market_returns", t, self.market_returns.loc[t])

        logging.info("Backtest ended, from %s to %s" % (simulation_times[0], simulation_times[-1]))
        return results

    def run_multiple_backtest(
        self,
        initial_portf,
        start_time,
        end_time,
        policies,
        loglevel=logging.WARNING,
        parallel=False,
        rebalance_on=None,
    ):
        """Backtest multiple policies."""

        def _run_backtest(policy, tqdm_pos=0):
            return self.run_backtest(
                initial_portf, start_time, end_time, policy, rebalance_on=rebalance_on, tqdm_pos=tqdm_pos
            )

        num_workers = min(multiprocess.cpu_count(), len(policies))
        if parallel:
            # Note: multiprocess will not work with Gurobi (and maybe other optimizers) due to pickling of objects that can't be serialized.
            workers = multiprocess.Pool(num_workers)
            # results = workers.map(_run_backtest, policies)
            worker_iter = [[policies[i], i] for i in range(num_workers)]
            # starmap iterates over "pre-zipped" tuples
            results = workers.starmap(_run_backtest, worker_iter)
            workers.close()
            return results
        else:
            return list(map(_run_backtest, policies))

    def what_if(self, time, results, alt_policies, parallel=True):
        """Run alternative policies starting from given time."""
        # TODO fix
        initial_portf = copy.copy(results.h.loc[time])
        all_times = results.h.index
        alt_results = self.run_multiple_backtest(initial_portf, time, all_times[-1], alt_policies, parallel)
        for idx, alt_result in enumerate(alt_results):
            alt_result.h.loc[time] = results.h.loc[time]
            alt_result.h.sort_index(axis=0, inplace=True)
        return alt_results

    @staticmethod
    def reduce_signal_perturb(initial_weights, delta):
        """Compute matrix of perturbed weights given initial weights."""
        perturb_weights_matrix = np.zeros((len(initial_weights), len(initial_weights)))
        for i in range(len(initial_weights)):
            perturb_weights_matrix[i, :] = initial_weights / (1 - delta * initial_weights[i])
            perturb_weights_matrix[i, i] = (1 - delta) * initial_weights[i]
        return perturb_weights_matrix

    def attribute(self, true_results, policy, selector=None, delta=1, fit="linear", parallel=True):
        """Attributes returns over a period to individual alpha sources.

        Args:
            true_results: observed results.
            policy: the policy that achieved the returns.
                    Alpha model must be a stream.
            selector: A map from SimulationResult to time series.
            delta: the fractional deviation.
            fit: the type of fit to perform.
        Returns:
            A dict of alpha source to return series.
        """
        # Default selector looks at profits.
        if selector is None:

            def selector(result):
                return result.v - sum(result.initial_portfolio)

        alpha_stream = policy.return_forecast
        assert isinstance(alpha_stream, MultipleReturnsForecasts)
        times = true_results.h.index
        weights = alpha_stream.weights
        assert np.sum(weights) == 1
        alpha_sources = alpha_stream.alpha_sources
        num_sources = len(alpha_sources)
        Wmat = self.reduce_signal_perturb(weights, delta)
        perturb_pols = []
        for idx in range(len(alpha_sources)):
            new_pol = copy.copy(policy)
            new_pol.return_forecast = MultipleReturnsForecasts(alpha_sources, Wmat[idx, :])
            perturb_pols.append(new_pol)
        # Simulate
        p0 = true_results.initial_portfolio
        alt_results = self.run_multiple_backtest(p0, times[0], times[-1], perturb_pols, parallel=parallel)
        # Attribute.
        true_arr = selector(true_results).values
        attr_times = selector(true_results).index
        Rmat = np.zeros((num_sources, len(attr_times)))
        for idx, result in enumerate(alt_results):
            Rmat[idx, :] = selector(result).values
        Pmat = cvx.Variable((num_sources, len(attr_times)))
        if fit == "linear":
            prob = cvx.Problem(cvx.Minimize(0), [Wmat @ Pmat == Rmat])
            prob.solve()
        elif fit == "least-squares":
            error = cvx.sum_squares(Wmat @ Pmat - Rmat)
            prob = cvx.Problem(cvx.Minimize(error), [Pmat.T @ weights == true_arr])
            prob.solve()
        else:
            raise Exception("Unknown fitting method.")
        # Dict of results.
        wmask = np.tile(weights[:, np.newaxis], (1, len(attr_times))).T
        data = pd.DataFrame(
            columns=[s.name for s in alpha_sources],
            index=attr_times,
            data=Pmat.value.T * wmask,
        )
        data["residual"] = true_arr - np.asarray((weights @ Pmat).value).ravel()
        data["RMS error"] = np.asarray(cvx.norm(Wmat @ Pmat - Rmat, 2, axis=0).value).ravel()
        data["RMS error"] /= np.sqrt(num_sources)
        return data

    def rebalance(self, x_init, cash_init, w_opt, cur_prices, turnover=1, interest=0):
        """MIP
        This method returns the optimal transactions based on available cash
        TODO: Adapt this to work with Simulator Object

        x_delta: amount traded without tx costs and integer constraints
        x_delta_opt: absolute amount traded after transaction costs and integer contraints
        tx_cost: tx costs associated with x_delta_opt
        x_opt_int: optimal positions based on x_init + sign_x_delta * x_delta_opt
        sign_x_delta: signs of trades, positive when buying, negative when selling
        """
        n = len(x_init)
        w_opt = np.array(w_opt)

        # Portfolio value
        Vp = np.dot(x_init, cur_prices) + cash_init

        # ideal positions with no consideration for constraints
        try:
            x_opt = Vp * w_opt / cur_prices
        except:
            print("No solution existed - returned initial portfolio")
            return x_init, cash_init, 0

        # Check that all positions are non-negative (no shorting)
        if not all(i >= 0 for i in x_opt):
            raise ValueError

        x_delta = x_opt - x_init  # if positive -> buy, if negative -> sell
        abs_x_delta = np.abs(
            x_delta
        )  # ideal amount transacted in an absolute sense (without tx costs and integer consideration)

        sign_x_delta = x_delta / abs_x_delta  # used to track the signs of the original transaction directions

        # Optimization
        cpx = cplex.Cplex()
        cpx.objective.set_sense(cpx.objective.sense.minimize)

        var_names = [f"x{i}" for i in range(n)] + [f"x_d{i}" for i in range(n)] + [f"x_t{i}" for i in range(n)] + ["c"]
        a = 1  # Used to reduce or increase penalty on holding cash
        c = [-1] * n + [0] * n + [0] * n + [a]  # Vector of ones for each x_delta and 0's for transaction costs
        # TODO: Think about building a A matrix creator
        A = []
        for i in range(n):
            A.append(
                [
                    [i, n, n + 1 + i],
                    [1.0, sign_x_delta[i] * cur_prices[i], -1 * 0.005 * cur_prices[i]],
                ]
            )
        for i in range(n):
            A.append([[i], [1.0]])
        for i in range(n):
            A.append(
                [
                    [n, n + 1 + i],
                    [1.0, 1.0],
                ]
            )
        # Cash constraints
        A.append([[n, 2 * n + 1], [1.0, 1.0]])

        zeros = [0] * n
        # NOTE: cash_init - interest ensures that the opt cash > interest amount
        rhs = (turnover * abs_x_delta).tolist() + [cash_init - interest] + zeros + [interest]
        senses = "E" * (2 * n + 1) + "G"

        cpx.linear_constraints.add(rhs=rhs, senses=senses)
        cpx.variables.add(
            obj=c,
            columns=A,
            names=var_names,
            types=[cpx.variables.type.integer] * 20 + [cpx.variables.type.continuous] * 41,
        )

        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.solve()

        # Results
        val_star = np.array(cpx.solution.get_values())

        x_delta_opt = np.array(val_star[:20])
        x_opt_int = x_init + sign_x_delta * x_delta_opt
        tx_opt = np.array(val_star[40:60])
        cash_opt = val_star[-1]

        # If any position got shorted in the optimization problem, simply remove the shorted assets.
        # Remove from cash amount associated with shorted assets
        # Add to cash the amount associated with transaction costs of shorted assets.
        # NOTE:This is slightly naive but wanted to have simple solution here
        for i in range(n):
            if x_opt_int[i] < 0:
                cash_opt += np.abs(x_opt_int[i]) * cur_prices[i] * (0.005 - 1)
                x_opt_int[i] = 0
        return x_opt_int, cash_opt, tx_opt
