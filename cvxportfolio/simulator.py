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

# TODO update benchmark weights (?)
# Also could try jitting with numba.


class MarketSimulator:
    logger = None

    def __init__(self, market_returns, costs, market_volumes=None, cash_key="cash"):
        """Provide market returns object and cost objects."""
        self.market_returns = market_returns
        if market_volumes is not None:
            self.market_volumes = market_volumes[market_volumes.columns.difference([cash_key])]
        else:
            self.market_volumes = None

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
        assert u.index.equals(h.index)

        if self.market_volumes is not None:
            # don't trade if volume is null
            null_trades = self.market_volumes.columns[self.market_volumes.loc[t] == 0]
            if len(null_trades):
                logging.info("No trade condition for stocks %s on %s" % (null_trades, t))
                u.loc[null_trades] = 0.0

        hplus = h + u
        costs = [cost.value_expr(t, h_plus=hplus, u=u) for cost in self.costs]
        for cost in costs:
            assert not pd.isnull(cost)
            assert not np.isinf(cost)

        u[self.cash_key] = -sum(u[u.index != self.cash_key]) - sum(costs)
        hplus[self.cash_key] = h[self.cash_key] + u[self.cash_key]

        # logging.info(hplus.index.sort_values())
        # logging.info(self.market_returns.columns.sort_values())

        assert hplus.index.sort_values().equals(self.market_returns.columns.sort_values())
        h_next = self.market_returns.loc[t] * hplus + hplus

        # assert (not h_next.isnull().values.any())
        # assert (not u.isnull().values.any())
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
        # diffs = {}
        for t in tqdm(simulation_times, position=tqdm_pos):
            # for t in simulation_times:
            # # Used for debugging
            # if t == datetime.strptime("2020-06-25", "%Y-%m-%d"):
            #     print("check time")
            if t in rebalance_on:
                logging.info("Getting trades at time %s" % t)
                start = time.time()
                try:
                    # u, diff = policy.get_trades(h, t)
                    u = policy.get_trades(h, t)
                    # results.log_data("diff_obj", t, policy.diff_obj.value)
                    # results.log_data("index_obj", t, policy.index_obj)
                    # results.log_data("portfolio_obj", t, policy.portfolio_obj.value)
                except Exception as e:
                    logging.warning("Solver failed on timestamp %s. Default to no trades." % t)
                    print(e)
                    u = pd.Series(index=h.index, data=0.0)
                end = time.time()
                results.log_policy(t, end - start)
            else:
                logging.info(f"{t} is not a rebalancing date")
                u = pd.Series(index=h.index, data=0.0)

            # diffs[t] = diff.value
            # print(self.cash_key in u.index)
            # print(self.cash_key in h.index)

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
            results.log_data("w_index", t, policy.w_index)
            results.log_data("market_returns", t, self.market_returns.loc[t])
        # pd_diff = pd.DataFrame(diffs)
        # pd_diff.to_csv("diffs.csv")
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
            results = workers.starmap(_run_backtest, worker_iter)
            # results = tqdm(
            #     workers.imap_unordered(_run_backtest, policies),
            #     total=num_workers,
            # )
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
