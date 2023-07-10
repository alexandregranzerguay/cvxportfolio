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

from abc import abstractmethod
import logging

import cvxpy as cvx
import numpy as np
import pandas as pd

from .costs import BaseCost
from .constraints import BaseConstraint
from .utils import values_in_time

from cvxpy.atoms.affine.wraps import psd_wrap

logger = logging.getLogger(__name__)


__all__ = [
    "FullSigma",
    "onlineFullSigma",
    "EmpSigma",
    "SqrtSigma",
    "WorstCaseRisk",
    "RobustFactorModelSigma",
    "RobustSigma",
    "FactorModelSigma",
    "FullSigmaTEConst",
    "FullSigmaTECost",
]


# def locator(obj, t):
#     """Picks last element before t."""
#     try:
#         if isinstance(obj, pd.Panel):
#             return obj.iloc[obj.axes[0].get_loc(t, method='pad')]

#         elif isinstance(obj.index, pd.MultiIndex):
#             prev_t = obj.loc[:t, :].index.values[-1][0]
#         else:
#             prev_t = obj.loc[:t, :].index.values[-1]

#         return obj.loc[prev_t, :]

#     except AttributeError:  # obj not pandas
#         return obj


class BaseRiskModel(BaseCost):
    def __init__(self, **kwargs):
        self.w_bench = kwargs.pop("w_bench", 0.0)
        super(BaseRiskModel, self).__init__()
        self.gamma_half_life = kwargs.pop("gamma_half_life", np.inf)

    def weight_expr(self, t, w_plus, z, value):
        self.expression = self._estimate(t, w_plus - self.w_bench, z, value)
        return self.gamma * self.expression, []

    @abstractmethod
    def _estimate(self, t, w_plus, z, value):
        pass

    def weight_expr_ahead(self, t, tau, w_plus, z, value):
        """Estimate risk model at time tau in the future."""
        if self.gamma_half_life == np.inf:
            gamma_multiplier = 1.0
        else:
            decay_factor = 2 ** (-1 / self.gamma_half_life)
            # TODO not dependent on days
            gamma_init = decay_factor ** ((tau - t).days)
            gamma_multiplier = gamma_init * (1 - decay_factor) / (1 - decay_factor)

        return gamma_multiplier * self.weight_expr(t, w_plus, z, value)[0], []

    def optimization_log(self, t):
        if self.expression.value:
            return self.expression.value
        else:
            return np.NaN


class BaseRiskModelConst(BaseConstraint):
    def __init__(self, **kwargs):
        self.w_bench = kwargs.pop("w_bench", 0.0)
        super(BaseRiskModelConst, self).__init__()
        self.gamma_half_life = kwargs.pop("gamma_half_life", np.inf)

    def weight_expr(self, t, w_plus, z, value):
        self.expression = self._estimate(t, w_plus - self.w_bench, z, value)
        return self.expression

    @abstractmethod
    def _estimate(self, t, w_plus, z, value):
        pass

    # def weight_expr_ahead(self, t, tau, w_plus, z, value):
    #     """Estimate risk model at time tau in the future."""
    #     if self.gamma_half_life == np.inf:
    #         gamma_multiplier = 1.0
    #     else:
    #         decay_factor = 2 ** (-1 / self.gamma_half_life)
    #         # TODO not dependent on days
    #         gamma_init = decay_factor ** ((tau - t).days)
    #         gamma_multiplier = gamma_init * (1 - decay_factor) / (1 - decay_factor)

    # return gamma_multiplier * self.weight_expr(t, w_plus, z, value)[0], []

    def optimization_log(self, t):
        if self.expression.value:
            return self.expression.value
        else:
            return np.NaN


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


class FullSigma(FilterAssets, BaseRiskModel):
    """Quadratic risk model with full covariance matrix.

    Args:
        Sigma (:obj:`pd.Panel`): Panel of Sigma matrices,
            or single matrix.

    """

    def __init__(self, Sigma, conditioning=0, **kwargs):
        self.Sigma = Sigma  # Sigma is either a matrix or a pd.Panel
        self.cond = conditioning
        super().__init__(**kwargs)

    def _estimate(self, t, wplus, z, value):
        Sigma = values_in_time(self.Sigma, t)
        idx = Sigma.columns.get_indexer(self.assets)
        Sigma = Sigma.loc[:, self.assets].iloc[idx]
        Sigma = Sigma + self.cond * np.diag(np.ones(Sigma.shape[0]))  # Shrinkage
        Sigma = psd_wrap(Sigma)  # Assert PSD
        try:
            self.expression = cvx.quad_form(wplus, Sigma)
        except TypeError:
            self.expression = cvx.quad_form(wplus, Sigma.values)
        return self.expression

    def get_sigma(self, t):
        Sigma = values_in_time(self.Sigma, t)
        idx = Sigma.columns.get_indexer(self.assets)
        Sigma = Sigma.loc[:, self.assets].iloc[idx]
        return Sigma + self.cond * np.diag(np.ones(Sigma.shape[0]))


class onlineFullSigma(FullSigma):
    def __init__(self, returns, lookback, conditioning=0, **kwargs):
        self.online = True
        self.returns = returns
        self.lookback = lookback
        super().__init__(Sigma=None, conditioning=conditioning, **kwargs)

    def update(self, t):
        if self.assets is not None:
            returns = self.returns[self.assets]
        else:
            returns = self.returns
        idx = returns.index.get_indexer([t])[0]

        cov = returns.iloc[max(0, idx - self.lookback) : idx].cov()
        # check if cov has any np.inf or np.nan or 0.0
        if np.isinf(cov).values.any() or np.isnan(cov).values.any() or (cov == 0.0).values.any():
            cov = cov.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # is_psd = np.all(np.linalg.eigvals(cov) + self.cond * np.diag(np.ones(cov.shape[0])) >= 0)
        self.Sigma = pd.concat({t: cov}, names=["date"]).droplevel(1)


class FullSigmaTEConst(BaseRiskModelConst):
    """Quadratic risk model with full covariance matrix.

    Args:
        Sigma (:obj:`pd.Panel`): Panel of Sigma matrices,
            or single matrix.

    """

    def __init__(self, Sigma, float_shares, index_prices, limit=None, **kwargs):
        self.Sigma = Sigma  # Sigma is either a matrix or a pd.Panel
        self.float_shares = float_shares
        self.index_prices = index_prices
        self.limit = limit
        try:
            assert not pd.isnull(Sigma).values.any()
        except AttributeError:
            assert not pd.isnull(Sigma).any()
        super(FullSigmaTEConst, self).__init__(**kwargs)

    def weight_expr(self, t, w_plus, z, value):
        self.w_bench = self._get_index_weights(t)
        self.expression = self._estimate(t, w_plus - self.w_bench, z, value)
        return [self.expression <= self.limit]

    # def weight_expr_ahead(self, t, tau, w_plus, z, value):
    #     """Estimate risk model at time tau in the future."""
    #     if self.gamma_half_life == np.inf:
    #         gamma_multiplier = 1.0
    #     else:
    #         decay_factor = 2 ** (-1 / self.gamma_half_life)
    #         # TODO not dependent on days
    #         gamma_init = decay_factor ** ((tau - t).days)
    #         gamma_multiplier = gamma_init * (1 - decay_factor) / (1 - decay_factor)
    #     expr = self.weight_expr(t, w_plus, z, value)
    #     return gamma_multiplier * expr[0], expr[1]

    def _estimate(self, t, wplus, z, value):
        try:
            self.expression = cvx.quad_form(wplus, values_in_time(self.Sigma, t))
        except TypeError:
            self.expression = cvx.quad_form(wplus, values_in_time(self.Sigma, t).values)
        return self.expression

    def _get_index_weights(self, t):
        market_cap = self.float_shares["float_shares"].multiply(values_in_time(self.index_prices, t)).fillna(0)
        index_weights = market_cap / market_cap.sum()
        index_weights["Cash"] = 0
        return index_weights


class FullSigmaTECost(BaseRiskModel):
    """Quadratic risk model with full covariance matrix.

    Args:
        Sigma (:obj:`pd.Panel`): Panel of Sigma matrices,
            or single matrix.

    """

    def __init__(self, Sigma, float_shares, index_prices, limit=None, **kwargs):
        self.Sigma = Sigma  # Sigma is either a matrix or a pd.Panel
        self.float_shares = float_shares
        self.index_prices = index_prices
        self.limit = limit
        try:
            assert not pd.isnull(Sigma).values.any()
        except AttributeError:
            assert not pd.isnull(Sigma).any()
        super(FullSigmaTECost, self).__init__(**kwargs)

    def weight_expr(self, t, w_plus, z, value):
        self.w_bench = self._get_index_weights(t)
        self.expression = self._estimate(t, w_plus - self.w_bench, z, value)
        return self.gamma * self.expression, []

    def weight_expr_ahead(self, t, tau, w_plus, z, value):
        """Estimate risk model at time tau in the future."""
        if self.gamma_half_life == np.inf:
            gamma_multiplier = 1.0
        else:
            decay_factor = 2 ** (-1 / self.gamma_half_life)
            # TODO not dependent on days
            gamma_init = decay_factor ** ((tau - t).days)
            gamma_multiplier = gamma_init * (1 - decay_factor) / (1 - decay_factor)

        # using tau here instead of t as I think this will impact control optimization
        return gamma_multiplier * self.weight_expr(t, w_plus, z, value)[0], []

    def _estimate(self, t, wplus, z, value):
        try:
            self.expression = cvx.quad_form(wplus, values_in_time(self.Sigma, t))
        except TypeError:
            self.expression = cvx.quad_form(wplus, values_in_time(self.Sigma, t).values)
        return self.expression

    def _get_index_weights(self, t):
        market_cap = self.float_shares["float_shares"].multiply(values_in_time(self.index_prices, t)).fillna(0)
        index_weights = market_cap / market_cap.sum()
        index_weights["Cash"] = 0
        return index_weights


class EmpSigma(BaseRiskModel):
    """Empirical Sigma matrix, built looking at *lookback* past returns."""

    def __init__(self, returns, lookback, **kwargs):
        """returns is dataframe, lookback is int"""
        self.returns = returns
        self.lookback = lookback
        assert not np.any(pd.isnull(returns))
        super(EmpSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, z, value):
        idx = self.returns.index.get_loc(t)
        # TODO make sure pandas + cvxpy works
        R = self.returns.iloc[max(idx - 1 - self.lookback, 0) : idx - 1]
        assert R.shape[0] > 0
        self.expression = cvx.sum_squares(R.values * wplus) / self.lookback
        return self.expression


class SqrtSigma(BaseRiskModel):
    def __init__(self, sigma_sqrt, **kwargs):
        """returns is dataframe, lookback is int"""
        self.sigma_sqrt = sigma_sqrt
        assert not np.any(pd.isnull(sigma_sqrt))
        super(SqrtSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, z, value):
        # TODO make sure pandas + cvxpy works
        self.expression = cvx.sum_squares(wplus.T * self.sigma_sqrt.values)
        return self.expression


class FactorModelSigma(BaseRiskModel):
    def __init__(self, exposures, factor_Sigma, idiosync, **kwargs):
        """Each is a pd.Panel (or ) or a vector/matrix"""
        self.exposures = exposures
        assert not exposures.isnull().values.any()
        self.factor_Sigma = factor_Sigma
        assert not factor_Sigma.isnull().values.any()
        self.idiosync = idiosync
        assert not idiosync.isnull().values.any()
        super(FactorModelSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, z, value):
        self.expression = cvx.sum_squares(
            cvx.multiply(np.sqrt(values_in_time(self.idiosync, t)), wplus)
        ) + cvx.quad_form(
            (wplus.T @ values_in_time(self.exposures, t).values.T).T,
            values_in_time(self.factor_Sigma, t).values,
        )
        return self.expression


class RobustSigma(BaseRiskModel):
    """Implements covariance forecast error risk."""

    def __init__(self, Sigma, gamma_risk, epsilon, **kwargs):
        self.Sigma = Sigma  # pd.Panel or matrix
        self.epsilon = epsilon  # pd.Series or scalar
        self.gamma_risk = gamma_risk
        super(RobustSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, z, value):
        self.expression = (
            self.gamma_risk * cvx.quad_form(wplus, values_in_time(self.Sigma, t))
            + values_in_time(self.epsilon, t) * (cvx.abs(wplus).T @ np.diag(values_in_time(self.Sigma, t))) ** 2
        )

        return self.expression


class RobustFactorModelSigma(BaseRiskModel):
    """Implements covariance forecast error risk."""

    def __init__(self, exposures, factor_Sigma, idiosync, epsilon, **kwargs):
        """Each is a pd.Panel (or ) or a vector/matrix"""
        self.exposures = exposures
        assert not exposures.isnull().values.any()
        self.factor_Sigma = factor_Sigma
        assert not factor_Sigma.isnull().values.any()
        self.idiosync = idiosync
        assert not idiosync.isnull().values.any()
        self.epsilon = epsilon
        super(RobustFactorModelSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, z, value):
        F = values_in_time(self.exposures, t)
        f = (wplus.T * F.T).T
        Sigma_F = values_in_time(self.factor_Sigma, t)
        D = values_in_time(self.idiosync, t)
        self.expression = (
            cvx.sum_squares(cvx.multiply(np.sqrt(D), wplus))
            + cvx.quad_form(f, Sigma_F)
            + self.epsilon * (cvx.abs(f).T * np.sqrt(np.diag(Sigma_F))) ** 2
        )

        return self.expression


class WorstCaseRisk(BaseRiskModel):
    def __init__(self, riskmodels, **kwargs):
        self.riskmodels = riskmodels
        super(WorstCaseRisk, self).__init__(**kwargs)

    def _estimate(self, t, wplus, z, value):
        self.risks = [risk.weight_expr(t, wplus, z, value) for risk in self.riskmodels]
        return cvx.max_elemwise(*self.risks)

    def optimization_log(self, t):
        """Return data to log in the result object."""
        return pd.Series(
            index=[model.__class__.__name__ for model in self.riskmodels],
            data=[risk.value[0, 0] for risk in self.risks],
        )
