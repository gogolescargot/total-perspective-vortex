import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, reg=0.0, log=True):
        self.n_components = int(n_components)
        self.reg = reg
        self.log = bool(log)

    def _validate_inputs(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("X must be 3D: (n_epochs, n_channels, n_times)")
        if y is not None:
            y = np.asarray(y)
        return X, y

    def _epoch_cov(self, X):
        cov = X @ X.T
        trace = np.trace(cov)
        if trace <= 0:
            return cov
        return cov / trace

    def _regularize(self, cov):
        reg = float(self.reg) if self.reg is not None else 0.0
        if reg:
            cov = cov + reg * np.eye(cov.shape[0])
        return cov

    def _compute_class_covs(self, X, y):
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("CustomCSP supports exactly 2 classes")

        covs = {classes[0]: [], classes[1]: []}
        for xi, yi in zip(X, y):
            cov = self._epoch_cov(xi)
            cov = self._regularize(cov)
            covs[yi].append(cov)

        C0 = np.mean(covs[classes[0]], axis=0)
        C1 = np.mean(covs[classes[1]], axis=0)
        return C0, C1

    def _composite_cov(self, C0, C1):
        return C0 + C1

    def _solve_generalized_eig(self, C0, composite):
        eigvals, eigvecs = eigh(C0, composite)
        return eigvals, eigvecs

    def _select_filters(self, eigvals, eigvecs):
        half = self.n_components // 2
        selected = np.concatenate(
            [np.arange(half), np.arange(len(eigvals) - half, len(eigvals))]
        )
        W = eigvecs[:, selected].T
        return W, eigvals[selected]

    def _project(self, X):
        return np.array([self.filters_.dot(epoch) for epoch in X])

    def _features_from_projection(self, proj):
        var = proj.var(axis=2)
        var_norm = var / (var.sum(axis=1, keepdims=True) + 1e-12)
        if self.log:
            return np.log(var_norm)
        return var_norm

    def fit(self, X, y):
        X, y = self._validate_inputs(X, y)

        n_channels = X.shape[1]
        if self.n_components <= 0 or self.n_components > n_channels:
            raise ValueError("n_components must be in (0, n_channels]")
        if self.n_components % 2 != 0:
            raise ValueError(
                "n_components should be even (pairs from both ends)"
            )

        C0, C1 = self._compute_class_covs(X, y)
        composite = self._composite_cov(C0, C1)

        eigvals, eigvecs = self._solve_generalized_eig(C0, composite)

        W, selected_eigvals = self._select_filters(eigvals, eigvecs)

        self.filters_ = W
        self.eigenvalues_ = selected_eigvals
        return self

    def transform(self, X):
        X, _ = self._validate_inputs(X)
        if not hasattr(self, "filters_"):
            raise RuntimeError("Call fit before transform")

        proj = self._project(X)
        return self._features_from_projection(proj)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
