"""Registry of gate classifiers with a uniform scikit-learn-style interface."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

GateBuilder = Callable[..., Any]

GATE_REGISTRY: dict[str, GateBuilder] = {}


def register_gate(name: str) -> Callable[[GateBuilder], GateBuilder]:
    def decorator(builder: GateBuilder) -> GateBuilder:
        GATE_REGISTRY[name] = builder
        return builder

    return decorator


@register_gate("logistic_regression")
def _build_logreg(**kwargs: Any) -> LogisticRegression:
    return LogisticRegression(random_state=42, max_iter=1000, **kwargs)


@register_gate("xgboost")
def _build_xgb(**kwargs: Any) -> Any:
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "xgboost is required for the 'xgboost' gate. Install with `uv pip install xgboost`."
        ) from exc

    defaults = {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def build_gate(name: str = "logistic_regression", **kwargs: Any) -> Any:
    """Instantiate a registered gate classifier by name."""
    if name not in GATE_REGISTRY:
        raise ValueError(
            f"Unknown gate '{name}'. Registered gates: {sorted(GATE_REGISTRY)}"
        )
    return GATE_REGISTRY[name](**kwargs)


def fit_predict_proba(
    gate: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray | None = None,
) -> tuple[Any, np.ndarray]:
    """Fit gate on training data and return positive-class probabilities on X_eval."""
    gate.fit(X_train, y_train)
    source = X_eval if X_eval is not None else X_train
    proba = gate.predict_proba(source)[:, 1]
    return gate, proba
