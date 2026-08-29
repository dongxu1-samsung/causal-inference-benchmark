"""
Tree-based and meta-learner models for causal inference benchmark.
Models: CausalForestDML, DR-Learner, X-Learner, R-Learner, Uplift RF, BART.

Each model follows the interface:
    train_<model>(X_train, t_train, y_train, X_val, t_val, y_val) -> model_dict
    predict_<model>(model_dict, X_test) -> ite_pred (numpy array)
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


def _get_lgbm_params(n_samples, n_features):
    """Adaptive LGBM hyperparameters based on dataset size."""
    # For high-dim datasets, use fewer estimators and shallower trees
    if n_features > 1000:
        return dict(n_estimators=100, max_depth=4, learning_rate=0.15,
                    verbose=-1, n_jobs=-1, random_state=42,
                    colsample_bytree=0.3, subsample=0.8)
    elif n_features > 100:
        return dict(n_estimators=150, max_depth=5, learning_rate=0.1,
                    verbose=-1, n_jobs=-1, random_state=42,
                    colsample_bytree=0.5)
    else:
        return dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                    verbose=-1, n_jobs=-1, random_state=42)


def _estimate_propensity(X, t):
    """Estimate propensity scores using LightGBM."""
    from lightgbm import LGBMClassifier
    params = _get_lgbm_params(len(X), X.shape[1])
    prop_model = LGBMClassifier(**params)
    prop_model.fit(X, t)
    p = prop_model.predict_proba(X)[:, 1]
    return np.clip(p, 0.025, 0.975), prop_model


def _reduce_dims(X, max_dim=500):
    """Reduce dimensionality for models that are slow on high-dim data.
    Uses variance-based feature selection (keep top-k by variance).
    Returns (X_reduced, selector_indices)."""
    if X.shape[1] <= max_dim:
        return X, None
    var = np.var(X, axis=0)
    top_idx = np.argsort(-var)[:max_dim]
    top_idx = np.sort(top_idx)  # maintain order
    return X[:, top_idx], top_idx


# ==============================================================================
# 1. CausalForestDML (EconML)
# ==============================================================================
def train_causal_forest_dml(X_train, t_train, y_train, X_val, t_val, y_val):
    from econml.dml import CausalForestDML
    from lightgbm import LGBMRegressor

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    # Reduce dims for CausalForestDML (very slow on high-dim)
    X_red, sel_idx = _reduce_dims(X_all, max_dim=200)

    n_samples = len(X_red)
    n_features = X_red.shape[1]
    lgbm_params = _get_lgbm_params(n_samples, n_features)
    lgbm_params.pop('random_state', None)

    n_est = min(300, max(100, n_samples // 20))
    min_leaf = max(5, n_samples // 200)

    model = CausalForestDML(
        model_y=LGBMRegressor(**lgbm_params),
        model_t=LGBMRegressor(**lgbm_params),
        discrete_treatment=True,
        n_estimators=n_est,
        max_depth=None,
        min_samples_leaf=min_leaf,
        honest=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(y_all, t_all, X=X_red)
    return {'model': model, 'sel_idx': sel_idx, 'type': 'causal_forest_dml'}


def predict_causal_forest_dml(model_dict, X_test):
    model = model_dict['model']
    sel_idx = model_dict['sel_idx']
    X = X_test[:, sel_idx] if sel_idx is not None else X_test
    cate = model.effect(X)
    return cate.flatten()


# ==============================================================================
# 2. DR-Learner (CausalML) with LightGBM
# ==============================================================================
def train_dr_learner(X_train, t_train, y_train, X_val, t_val, y_val):
    from causalml.inference.meta import BaseDRLearner
    from lightgbm import LGBMRegressor

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    p_all, prop_model = _estimate_propensity(X_all, t_all)

    lgbm_params = _get_lgbm_params(len(X_all), X_all.shape[1])
    learner = BaseDRLearner(
        learner=LGBMRegressor(**lgbm_params),
    )
    learner.fit(X=X_all, treatment=t_all, y=y_all, p=p_all)
    return {'model': learner, 'prop_model': prop_model, 'type': 'dr_learner'}


def predict_dr_learner(model_dict, X_test):
    model = model_dict['model']
    cate = model.predict(X=X_test)
    return cate.flatten()


# ==============================================================================
# 3. X-Learner (CausalML) with XGBoost
# ==============================================================================
def train_x_learner(X_train, t_train, y_train, X_val, t_val, y_val):
    from causalml.inference.meta import BaseXLearner
    from xgboost import XGBRegressor

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    p_all, prop_model = _estimate_propensity(X_all, t_all)

    n_features = X_all.shape[1]
    n_est = 100 if n_features > 1000 else 200
    max_d = 4 if n_features > 1000 else 6
    colsample = 0.3 if n_features > 1000 else 1.0

    learner = BaseXLearner(
        learner=XGBRegressor(
            n_estimators=n_est, max_depth=max_d, learning_rate=0.1,
            colsample_bytree=colsample, subsample=0.8 if n_features > 1000 else 1.0,
            verbosity=0, n_jobs=-1, random_state=42,
            tree_method='hist',
        ),
    )
    learner.fit(X=X_all, treatment=t_all, y=y_all, p=p_all)
    return {'model': learner, 'prop_model': prop_model, 'type': 'x_learner'}


def predict_x_learner(model_dict, X_test):
    model = model_dict['model']
    prop_model = model_dict['prop_model']
    p_test = prop_model.predict_proba(X_test)[:, 1]
    p_test = np.clip(p_test, 0.025, 0.975)
    cate = model.predict(X=X_test, p=p_test)
    return cate.flatten()


# ==============================================================================
# 4. R-Learner (CausalML) with LightGBM
# ==============================================================================
def train_r_learner(X_train, t_train, y_train, X_val, t_val, y_val):
    from causalml.inference.meta import BaseRLearner
    from lightgbm import LGBMRegressor

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    p_all, prop_model = _estimate_propensity(X_all, t_all)

    lgbm_params = _get_lgbm_params(len(X_all), X_all.shape[1])
    learner = BaseRLearner(
        learner=LGBMRegressor(**lgbm_params),
        outcome_learner=LGBMRegressor(**lgbm_params),
        effect_learner=LGBMRegressor(**lgbm_params),
    )
    learner.fit(X=X_all, treatment=t_all, y=y_all, p=p_all)
    return {'model': learner, 'prop_model': prop_model, 'type': 'r_learner'}


def predict_r_learner(model_dict, X_test):
    model = model_dict['model']
    cate = model.predict(X=X_test)
    return cate.flatten()


# ==============================================================================
# 5. Uplift Random Forest (CausalML)
# ==============================================================================
def train_uplift_rf(X_train, t_train, y_train, X_val, t_val, y_val):
    from causalml.inference.tree import UpliftRandomForestClassifier

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    # Reduce dims for UpliftRF (slow on high-dim)
    X_red, sel_idx = _reduce_dims(X_all, max_dim=200)
    n_samples = len(X_red)

    # UpliftRF needs binary outcome; discretize continuous outcome at median
    y_median = np.median(y_all)
    y_binary = (y_all > y_median).astype(int)

    # Treatment as string labels (required by CausalML UpliftRF)
    t_str = np.array(['treatment' if ti == 1 else 'control' for ti in t_all])

    n_est = min(300, max(100, n_samples // 20))

    model = UpliftRandomForestClassifier(
        n_estimators=n_est,
        max_depth=8,
        min_samples_leaf=max(10, n_samples // 200),
        evaluationFunction='KL',
        control_name='control',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_red, treatment=t_str, y=y_binary)
    return {'model': model, 'sel_idx': sel_idx,
            'y_std': y_all.std() + 1e-8, 'type': 'uplift_rf'}


def predict_uplift_rf(model_dict, X_test):
    model = model_dict['model']
    sel_idx = model_dict['sel_idx']
    X = X_test[:, sel_idx] if sel_idx is not None else X_test
    uplift = model.predict(X)
    if isinstance(uplift, np.ndarray):
        if uplift.ndim == 2:
            cate = uplift[:, 0]
        else:
            cate = uplift
    else:
        cate = np.array(uplift).flatten()
    cate = cate * model_dict['y_std']
    return cate.flatten()


# ==============================================================================
# 6. BART - Bayesian Additive Regression Trees (T-learner with HistGBR)
# ==============================================================================
def train_bart(X_train, t_train, y_train, X_val, t_val, y_val):
    from sklearn.ensemble import HistGradientBoostingRegressor

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    treated_mask = t_all == 1
    control_mask = t_all == 0

    min_leaf = max(5, min(treated_mask.sum(), control_mask.sum()) // 50)

    model_t1 = HistGradientBoostingRegressor(
        max_iter=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=min_leaf, random_state=42,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
    )
    model_t0 = HistGradientBoostingRegressor(
        max_iter=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=min_leaf, random_state=42,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
    )

    model_t1.fit(X_all[treated_mask], y_all[treated_mask])
    model_t0.fit(X_all[control_mask], y_all[control_mask])

    return {'model_t1': model_t1, 'model_t0': model_t0, 'type': 'bart'}


def predict_bart(model_dict, X_test):
    mu1 = model_dict['model_t1'].predict(X_test)
    mu0 = model_dict['model_t0'].predict(X_test)
    return mu1 - mu0


# ==============================================================================
# Unified interface
# ==============================================================================
MODEL_REGISTRY = {
    'causal_forest_dml': (train_causal_forest_dml, predict_causal_forest_dml),
    'dr_learner': (train_dr_learner, predict_dr_learner),
    'x_learner': (train_x_learner, predict_x_learner),
    'r_learner': (train_r_learner, predict_r_learner),
    'uplift_rf': (train_uplift_rf, predict_uplift_rf),
    'bart': (train_bart, predict_bart),
}


def train_ml_model(model_name, X_train, t_train, y_train, X_val, t_val, y_val):
    train_fn, _ = MODEL_REGISTRY[model_name]
    return train_fn(X_train, t_train, y_train, X_val, t_val, y_val)


def predict_ml_model(model_name, model_dict, X_test):
    _, predict_fn = MODEL_REGISTRY[model_name]
    return predict_fn(model_dict, X_test)
