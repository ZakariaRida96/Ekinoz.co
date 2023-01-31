import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# Application du xgboost et Gridsearchcv
def xgb_gridsearch(
    feature_train: pd.DataFrame,
    feature_test: pd.DataFrame,
    label_train: pd.DataFrame,
    label_test: pd.DataFrame,
    n_folds=5,
):
    # hyperparametre du tuning
    xgb_hyperparameters = {
        "booster": ["gbtree"],
        "objective": ["binary:logistic"],
        "eta": [0.001, 0.01, 0.1],
        "base_score": [0.1, 0.2],
        "max_depth": [2, 3, 4],
        "min_child_weight": [2, 5, 7],
        "subsample": [0.1, 0.3, 0.8],
        "colsample_bytree": [0.1, 0.3, 0.5],
        "n_estimators": [500, 800, 1000],
        "eval_metric": ["auc"],
        "early_stopping_rounds": [10],
    }

    xgb_classifier = xgb.XGBClassifier()

    xgb_grid = GridSearchCV(
        xgb_classifier, xgb_hyperparameters, cv=n_folds, n_jobs=4, verbose=True
    )

    xgb_grid.fit(
        feature_train,
        label_train,
        eval_set=[(feature_train, label_train), (feature_test, label_test)],
    )
    print("-" * 20)
    print(f"Moyenne du score AUC: {100*(xgb_grid.best_score_).round(1)}%")
    print("-" * 20)
    print("Les parametre optimaux sont:", (xgb_grid.best_params_))
    return xgb_grid.best_estimator_
