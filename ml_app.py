import pickle as pk

import pandas as pd
from sklearn.pipeline import Pipeline

from packages.feature_prepocessing import feature_engineering
from packages.label_prepocessing import CheckMissing, LabelClass, TransformLabel  # noqa
from packages.ml_evaluation import evaluation_metrics, features_importance_xgb
from packages.ml_load_score import apply_predict_raw
from packages.ml_training import xgb_gridsearch
from packages.split_train_test import split_data


def main(path):
    dataset = pd.read_csv(path)

    # Définition de la pipeline qui traite le label
    pipeline_label = Pipeline(
        [
            ("transformlabel", TransformLabel()),
            ("missingchecker", CheckMissing()),
            ("displaylabel", LabelClass()),
        ]
    )

    # Fit de la pipeline sur la base
    df_labeled = pipeline_label.fit_transform(dataset)

    # Création des tables feature et label
    X = df_labeled.drop("Target", axis=1)
    y = df_labeled.Target

    # Création des instances
    X_featured, var_name = feature_engineering(df_input=X)
    # Sauvegarde de la table X_featured
    X_featured.to_pickle("./data/processed/X_featured.pkl")

    # Split train et test
    X_train, X_test, y_train, y_test = split_data(
        df_feature=X_featured,
        df_label=y,
    )

    # Entrainement du modele xgboost sur les hyperparametre optimaux
    model_xgb = xgb_gridsearch(
        feature_train=X_train,
        feature_test=X_test,
        label_train=y_train,
        label_test=y_test,
    )

    # Evaluation du modele
    evaluation_metrics(
        model=model_xgb,
        feature_train=X_train,
        feature_test=X_test,
        label_train=y_train,
        label_test=y_test,
    )

    # Feature importance score
    df_feature_importance, plot = features_importance_xgb(
        model=model_xgb, var_cols=var_name
    )

    # Sauvgarde du modele sous format pickle
    pk.dump(model_xgb, open("./model/model_xgb.pkl", "wb"))

    # Application des prédiction sur la base source
    apply_predict_raw()
