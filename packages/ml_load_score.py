import pickle as pk

import pandas as pd


def apply_predict_raw() -> pd.DataFrame:
    # Importation des bases et du modele
    raw_data = pd.read_csv("./data/raw/student_data.csv")
    X_featured = pd.read_pickle("./data/processed/X_featured.pkl")
    pickled_model = pk.load(open("./model/model_xgb.pkl", "rb"))

    # Prédiction
    probability = pickled_model.predict_proba(X_featured)[:, 1]
    # Récupération des probas
    probability_frame = pd.DataFrame(probability, columns=["score"])
    raw_data["score"] = probability_frame
    raw_data["score"] = (raw_data["score"]) * 100

    raw_data.to_pickle("./data/processed/student_data_score.pkl")
