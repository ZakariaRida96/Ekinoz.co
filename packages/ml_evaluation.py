import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluation_metrics(
    model,
    feature_train: pd.DataFrame,
    feature_test: pd.DataFrame,
    label_train: pd.DataFrame,
    label_test: pd.DataFrame,
):

    # Prédiction de class
    label_test_class = model.predict(feature_test)
    label_train_class = model.predict(feature_train)

    # Prédiction des probabilité
    predictions_test = [round(value) for value in label_test_class]
    predictions_train = [round(value) for value in label_train_class]

    # accuracy du training et du test
    accuracylabel_train = accuracy_score(label_train, predictions_train)
    print(f"Accuracy train: {accuracylabel_train* 100}%")

    accuracylabel_test = accuracy_score(label_test, predictions_test)
    print(f"Accuracy test: {accuracylabel_test* 100}%")

    # AUC du test
    label_test_proba = model.predict_proba(feature_test)
    label_test_proba = label_test_proba[:, 1]
    xgb_auc = roc_auc_score(label_test, label_test_proba)

    print(f"AUC: {xgb_auc* 100}%")


def features_importance_xgb(model, var_cols: list):

    df_var_imp = pd.DataFrame(
        {"Variable": var_cols, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(15, 8))
    plt.title("Feature Importance")
    img = sns.barplot(data=df_var_imp.head(10), x="Importance", y="Variable")
    plt.savefig("./figures/features_importances.png")
    return df_var_imp, img
