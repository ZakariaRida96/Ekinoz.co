from sklearn.base import BaseEstimator, TransformerMixin


# Création d'une classe qui permet la variable cible d'une binary,
# description des valeurs manquante,
# encodage du label et compte du label.
class TransformLabel(BaseEstimator, TransformerMixin):
    def fit(self, y=None):
        return self

    def transform(self, X):

        # if FinalGrade gt 10 --> Target= 0 else 1
        X["Target"] = X["FinalGrade"].apply(
            lambda x: "Pass" if x >= 10 else "Fail"
        )  # noqa
        X["Target"] = X["Target"].map(lambda v: 0 if v == "Pass" else 1)

        # Suppression des variables: studentid,firstname,familyname
        X.drop(["StudentID", "FirstName", "FamilyName", "FinalGrade"], axis=1)

        return X.drop(["StudentID", "FirstName", "FamilyName"], axis=1)  # noqa


class CheckMissing(BaseEstimator, TransformerMixin):
    def fit(self, y=None):
        return self

    def transform(self, X):
        print(
            "\nComptage des valeurs manquantes sur la variable Target:\n",
            X["Target"].isnull().sum(),
        )

        return X


class LabelClass(BaseEstimator, TransformerMixin):
    def fit(self, y=None):
        return self

    def transform(self, X):
        counts = X["Target"].value_counts(normalize=True).mul(100).round(1)

        print("\nComptage des classes du label :\n", counts)

        return X
