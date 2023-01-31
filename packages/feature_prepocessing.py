import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


def feature_engineering(df_input: pd.DataFrame) -> pd.DataFrame:

    # liste des variable
    var_cols = df_input.columns.values.tolist()

    # Description des valeurs manquantes sur la table
    print(
        "Comptage des valeurs manquantes sur la variable Target:",
        df_input.isnull().sum(),
    )

    # Feature engineering sur les features
    # Numeric -> MinMaxScaler
    # categorical -> OrdinalEncoder

    numerical_columns_selector = make_column_selector(dtype_exclude=object)
    categorical_columns_selector = make_column_selector(dtype_include=object)

    numerical_columns = numerical_columns_selector(df_input)
    categorical_columns = categorical_columns_selector(df_input)

    categorical_preprocessor = OrdinalEncoder()
    numerical_preprocessor = MinMaxScaler()

    feature_processing = ColumnTransformer(
        [
            ("ordinal-encoder", categorical_preprocessor, categorical_columns),
            ("minmax_scaler", numerical_preprocessor, numerical_columns),
        ]
    )

    # Application des transformation et jointure
    feature_processing.set_output(transform="pandas")
    df_featured = feature_processing.fit_transform(df_input)

    return df_featured, var_cols
