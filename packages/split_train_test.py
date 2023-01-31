import numpy as np
from sklearn.model_selection import train_test_split


def split_data(df_feature: np.array, df_label: np.array) -> np.array:
    # Split train et validation et test
    X_train, X_test, y_train, y_test = train_test_split(
        df_feature, df_label, test_size=0.2, random_state=8
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test
