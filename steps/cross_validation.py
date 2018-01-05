import pandas as pd

from sklearn.model_selection import train_test_split


def train_test_split_atleast_one(X, y, label_column='whaleID', test_size=0.2, random_state=1234):
    label = y[[label_column]]
    unique_index = label.sort_index().groupby(label_column).filter(lambda group: len(group) == 1)[label_column].index
    non_unique_index = label.sort_index().groupby(label_column).filter(lambda group: len(group) != 1)[
        label_column].index

    X_non_unique, y_non_unique = X.iloc[non_unique_index], y.iloc[non_unique_index]
    X_unique, y_unique = X.iloc[unique_index], y.iloc[unique_index]

    X_train, X_test, y_train, y_test = train_test_split(X_non_unique, y_non_unique, test_size=test_size,
                                                        random_state=random_state, stratify=y_non_unique[label_column])

    for obj in [X_train, X_test, y_train, y_test]:
        obj.reset_index(drop=True, inplace=True)

    X_train_ = pd.concat([X_train, X_unique], axis=0).reset_index(drop=True)
    y_train_ = pd.concat([y_train, y_unique], axis=0).reset_index(drop=True)

    return X_train_, X_test, y_train_, y_test
