
import os
import joblib

import pdb
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics

from src import categorical
from src import cross_validation
from src import dispatcher

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

def get_encoding(df, df_test):
    train_len = len(df)
    df_test["target"] = -1
    full_data = pd.concat([df, df_test])
    cols = [c for c in df.columns if c not in ["id", "target"]]
    encoding_type = "label"
    cat_feats = categorical.CategoricalFeatures(full_data,
                                                categorical_features=cols,
                                                encoding_type=encoding_type,
                                                handle_na=True
                                                )
    pdb.set_trace()

    full_data_transformed = cat_feats.fit_transform()
    if encoding_type == "label":
        X = full_data_transformed.iloc[:train_len, :]
        X_test = full_data_transformed.iloc[train_len:, :]
        return X, X_test

    elif encoding_type == 'ohe':
        X = full_data_transformed[:train_len, :]
        X_test = full_data_transformed[train_len:, :]
        ytrain = df.target.values
        return X, ytrain, X_test

def get_splitdf(df):

    cv = cross_validation.CrossValidation(df, shuffle=True, target_cols=["target"],
                        problem_type="binary_classification")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
    return df_split

def sparse_train(X, ytrain, model_path, model_type):
    pdb.set_trace()
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    for FOLD, (train_idx, valid_idx) in enumerate(kf.split(X=X, y=ytrain)):
        clf = dispatcher.MODELS[model_type]
        train_X, train_y = X[train_idx], ytrain[train_idx]
        valid_X, valid_y = X[valid_idx], ytrain[valid_idx]
        clf.fit(train_X, train_y)
        preds = clf.predict_proba(valid_X)[:, 1]
        print(metrics.roc_auc_score(valid_y, preds))

        joblib.dump(clf, f"{model_path}/{model_type}_{FOLD}.pkl")


def sparse_predict(X_test, df_test, model_path, model_type):
    pdb.set_trace()
    test_idx = df_test["id"].values
    predict = None

    for FOLD in range(5):
        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        preds = clf.predict_proba(X_test)[:, 1]
        if FOLD == 0:
            predict = preds
        else:
            predict += preds

    predict /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predict)), columns=["id", "target"])
    sub["id"] = sub["id"].astype("int")
    return sub


def train(df, model_path, model_type):

    for FOLD in range(5):
        train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
        valid_df = df[df.kfold == FOLD].reset_index(drop=True)

        ytrain = train_df.target.values
        yvalid = valid_df.target.values

        train_df.drop(["id", "target", "kfold"], axis=1, inplace=True)
        valid_df.drop(["id", "target", "kfold"], axis=1, inplace=True)

        valid_df = valid_df[train_df.columns]
        pdb.set_trace()
        clf = dispatcher.MODELS[model_type]
        clf.fit(train_df, ytrain)
        preds = clf.predict_proba(valid_df)[:, 1]
        print(metrics.roc_auc_score(yvalid, preds))

        joblib.dump(clf, f"{model_path}/{model_type}_{FOLD}.pkl")
        joblib.dump(train_df.columns, f"{model_path}/{model_type}_{FOLD}_columns.pkl")

def predict(df_test, model_path, model_type):
    test_idx = df_test["id"].values
    predict = None

    for FOLD in range(5):
        pdb.set_trace()
        print("current FOLD is {}".format(FOLD))
        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))
        df_test = df_test[cols]
        preds = clf.predict_proba(df_test)[:, 1]
        if FOLD == 0:
            predict = preds
        else:
            predict += preds

    predict /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predict)), columns=["id", "target"])
    sub["id"] = sub["id"].astype("int")
    return sub

if __name__ == '__main__':
    TRAIN_PATH = os.environ.get("TRAIN_PATH")
    TEST_PATH = os.environ.get("TEST_PATH")
    SUBMISSION = os.environ.get("SUBMISSION")
    MODEL_TYPE = os.environ.get("MODEL")
    MODEL_PATH = os.environ.get("MODEL_PATH")

    pdb.set_trace()
    df = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    # sample = pd.read_csv(SUBMISSION
    # df, df_test = get_encoding(df, df_test)
    # df= get_splitdf(df)
    #
    # train(df=df, model_type=MODEL_TYPE, model_path=MODEL_PATH)
    # sub = predict(df_test=df_test, model_path=MODEL_PATH, model_type=MODEL_TYPE)
    X, X_test = get_encoding(df, df_test)
    # sparse_train(X, ytrain, model_path=MODEL_PATH, model_type=MODEL_TYPE)
    # sub = sparse_predict(X_test=X_test, df_test=df_test, model_path=MODEL_PATH, model_type=MODEL_TYPE)
    pdb.set_trace()
    # sub.to_csv(f"{SUBMISSION}/{MODEL_TYPE}_1.csv", index=False)








