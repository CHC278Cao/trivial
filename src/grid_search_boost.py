
import os
import pdb
import joblib
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
from catboost import CatBoostClassifier, Pool
from target_encoding import TargetEncoding

class BoostModel(object):
    def __init__(self, save_path, cv = 5, random_state = 42):
        """
            Generate a boostModel
        :param cv: number of folders
        :param random_state: random_state
        :param save_path: path for saving model
        """
        self.cv = cv
        self.random_state = random_state
        self.save_path = save_path

        self.model = None
        self.param = None
        self.model_type = None
        self.cat_feature = None
        self.output = None

    def set_model(self, model_type, param, cat_feature = None):
        self.model_type = model_type
        if model_type == "catboost":
            self.model = cb.CatBoostClassifier(**param)
        # elif self.model_type == "xgboost":
        #     # self.model = xgb.
        elif self.model_type == "lightboost":
            self.model = lgbm.LGBMClassifier(**param)
        else:
            raise Exception("model should be in xgboost, lightboost, catboost")

        self.cat_feature = cat_feature

    def _catboost_fit(self, df, target, test_df):
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X=df,
                                                                y=target)):
            X_train, y_train = df.iloc[train_idx, :], target[train_idx]
            X_valid, y_valid = df.iloc[valid_idx, :], target[valid_idx]

            # X_train = X_train.astype("str")
            # X_valid = X_valid.astype("str")

            train = Pool(data=X_train, label=y_train,
                         feature_names=list(X_train.columns), cat_features=self.cat_feature)

            valid = Pool(data=X_valid, label=y_valid,
                         feature_names=list(X_valid.columns), cat_features=self.cat_feature)

            self.model.fit(train, eval_set=valid, verbose_eval=100,
                           early_stopping_rounds=100, use_best_model=True)

            output = self.model.predict_proba(X_valid)
            self._metrix_check(y_valid, output)

            # joblib.dump(self.model, f"{self.save_path}/{self.model_type}_{fold_idx}.pkl")

    def _lgb_fit(self, df, target):
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X=df,
                                                                y=target)):
            X_train, y_train = df.iloc[train_idx, :], target[train_idx]
            X_valid, y_valid = df.iloc[valid_idx, :], target[valid_idx]

            self.model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                                   verbose=500, eval_metric='auc', early_stopping_rounds=100)

            output = self.model.predict_proba(X_valid)
            self._metrix_check(y_valid, output)

            # joblib.dump(self.model, f"{self.save_path}/{self.model_type}_{fold_idx}.pkl")

    # def _xgb_fit(self, df, target):


    def fit(self, df, target):
        if self.model_type == "lightboost":
            self._lgb_fit(df, target)
        elif self.model_type == "catboost":
            self._catboost_fit(df, target)
        else:
            raise Exception("model_type is not defined")


    def _metrix_check(self, target, output):
        pdb.set_trace()
        if output.shape[1] <= 2:
            print("roc_auc_score is {}".format(metrics.roc_auc_score(target, output[:, 1])))

        preds = np.argmax(output, axis=1).reshape((-1, 1))
        print("accuracy is {}".format(metrics.accuracy_score(target, preds)))
        print("F1 score is {}".format(metrics.f1_score(target, preds)))

    def predict(self, df):
        cols = [x for x in df.columns if x not in ["id"]]
        test_data = df[cols]
        if self.model_type == "catboost":
            test_data = test_data.astype("str")
            test_data = Pool(data=test_data, feature_names=cols, cat_features=self.cat_feature)

        out = None
        for fold_idx in range(self.cv):
            model = joblib.load(f"{self.save_path}/{self.model_type}_{fold_idx}.pkl")
            output = model.predict_proba(test_data)[:, 1]

            if out is None:
                out = output
            else:
                out += output

        out /= 5




def label_preprocess(train_df, test_df):
    train_len = len(train_df)
    test_df["target"] = -1
    df = pd.concat((train_df, test_df), axis=0)
    cols = [x for x in train_df.columns if x not in ("id", "target")]
    for x in cols:
        lbl = preprocessing.LabelEncoder()
        df.loc[:, x] = df.loc[:, x].astype(str).fillna("None")
        lbl.fit(df[x].values)
        df.loc[:, x] = lbl.transform(df[x].values)

    train_df = df.iloc[:train_len, :]
    test_df = df.iloc[train_len:, :]
    return train_df, test_df





if __name__ == "__main__":
    TRAIN_PATH = '../input_II/train.csv'
    TEST_PATH = '../input_II/test.csv'
    pdb.set_trace()
    df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
    df_test = pd.read_csv(TEST_PATH).reset_index(drop=True)
    df, df_test = label_preprocess(df, df_test)
    df = df.sample(frac=1, random_state=42)
    print(df.head())
    # print(df.nunique())
    cols = [x for x in df.columns if x not in ("id", "target")]



        # elif df[x].dtypes == 'int':
        #     print()

    pdb.set_trace()
    trg_encoder = TargetEncoding(df, cols, 'target', smoothing=0.3)
    df = trg_encoder.fit_transform(out_fold=5)
    print(df.head())
    # print(df.nunique())
    test_df = trg_encoder.transform(df_test)
    print(test_df.head())

    boostmodel = BoostModel(save_path='../input_II')

    # model_type = "lightboost"
    model_type = "catboost"

    lgb_params = {
        'learning_rate': 0.05,
        'feature_fraction': 0.1,
        'min_data_in_leaf': 12,
        'max_depth': 3,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'objective': 'binary',
        'metric': 'auc',
        'n_jobs': -1,
        'n_estimators': 200,
        'feature_fraction_seed': 42,
        'bagging_seed': 42,
        'boosting_type': 'gbdt',
        'verbose': 1,
        'is_unbalance': True,
        'boost_from_average': False}

    cat_params = {'bagging_temperature': 0.8,
                   'depth': 5,
                   'iterations': 1000,
                   'l2_leaf_reg': 3,
                   'learning_rate': 0.03,
                   'random_strength': 0.8,
                    'loss_function': 'Logloss',
                    'eval_metric': 'AUC',
                    'nan_mode': 'Min',
                    'thread_count': 4,
                    'verbose': False}
    pdb.set_trace()
    boostmodel.set_model(model_type=model_type, param = cat_params, cat_feature=cols)
    # boostmodel.set_model(model_type=model_type, param=lgb_params)

    train_df = df[cols]
    target = df["target"].values

    boostmodel.fit(train_df, target=target)


