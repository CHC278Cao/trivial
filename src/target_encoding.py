
import os
import pdb

import pandas as pd
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold


class TargetEncoding(object):
    def __init__(self, df, cat_feats, target, smoothing, handle_na = True):
        self.df = df
        self.target = target
        self.cat_feats = cat_feats
        self.smoothing = smoothing

        self.target_encoder = None


    def fit_transform(self, out_fold = 5):
        oof = pd.DataFrame([])
        for tr_idx, oof_idx in StratifiedKFold(n_splits=5, shuffle=True,
                                               random_state=42).split(self.df, self.df[self.target]):
            ce_target_encoder = ce.TargetEncoder(cols=self.cat_feats, smoothing=self.smoothing)
            ce_target_encoder.fit(self.df.iloc[tr_idx, :], self.df[self.target].iloc[tr_idx])
            oof = oof.append(ce_target_encoder.transform(self.df.iloc[oof_idx, :]), ignore_index=False)

        ce_target_encoder = ce.TargetEncoder(cols=self.cat_feats, smoothing=self.smoothing)
        ce_target_encoder.fit(self.df[self.cat_feats], self.df[self.target])
        self.target_encoder = ce_target_encoder
        train_df = oof.sort_index()
        return train_df

    def transform(self, df):
        df = self.target_encoder.transform(df[self.cat_feats])
        return df


if __name__ == '__main__':
    TRAIN_PATH = '../input_II/train.csv'
    TEST_PATH = '../input_II/test.csv'
    pdb.set_trace()
    df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
    df_test = pd.read_csv(TEST_PATH).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42)
    print(df.head())
    # print(df.nunique())
    cols = [x for x in df.columns if x not in ("id", "target")]
    pdb.set_trace()
    trg_encoder = TargetEncoding(df, cols, 'target', smoothing=0.3)
    df = trg_encoder.fit_transform(out_fold=5)
    print(df.head())
    # print(df.nunique())
    test_df = trg_encoder.transform(df_test)
    print(test_df.head())