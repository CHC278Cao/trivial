import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

# test_data_path = os.environ.get("TEST_DATA")
# MODEL = os.environ.get("MODEL")

def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    test_idx = df["id"].values
    predict = None

    for FOLD in range(5):
        print("current FOLD is {}".format(FOLD))
        df = pd.read_csv(test_data_path)
        encoders = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))
        for c in encoders:
            print(c)
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df[c].values.tolist())


        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predict = preds
        else:
            predict += preds

    predict /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predict)), columns=["id", "target"])
    sub["id"] = sub["id"].astype("int")
    return sub

if __name__ == "__main__":
     submission = predict(test_data_path = "data/test.csv", model_type="randomfroest",
                          model_path = "models")
     submission.loc[:, "id"] = submission.loc[:, "id"].astype('int')
     submission.to_csv("models/lr_submssion.csv", index=False)