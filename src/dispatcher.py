from sklearn import ensemble
from sklearn import linear_model

MODELS = {
    # "randomforest": ensemble.RandomForestClassifier(n_jobs=1, verbose=2),
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_jobs=1, verbose=2),
    "logisticregression": linear_model.LogisticRegression(solver="lbfgs", C=0.1),
}