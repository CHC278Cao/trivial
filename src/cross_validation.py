
"""
    Build a new dataframe for cross validation for classification or regression
    problem_type in ("binary_classification", "multiclass_classification", "single_col_regression",
    "multi_col_regression", "multilabel_classification")
        time_series data, set shuffle = False, and choose "handout_*" as problem_type,
    and '*' represents the percentage of validation data in all training data


"""
from sklearn import model_selection
import pandas as pd

class CrossValidation(object):
    def __init__(self, df, target_cols, shuffle, problem_type=None,
                 multilabel_delimiter=None, num_folds=5, random_state=42):
        """
            Create cross validation data for original dataframe
        :param df: type: DataFrame, train dataframe
        :param target_cols: type: str, target column label
        :param shuffle: type: bool, shuffle data
        :param problem_type: type: str, classification or regression
        :param multilabel_delimiter: type: str, delimiter for spliting multi-labels
        :param num_folds: type: int, number of folders for cross validation
        :param random_state: type: int, random_state
        """
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(self.target_cols)
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state

        if problem_type is None:
            raise Exception("problem type shouldn't be None")
        self.problem_type = problem_type

        if multilabel_delimiter is not None:
            self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        self.dataframe['kfold'] = -1

    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=False)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")

            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)

            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1

        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")

            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        else:
            raise Exception("problem type is not defined")

        return self.dataframe


if __name__ == "__main__":
    df = pd.read_csv("../input_II/train.csv")
    cv = CrossValidation(df, shuffle=True, target_cols=["target"],
                         problem_type="binary_classification")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
    df_split.to_csv("../input_II/train_folds.csv", index=False)