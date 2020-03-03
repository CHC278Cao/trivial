
import os

import pdb
import pandas as pd
import numpy as np
import time
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler

from src import categorical
from src import cross_validation
from .import entityModel

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
    # pdb.set_trace()

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


def train_epoch(model, data_loader, optimizer, criterion, device, scheduler = None):
    model.train()
    model.to(device)
    running_loss = 0.0

    for batch_idx, data in enumerate(data_loader):
        # pdb.set_trace()
        optimizer.zero_grad()
        out = model(data)
        target = data["target"].to(device)
        loss = criterion(out, F.one_hot(target).float())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (batch_idx % 1000) == 999:
            print(f"batch_idx = {batch_idx+1}, loss = {loss.item()}")

    return running_loss / len(data_loader)

def valid_epoch(model, data_loader, criterion, device):
    model.eval()
    model.to(device)
    fin_targets = []
    fin_outputs = []
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            outputs = model(data)
            targets = data["target"].to(device)
            loss = criterion(outputs, F.one_hot(targets).float())

            running_loss += loss.item()
            fin_outputs.append(outputs.cpu().detach().numpy()[:, -1])
            fin_targets.append(targets.view(-1).cpu().detach().numpy())

    return running_loss/len(data_loader), np.vstack(fin_outputs), np.vstack(fin_targets)


def train(df, df_test, model_path):
    EPOCHS = 50
    LR = 0.01
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    features = [x for x in df.columns if x not in ["id", "target", "kfold"]]
    emb_size_dict = {}
    for c in features:
        num_features = int(df[c].nunique())
        emb_size = int(min(np.ceil(num_features / 2), 50))
        emb_size_dict[c] = (num_features + 1, emb_size)

    pdb.set_trace()

    for FOLD in range(5):
        train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
        valid_df = df[df.kfold == FOLD].reset_index(drop=True)


        train_dataset = entityModel.entityDataset(train_df, emb_size_dict.keys(), 'target')
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        valid_dataset = entityModel.entityDataset(valid_df, emb_size_dict.keys(), 'target')
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = entityModel.EmbedModel(emb_size_dict, 2, 0.3, 0.2, DEVICE)
        criterion = nn.BCELoss()
        optimizer = Adam(model.parameters(), lr = LR)

        STEP_SIZE = int(len(train_dataset) / BATCH_SIZE * 3)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.9)

        PATIENT = 3
        BEST_VALID_LOSS = float("inf")
        cnt = 0

        pdb.set_trace()
        for epoch in range(EPOCHS):
            start_time = time.time()
            train_loss = train_epoch(model, train_data_loader, optimizer, criterion, DEVICE, scheduler)
            print(f"epoch = {epoch+1}, loss = {train_loss}, time = {time.time() - start_time}")
            valid_loss, fin_outputs, fin_targets = valid_epoch(model, valid_data_loader,
                                                               criterion, DEVICE)
            print(f"epcoh = {epoch+1}, loss = {valid_loss}")

            if valid_loss < BEST_VALID_LOSS:
                BEST_VALID_LOSS = valid_loss
                cnt = 0
                torch.save(model.state_dict(), f"{model_path}/entity_{FOLD}.bin")
            else:
                cnt += 1
                if cnt > PATIENT:
                    print("Early stopping")
                    break

            print(metrics.roc_auc_score(fin_targets, fin_outputs))

    predict(df_test, emb_size_dict, model, model_path)


def predict(df_test, emb_size_dict, model, model_path):
    test_idx = df_test.id.values
    test_dataset = entityModel.entityDataset(df_test, emb_size_dict.keys())
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    predict = None
    for FOLD in range(5):
        print("current FOLD is {}".format(FOLD))
        model.load_state_dict(torch.load(f"{model_path}/entity_{FOLD}.bin", map_location=torch.device('cpu')))
        model.eval()
        fold_predict = []
        for batch_idx, data in enumerate(test_data_loader):
            out = model(data)
            preds = out.cpu().detach().numpy()[:, 1].reshape(-1, 1)
            fold_predict.append(preds)

        fold_predict = np.vstack(fold_predict)

        if predict is None:
            predict = fold_predict
        else:
            predict += fold_predict



    predict /= 5
    sub = pd.DataFrame(np.column_stack((test_idx, predict)), columns=["id", "target"])
    sub["id"] = sub["id"].astype("int")
    return sub

if __name__ == '__main__':

    TRAIN_PATH = os.environ.get("TRAIN_PATH")
    TEST_PATH = os.environ.get("TEST_PATH")
    SUBMISSION = os.environ.get("SUBMISSION")
    # MODEL_TYPE = os.environ.get("MODEL")
    MODEL_PATH = os.environ.get("MODEL_PATH")


    df = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # train(df=df, model_type=MODEL_TYPE, model_path=MODEL_PATH)
    # sub = predict(df_test=df_test, model_path=MODEL_PATH, model_type=MODEL_TYPE)
    df, df_test = get_encoding(df, df_test)
    # X = get_splitdf(df)

    # train(X, df_test, MODEL_PATH)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = [x for x in df.columns if x not in ["id", "target", "kfold"]]
    emb_size_dict = {}
    for c in features:
        num_features = int(df[c].nunique())
        emb_size = int(min(np.ceil(num_features / 2), 50))
        emb_size_dict[c] = (num_features + 1, emb_size)

    model = entityModel.EmbedModel(emb_size_dict, 2, 0.3, 0.2, DEVICE)

    sub = predict(df_test, emb_size_dict, model, MODEL_PATH)
    pdb.set_trace()
    sub.to_csv(f"{SUBMISSION}/entity_1.csv", index=False)








