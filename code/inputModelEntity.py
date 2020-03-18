
import os

import pdb
import pandas as pd
import numpy as np
import time
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler

from src import categorical
from src import cross_validation
from .import entityModel


def get_encoding(train_df, test_df):
    test_df["target"] = -1
    full_data = pd.concat([df, df_test]).reset_index(drop=True)
    cols = [c for c in df.columns if c not in ["id", "target"]]
    for c in cols:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(full_data[c].astype(str).fillna('None').values)
        train_df[c] = lbl.transform(train_df[c].astype(str).fillna("None").values)
        test_df[c] = lbl.transform(test_df[c].astype(str).fillna('None').values)

    test_df.drop('target', axis=1, inplace=True)
    return train_df, test_df


def train_epoch(model, data_loader, optimizer, criterion, device, scheduler = None):
    model.train()
    model.to(device)
    running_loss = 0.0

    for batch_idx, data in enumerate(data_loader):
        pdb.set_trace()
        optimizer.zero_grad()
        inputs = data["data"].to(device)
        target = data["targets"].to(device)
        outputs = model(inputs)
        loss = criterion(torch.log(outputs), target.view(-1))
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
            inputs = data['data'].to(device)
            targets = data["targets"].to(device)
            outputs = model(inputs)
            loss = criterion(torch.log(outputs), targets.view(-1))
            running_loss += loss.item()

            fin_outputs.append(outputs.cpu().detach().numpy()[:, -1])
            fin_targets.append(targets.view(-1).cpu().detach().numpy())

    return running_loss/len(data_loader), np.vstack(fin_outputs), np.vstack(fin_targets)


def train(train_df, features, emb_size_dict, model_path):
    EPOCHS = 50
    LR = 0.01
    BATCH_SIZE = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for idx, (train_idx, valid_idx) in enumerate(kf.split(X = train_df, y=train_df['target'].values)):
        X_train, y_train = train_df[features].iloc[train_idx, :], train_df['target'].iloc[train_idx]
        X_valid, y_valid = train_df[features].iloc[valid_idx, :], train_df['target'].iloc[valid_idx]

        train_dataset = entityModel.entityDataset(X_train, y_train)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                        drop_last=True)

        valid_dataset = entityModel.entityDataset(X_valid, y_valid)
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                        drop_last=True)

        model = entityModel.EmbedModel(emb_size_dict, 2, 0.3, 0.2, DEVICE)
        criterion = nn.NLLLoss()
        optimizer = Adam(model.parameters(), lr = LR)

        STEP_SIZE = int(len(train_dataset) / BATCH_SIZE * 3)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.2)

        PATIENT = 3
        BEST_VALID_LOSS = float("inf")
        cnt = 0

        pdb.set_trace()
        for epoch in range(EPOCHS):
            start_time = time.time()
            train_loss = train_epoch(model, train_data_loader, optimizer, criterion, DEVICE,
                                     scheduler)
            print(f"epoch = {epoch+1}, loss = {train_loss}, time = {time.time() - start_time}")
            valid_loss, fin_outputs, fin_targets = valid_epoch(model, valid_data_loader,
                                                               criterion, DEVICE)
            print(f"epcoh = {epoch+1}, loss = {valid_loss}")

            if valid_loss < BEST_VALID_LOSS:
                BEST_VALID_LOSS = valid_loss
                cnt = 0
                torch.save(model.state_dict(), f"{model_path}/entity_{idx}.bin")
            else:
                cnt += 1
                if cnt > PATIENT:
                    print("Early stopping")
                    break

            print(metrics.roc_auc_score(fin_targets, fin_outputs))

    predict(df_test, emb_size_dict, model, model_path)


def predict(test_df, features, emb_size_dict, model_path):
    test_idx = df_test.id.values
    test_df = test_df[features]

    test_dataset = entityModel.entityDataset(test_df)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict = None
    for FOLD in range(5):
        print("current FOLD is {}".format(FOLD))

        model = entityModel.EmbedModel(emb_size_dict, 2, 0.3, 0.2, DEVICE)
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
    pdb.set_trace()
    train_df, test_df = get_encoding(df, df_test)

    features = [x for x in train_df.columns if x not in ["id", "target"]]
    emb_size_dict = []

    for c in features:
        num_features = int(df[c].nunique())
        emb_size = int(min(np.ceil(num_features / 2), 50))
        emb_size_dict.append((num_features + 1, emb_size))
    # X = get_splitdf(df)
    train(train_df, features, emb_size_dict, MODEL_PATH)

    # sub = predict(df_test, emb_size_dict, model, MODEL_PATH)
    # pdb.set_trace()
    # sub.to_csv(f"{SUBMISSION}/entity_1.csv", index=False)








