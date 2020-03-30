
import os
import pdb
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummaryX import summary

from res34Net import Resnet34_classification, load_pretained_weights
from dataset import ImgDataset
from res34Unet import res34Unet
from metrics import DiceBCELoss
from meter import Meter

warnings.filterwarnings('ignore')


def train_epoch(model, data_loader, optimizer, device, meter, schedule = None):
    model.train()
    running_loss = 0.0

    for batch_idx, data in enumerate(data_loader):
        optimizer.zero_grad()
        inputs = data["img"].to(device)
        labels = data["label"].to(device)
        targets = data["mask"].to(device)

        inputs = inputs.permute(0, 3, 1, 2)
        targets = targets.permute(0, 3, 1, 2).contiguous()
        cls_out, seg_out = model(inputs)
        cls_loss = F.binary_cross_entropy(torch.sigmoid(cls_out.squeeze()), labels, reduction = "mean")
        seg_loss = DiceBCELoss(torch.sigmoid(seg_out), targets)
        loss = cls_loss + seg_loss
        loss.backward()
        running_loss += loss.item()
        meter.update(seg_out.detach().cpu(), targets.detach().cpu())
        optimizer.step()
        if schedule is not None:
            schedule.step()

        if (batch_idx % 100) == 99:
            print("batch_idx = {}, Loss = {}".format(batch_idx+1, loss.item()))

        del data

    return running_loss / len(data_loader)


def valid_epoch(model, data_loader, device, meter):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            inputs = data["img"].to(device)
            labels = data["label"].to(device)
            targets = data["mask"].to(device)

            inputs = inputs.permute(0, 3, 1, 2)
            targets = targets.permute(0, 3, 1, 2).contiguous()
            cls_out, seg_out = model(inputs)
            cls_loss = F.binary_cross_entropy(cls_out.squeeze(), labels, reduction = "mean")
            seg_loss = DiceBCELoss(torch.sigmoid(seg_out), targets)
            loss = cls_loss + seg_loss
            running_loss += loss.item()
            meter.update(targets.detach().cpu(), seg_out.detach().cpu())

            del data

    return running_loss / len(data_loader)


def detect_img(data_loader):
    pdb.set_trace()

    row = 10
    col = 2
    plt.figure(figsize=(80, 100))
    for idx, data in enumerate(data_loader):
        # pdb.set_trace()
        if idx >= row:
            break
        print(data["label"])
        plt.subplot(row, col, 2*idx+1)
        img = np.squeeze(data["img"].numpy())
        plt.imshow(img, cmap="gray")
        plt.subplot(row, col, 2*idx+2)
        new_img = np.squeeze(data["new_img"].numpy())
        plt.imshow(new_img, cmap="gray")
        # plt.subplot(row, col, 2 * idx + 3)
        # mask1 = np.squeeze(data["mask"].squeeze().numpy()[:, :, 0])
        # plt.imshow(mask1, cmap="gray")
        # plt.subplot(row, col, 2 * idx + 4)
        # mask2 = np.squeeze(data["mask"].squeeze().numpy()[:, :, 1])
        # plt.imshow(mask2, cmap="gray")
        # plt.subplot(row, col, 2 * idx + 5)
        # mask3 = np.squeeze(data["mask"].squeeze().numpy()[:, :, 2])
        # plt.imshow(mask3, cmap="gray")
        # plt.subplot(row, col, 2 * idx + 6)
        # mask4 = np.squeeze(data["mask"].squeeze().numpy()[:, :, 3])
        # plt.imshow(mask4, cmap="gray")
    plt.show()



def train(df, img_dir, pretrained_file = None):
    pdb.set_trace()

    # set up dataset for training
    df = df.sample(frac=1., random_state = 42)
    train_number = int(len(df) * 0.8)
    train_df = df.iloc[:train_number, :]
    valid_df = df.iloc[train_number:, :]
    del df

    cols = ["label_" + str(idx) for idx in range(1, 5)]
    train_dataset = ImgDataset(train_df["imageId"].values, img_dir, mask_list = train_df[cols])
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )
    valid_dataset = ImgDataset(valid_df["imageId"].values, img_dir, mask_list = train_df[cols])
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=4,
        shuffle=True,
    )

    # set up model parameters

    model = res34Unet(num_classes=4)
    print(model)
    pdb.set_trace()
    if pretrained_file is not None:
        skip = ['block.5.weight', 'block.5.bias']
        load_pretained_weights(model, pretained_file, skip=skip, first_layer=["block.0.0.weight"])
    summary(model, torch.zeros(2, 1, 224, 224))

    LR = 3e-4
    optimizer = optim.Adam(model.parameters(), lr = LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)

    # set up train parameters
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVER = "./model_data/model.bin"
    EPOCHS = 50
    CNT = 0
    BEST_VALID_LOSS = float("inf")
    PATIENCE = 5
    train_meter = Meter(phase = "train")
    valid_meter = Meter(phase = "valid")

    model.to(DEVICE)
    for epoch in range(EPOCHS):
        st_time = time.time()
        train_loss = train_epoch(model, train_data_loader, optimizer,
                                                              DEVICE, train_meter, schedule = None)
        current_time = time.time()
        train_meter.epoch_log(epoch, train_loss, current_time - st_time)
        valid_loss = valid_epoch(model, valid_data_loader, DEVICE, valid_meter)
        valid_meter.epoch_log(epoch, valid_loss, time.time() - current_time)
        scheduler.step(valid_loss)

        if valid_loss < BEST_VALID_LOSS:
            CNT = 0
            BEST_VALID_LOSS = valid_loss
            torch.save(model.state_dict(), SAVER)
        else:
            CNT += 1
            if CNT >= PATIENCE:
                print("Early stopping ... ")
            break


if __name__ == "__main__":
    trainfile = os.environ.get("TRAINFILE")
    # trainfile = "../train.csv"
    train_df = pd.read_csv(trainfile).reset_index(drop=True)
    img_dir = os.environ.get('IMAGEDIR')
    # img_dir = '../train_images'
    # pretained_file = "./model_data/resnet34.pth"

    pretained_file = os.environ.get("PRETRAINEDFILE")

    print("train_df shape is {}".format(len(train_df)))
    print(train_df.head())
    print(len(train_df["ImageId"].unique()))

    train_df = train_df.pivot(index="ImageId", columns="ClassId",
                              values="EncodedPixels").rename_axis(None, axis=1).reset_index()
    title =["imageId"] + ["label_" + str(idx) for idx in range(1, 5)]
    train_df.columns = title
    train_df = train_df.fillna("None")

    print(train_df.head())

    train(train_df, img_dir, pretrained_file=pretained_file)