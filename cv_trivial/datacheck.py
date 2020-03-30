
import os
import pdb
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


if __name__ == "__main__":
    trainfile = "../train.csv"

    train_df = pd.read_csv(trainfile).reset_index(drop=True)
    print(train_df.head())


    img_dir = "../train_images"
    imgIndex = train_df.loc[5, "ImageId"]
    label_list = train_df[train_df["ImageId"] == imgIndex]["ClassId"].values
    imgpath = os.path.join(img_dir, imgIndex)

    pdb.set_trace()
    print(label_list)
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(imgpath)

    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap="gray")
    plt.subplot(2, 1, 2)
    plt.imshow(img2)
    plt.show()

    print(img.shape)
    masks = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)





