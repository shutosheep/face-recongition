#! /usr/bin/env python3.7
# -*- coding: utf-8 -*-

import datetime
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.offsetbox import OffsetImage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()
imageFolder = './images'
imageCroppedFolder = './images-cropped'
plotFolder = './plot'

def getFilename(filePath):
    return filePath.split('/')[-1]

def tensorToNumpy(tensor):
    return tensor.to('cpu').detach().numpy().copy()

def featureVector(imagePath):
    image = Image.open(imagePath)
    imageCropped = mtcnn(image, save_path=os.path.join(imageCroppedFolder, getFilename(imagePath))) # MTCNNで顔検出、切り取った160x160の画像を保存
    featureVector = resnet(imageCropped.unsqueeze(0)) # resnetで特徴ベクトルを取得

    return tensorToNumpy(featureVector.squeeze())

def imgScatter(x, y, image_path, ax=None, zoom=1): 
    if ax is None: 
        ax = plt.gca() 

    artists = [] 

    for x0, y0,image in zip(x, y,image_path): 
        image = plt.imread(image) 
        im = OffsetImage(image, zoom=zoom) 
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False) 
        artists.append(ax.add_artist(ab)) 

    return artists

def prepFolder():
    os.makedirs(imageCroppedFolder, exist_ok=True)
    os.makedirs(plotFolder, exist_ok=True)
    
def main():
    if len(sys.argv) < 2:
        print('Please specify the numbers of clusters')
        quit()

    imagePath = glob.glob(f"{imageFolder}/*")
    features = [] # numpyでやったほうがいいかも?

    for image in imagePath:
        features.append(featureVector(image))

    features = np.array(features)
    print(features.shape)

    # 主成分分析で多次元ベクトルを二次元にする
    pca = PCA(n_components=2)
    pca.fit(features)
    reduced = pca.fit_transform(features)
    print(reduced.shape) 

    # K-means法を用いてクラスタリング
    
    K = int(sys.argv[1])
    kmeans = KMeans(n_clusters=K).fit(reduced)
    pred_label = kmeans.predict(reduced)
    print(pred_label)

    # 視覚化

    x = reduced[:, 0]
    y = reduced[:, 1]

    # 画像なし版
    #plt.scatter(x, y, c=pred_label)
    #plt.colorbar()
    #plt.savefig(os.path.join(plotFolder, f"{fileName}.png"))
    # plt.show()

    # 画像あり版
    imageCroppedPath = glob.glob(f"{imageCroppedFolder}/*")
    fig, ax = plt.subplots() 
    imgScatter(x, y, imageCroppedPath, ax=ax,  zoom=.2) 
    ax.plot(x, y, 'ko',alpha=0) 
    ax.autoscale()

    dt_now = datetime.datetime.now()
    fileName = dt_now.strftime('%s')

    plt.savefig(os.path.join(plotFolder, f"{fileName}.png"))

    plt.show()

    return

if __name__ == "__main__":
    prepFolder()
    main()