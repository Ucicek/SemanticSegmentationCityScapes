from PIL import Image
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans

def LoadImage(name, path, rotation=0.0, flip=False, 
                cut_bottom=58,size=(256, 200)):
    """
    Loads image
    """
    img = Image.open(str(path)+"/"+str(name))
    img = np.array(img)
    seg = img[:-cut_bottom, 256:]
    img = img[:-cut_bottom, 0:256]

    for i in range(3):
        zimg = img[:,:,i]
        zimg = cv2.equalizeHist(zimg)
        img[:,:,i] = zimg

    img = Image.fromarray(img).resize(size)
    seg = Image.fromarray(seg).resize(size)
    img = img.rotate(rotation)
    seg = seg.rotate(rotation)

    img = np.array(img)
    seg = np.array(seg)

    if flip:
        img = img[:,::-1,:]
        seg = seg[:,::-1,:]

    #seg = np.round(seg/255.0)

    return img/255, seg/255

def CreateKMeans(num_clusters=15):
    files = os.listdir("data/data_og/train")[0:10]
    colors = []
    for file in files:
        img, seg = LoadImage(name=file, path="data/data_og/train")
        colors.append(seg.reshape(seg.shape[0]*seg.shape[1], 3))
    colors = np.array(colors)
    colors = colors.reshape((colors.shape[0]*colors.shape[1],3))
    km = KMeans(num_clusters)
    km.fit(colors)
    return km