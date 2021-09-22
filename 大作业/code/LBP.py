"""
Author:Anita
Version:0.0.1
Date:2021-7-5
Function:提取时空特征,LBP-TOP直方图特征
        Gabor特征作为纹理特征
"""
import operator
from functools import reduce
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io, data_dir,filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
from PIL import Image
import cv2 as cv
import os
from skimage import filters
# 将矩阵转换为向量

# 计算LBP值
# 相邻像素点个数P，半径R
# 输入待计算矩阵及形状参数
def calcuLBP(LBPmat,n_points=8,radius=3):
    lbp = local_binary_pattern(LBPmat, n_points, radius)
    return lbp
# 将process中的图像覆盖
# 图像尺寸：256*128*10
# 1.提取XT,YT平面的LBP直方图向量
# 2.提取峰值图像XY平面的LBP直方图向量
# 选取中间帧为峰值图像
def AllLBP(dirName):
    imgMatrix = []
    # 1.生成256*128*10的矩阵
    for file in os.listdir(dirName):
        img = cv.imread(dirName + '\\' + file)
        imgMatrix.append(img[:,:,0])
    imgMatrix = np.array(imgMatrix)
    # 2.提取XT矩阵(10*256)->(20*128)
    XT_matrix = np.zeros((10, 256))
    for i in range(128):
        XT_matrix += imgMatrix[:, :, i]
    XT_img = Image.fromarray(XT_matrix/128)
    XT_LBP = calcuLBP(XT_img)
    XT_LBP = np.mat(XT_LBP).reshape((20,128))
    # 3.提取YT矩阵(10*128)->(10*128)
    YT_matrix = np.zeros((10, 128))
    for i in range(256):
        YT_matrix += imgMatrix[:, i, :]
    YT_img = Image.fromarray(YT_matrix/256)
    YT_LBP = calcuLBP(YT_img)
    YT_LBP = np.mat(YT_LBP).reshape((10, 128))
    # 4.提取XY矩阵(256*128)->(256*128)
    XY_matrix = imgMatrix[5, :, :]
    XY_img = Image.fromarray(XY_matrix)
    XY_LBP = calcuLBP(XY_img)
    # 转换为矩阵 143*2*128
    LBP_feature = []
    i = 0
    while(i<20):
        LBP_feature.append(XT_LBP[i:i+2,:])
        i += 2
    i = 0
    while (i < 10):
        LBP_feature.append(YT_LBP[i:i + 2,:])
        i += 2
    i = 0
    while (i < 256):
        LBP_feature.append(XY_LBP[i:i + 2,:])
        i += 2
    # 返回峰值帧图样
    return XY_img, LBP_feature


def LBPFeature(dirName):
    XY_img, LBP_feature = AllLBP(dirName)
    LBP_feature = np.array(LBP_feature)
    return XY_img, LBP_feature

def feature(hostName):
    LBP = []
    # Gabor = []
    for dir in os.listdir(hostName):
        # os.mkdir('processed\\'+ dir)
        dirName = hostName + '\\' + dir
        XY_img, LBP_feature = LBPFeature(dirName)
        # 将特征转换为一维向量
        LBP_feature = LBP_feature.reshape(36608)
        # 获取Gabor特征（纹理特征）
        # Gabor_feature = Gaborfeature(XY_img)
        # 256*128
        # 64*32
        # Gabor_feature = Gabor_feature.reshape(32768)
        LBP.append(LBP_feature)
        # Gabor.append(Gabor_feature)
    # Gabor = np.array(Gabor)
    LBP = np.array(LBP)
    return LBP

def Gaborfeature(XY_img):
    # gabor变换
    real, imag = filters.gabor(XY_img, frequency=0.6,theta=45,n_stds=5)
    # 取模
    img_mod=np.sqrt(real.astype(float)**2+imag.astype(float)**2)
    # 图像缩放（下采样）
    # 64*32
    # 256*128
    # newimg = cv.resize(img_mod,(0,0),fx=1/4,fy=1/4,interpolation=cv.INTER_AREA)
    newfea = img_mod
    return newfea
