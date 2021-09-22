"""
Date:2021-6-29
Version:0.0.1
Author:Anita
Function:preprocessed
"""
import cv2 as cv
import numpy as np
import faceAlign
import os
# 训练集选取20张读入，处理后选择10张作为最终训练的内容
# 测试集同理

# 图像灰度归一化处理
def greyProcess(imageName):
    # 处理公式：I(gray)=IB×0.114+IG×0.587+IR×0.299
    # 读取彩色图像
    img = cv.imread(imageName)
    # 获取图像信息
    height, width, channel = img.shape
    # 创建同样高宽的画布
    new = np.zeros((height, width,3))
    # 遍历像素，通过加权平均得到灰度图
    for i in range(height):
        for j in range(width):
            new[i,j] = 0.299*img[i,j][0]+0.587*img[i,j][1]+0.114*img[i,j][2]

    # 显示图像
    # cv.namedWindow('Processed')
    # cv.imshow('Processed',new)
    # 写入路径
    # cv.imwrite('processed\\'+imageName, new)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # 返回灰度图
    return new

def preprocess(imageName):
    greyImage = greyProcess(imageName)
    cv.imwrite('processed\\' + imageName[6:], greyImage)
    tempImage = faceAlign.getEyes('processed\\' + imageName[6:])
    normalizeImage = faceAlign.normalize(tempImage, 256, 128)
    # 保存图像
    cv.imwrite('processed\\' + imageName[6:], normalizeImage)


def trainTraverse():
    for dir in os.listdir('train'):
        # os.mkdir('processed\\'+ dir)
        if dir != 'label.txt':
            for file in os.listdir('train\\' + dir):
                preprocess('train\\'+ dir +'\\' + file)
    # preprocess('train\\001\\00007.png')
    # preprocess('train\\004\\00007.png')
trainTraverse()
