"""
Date:2021-6-30
Version:0.0.1
Author:Anita
Function:裁剪并进行人脸对齐
"""
import cv2 as cv
import math
from PIL import Image
import numpy as np

# function:点的旋转函数
def rotate(origin, point, angle, row):
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)
# function：人脸对齐旋转:将照片img旋转seta
# input：左右眼中心坐标 类型：tuple
# input:待旋转图片
# output:旋转后的图片及旋转后的左右眼中心坐标
def align_face(lefteye,righteye,img):
    # 1.旋转图片
    # 计算左右眼中心坐标连线与水平方向夹角
    dy = lefteye[1] - righteye[1]
    dx = abs(lefteye[0] - righteye[0])
    angle = math.atan2(dy, dx)*180 / math.pi
    # 计算两只眼睛中心距离
    eye_center = ((lefteye[0]+righteye[0])//2, (lefteye[1]+righteye[1])//2)
    rotate_matrix = cv.getRotationMatrix2D(eye_center,angle,scale=1)
    rotate_img = cv.warpAffine(img,rotate_matrix,(img.shape[1],img.shape[0]))
    # 2.左右眼中心坐标的旋转
    left_r = rotate(lefteye, lefteye, angle, img.shape[0])
    right_r = rotate(righteye, righteye, angle, img.shape[0])
    return left_r, right_r, rotate_img

# 一.识别眼睛
# 如果检测到两个眼睛，直接获取坐标进行裁剪并对齐，如果无法识别眼睛
# 则识别人脸，按人脸大小进行裁剪，不进行对齐操作
# 若什么都没识别到，返回原图
# 如果一张图片有多个人脸，以最清楚识别的为标准
def getEyes(imageName):
    # 1.读取相应图片
    img = cv.imread(imageName)
    flag = 0
    # 2.载入分类器
    classifier_eye = cv.CascadeClassifier('haarcascade_eye.xml')
    classifier_face = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
    # 3.检测人脸
    classifier_face.load('haarcascade_frontalface_alt.xml')
    classifier_eye.load('haarcascade_eye.xml')
    faceRects_face = classifier_face.detectMultiScale(img, 1.3, 5,flags=cv.CASCADE_SCALE_IMAGE,minSize=(30,30))
    # 检测到人脸
    if len(faceRects_face)>0:
        face_x, face_y, face_w, face_h = faceRects_face[0]
        # 获得人脸高度一半的位置，为精确识别眼睛的位置
        halfFaceH = int(float(face_h/1.5))
        # 转换为int方便图像截取
        intFaceX = int(face_x)
        intFaceY = int(face_y)
        intFaceW = int(face_w)
        intFaceH = int(face_h)
        # 截取人脸
        img_face = img[intFaceY:intFaceY+intFaceH, intFaceX:intFaceX+intFaceW]
        # 截取一半人脸用作眼睛识别
        img_face_half = img[intFaceY:intFaceY+halfFaceH,intFaceX:intFaceX+intFaceW]
        flag = 1
    if flag == 0:
        return img
    # 4.检测眼睛
    faceRects_eyes = classifier_eye.detectMultiScale(img_face_half, 1.3, 3,flags=cv.CASCADE_SCALE_IMAGE)
    if len(faceRects_eyes) == 2:
        # 列表存放两只眼睛坐标
        # 0-左眼，1-右眼
        # 列表结构：[x1,y1,x2,y2]
        eyes_tag = []
        for eye in faceRects_eyes:
            x1, y1, w1, h1 = eye
            # 修正坐标
            eye_x = int(x1) + intFaceX
            eye_y = int(y1) + intFaceY
            eye_w = w1
            eye_h = h1
            # 存入数组
            temp_list = [eye_x, eye_y, eye_x+eye_w, eye_y+eye_h]
            eyes_tag.append(temp_list)
        # 根据眼睛坐标进行对齐操作
        left_center = (0.5*(eyes_tag[0][0]+eyes_tag[0][2]),0.5*(eyes_tag[0][1]+eyes_tag[0][3]))
        right_center = (0.5 * (eyes_tag[1][0] + eyes_tag[1][2]), 0.5 * (eyes_tag[1][1] + eyes_tag[1][3]))
        left_r,right_r,img_r = align_face(left_center,right_center,img)
        # 对img_r进行裁剪
        # 两眼间距
        d = abs(right_r[0] - left_r[0])
        center = ((left_r[0] + right_r[0])/2,(left_r[1] + right_r[1])/2)
        img_eye = img_r[int(center[1]-0.65*d):int(center[1]+1.6*d),int(center[0]-0.6*d):int(center[0]+0.6*d)]
        return img_eye
    if flag == 1:
        return img_face
    else:
        return img
# 二.几何归一化
# 对于改进的LBP‑TOP特征，M取256，N取128；对于Gabor特征，为了降低维度，M取112，N取96
def normalize(img, M, N):
    image = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    img_nor = image.resize((N, M), Image.BILINEAR)
    result = cv.cvtColor(np.asarray(img_nor), cv.COLOR_RGB2BGR)
    return result
