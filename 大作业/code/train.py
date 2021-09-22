"""
Author:Anita
Date:2021-7-5
Version:0.0.1
Function:SVM算法实现
"""
from sklearn.svm import NuSVC
import numpy as np
import LBP
# 加载数据集
def loadDataSet():
    # 加载数据
    LBP_dataSet = LBP.feature('processed')
    # 加载标签
    fr = open('train\\label.txt',encoding='utf-8')
    label = [line.strip().split('\t') for line in fr.readlines()]
    label = [int(x[0]) for x in label]
    # label = np.array(label)
    return LBP_dataSet, label
# 加载测试数据
def testSet():
    LBP_feature = LBP.feature('test')
    return LBP_feature

# 使用核函数：rbf
def SVM(dataSet,label):
    # 10 * 36608
    # 10 * 1
    model = NuSVC(kernel='rbf')
    # model = NuSVC()
    model.fit(dataSet, label)
    result = model.predict(dataSet)
    return model

# 使用交叉验证法获得平均结果
# 80%训练集，20%测试集
# 使用核函数：rbf
def main():
    LBP_dataSet, label = loadDataSet()
    # dataSet = np.array(dataSet)
    # dataSet:10*143*2*128(10 * 36608)
    # label:10*1
    # print(SVM(Gabor_dataSet, label))
    # 经过测试，LBP_dataSet效果较好，使用纹理特征反而影响结果
    model = SVM(LBP_dataSet, label)
    # predict写入结果
    test = testSet()
    result = model.predict(test)
    return result


main()