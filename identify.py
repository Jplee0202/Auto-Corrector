import numpy as np
import cv2, os, glob, gc

def calc_sift(img):
    """
    exract the feature of input image
    :param img: input image
    :return: feature
    """
    new_img=cv2.GaussianBlur(img,(5,5),0)
    cv2.resize(new_img, (28, 28))

    res = cv2.resize(img, None, fx=20, fy=20, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("res",res)
    #print(res.shape)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(200)
    kp, des = sift.detectAndCompute(gray, None)

    return des

k=600
def calc_feature(features, centers):
    vec_feature = np.zeros((1, k))
    for i in range(0, features.shape[0]):
        feature = features[i]
        diff = np.tile(feature, (k, 1)) - centers
        sq_sum = (diff**2).sum(axis = 1)
        norm_2 = sq_sum**0.5
        sorted_vec = norm_2.argsort()
        id = sorted_vec[0]
        vec_feature[0][id] = vec_feature[0][id] + 1
    return vec_feature


operator_list=["9","0","7","+","6","1","8","-","d","=","4","x","3","2","5"]
def recognize(img):

    dire = "dataset/train_imgaes/vocabulary/"

    labels, centers = np.load(dire + 'vocabulary' + ".npy", allow_pickle=True)
    features = calc_sift(img)
    if features is None:
        print("ERROR: image size is too small")
        return
    vec = calc_feature(features, centers)
    del features
    gc.collect()
    case = np.float32(vec)
    a,b = svm.predict(case)
    return operator_list[int(b[0][0])]


svm = cv2.ml.SVM_load("svm_60.clf")



"""
cr=0
for i in range(1,20):
    file="dataset/test_imgaes/"+ "1" +"/"+ "1" + "_" + str(i) +".jpg"
    print(file)
    img=cv2.imread(file)
    re=recognize(img)
    if re=="1":
        cr+=1

    print(cr)
"""