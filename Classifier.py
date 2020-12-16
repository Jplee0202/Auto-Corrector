import numpy as np
import cv2, os, glob, gc
from BOW import calc_sift


k=100
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


"""
Now Classifying!!!
"""

path_name = "dataset/test_imgaes/"
set_train = {}
set_test = {}
if os.path.isdir(path_name):
    next_path = os.listdir(path_name)
    for files in next_path:
        if "features" in files:
            break
        file = path_name + "/" + str(files)
        n_file = glob.glob(file + "/*.jpg")
        set_test[files] = len(n_file)

del set_test[".DS_Store"]

svm = cv2.ml.SVM_load("svm_60.clf")
dire ="dataset/train_imgaes/vocabulary/"

tot = 0
correct = 0
dict_id = 0
print(set_test)


for folder, number in set_test.items():

    crt = 0
    labels, centers = np.load(dire + folder + ".npy",allow_pickle=True)
    print("Classify Test ", folder)
    for i in range(number):
        file_name = path_name + folder +"/" + folder + "_" + str(i + 1) + ".jpg"
        flag = True
        img = cv2.imread(file_name)
        features = calc_sift(img)
        if features is None:
            continue
        vec = calc_feature(features, centers)
        del features
        gc.collect()
        case = np.float32(vec)
        dd = svm.predict(case)
        if np.array(dict_id) in dd[1]:
            crt = crt + 1
    print("right prediction:",crt,", total prediction:",number,", accuracy: ",crt/number)

    dict_id = dict_id + 1
