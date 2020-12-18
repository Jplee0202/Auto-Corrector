import numpy as np
import cv2, os, glob
from BOW import calc_sift

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

"""
Now Training!!!
"""

path_name = "dataset/train_imgaes"
set_train = {}
if os.path.isdir(path_name):
    next_path = os.listdir(path_name)
    #print(next_path)
    for files in next_path:
        if str(files)==".DS_Store":
            continue
        file = path_name + "/" + str(files)
        n_file = glob.glob(file + "/*.jpg")
        set_train[files] = len(n_file)

del set_train["features"]
del set_train["vocabulary"]

data = np.float32([]).reshape(0, k)
response = np.float32([])

dict_id = 0
for folder, number in set_train.items():
    dire = path_name + "/" + folder + "/"
    labels, centers = np.load(path_name + "/vocabulary/" + "vocabulary" + ".npy",allow_pickle=True)
    print("Training ", str(dict_id + 1))
    counter=number
    for i in range(number):
        file_name = dire + folder + "_" + str(i + 1) + ".jpg"
        print(file_name)
        img = cv2.imread(file_name)
        features = calc_sift(img)
        if features is None:
            counter-=1
            continue
        vec = calc_feature(features, centers)
        data = np.append(data, vec, axis = 0)
    res = np.repeat(np.int32([dict_id]), counter)
    response = np.append(response, res).astype(np.int32)
    dict_id = dict_id + 1


print("SVM Classifier")
data = np.float32(data)
response = response.reshape(-1, 1)
svm = cv2.ml.SVM_create()
svm.trainAuto(data,cv2.ml.ROW_SAMPLE,response)
svm.save("svm_60.clf")
print("Done")