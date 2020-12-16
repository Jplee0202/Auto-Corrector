import cv2, os, glob
import numpy as np

traing_num=400
def calc_sift(img):
    """
    exract the feature of input image
    :param img: input image
    :return: feature
    """

    res = cv2.resize(img, None, fx=15, fy=15, interpolation=cv2.INTER_CUBIC)
    #print(res.shape)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(200)
    kp, des = sift.detectAndCompute(gray, None)

    return des


if __name__ == "__main__":

    """
    generate dict of training set's info
    "digit" : number of training images
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
    print(set_train)
# """
# Calculate and Save the SIFT feature sets
# """

    for folder, number in set_train.items():
        dir = path_name + "/" + folder + "/"
        sets = np.float32([]).reshape(0, 128)
        print("Now processing ", folder)
        for i in range(traing_num):
            file_name = dir + folder + "_" + str(i + 1) + ".jpg"
            # print(file_name)
            img = cv2.imread(file_name)
            des = calc_sift(img)
            if des is None:
                continue
            print(file_name)
            sets = np.append(sets, des, axis=0)
        ff = sets.shape[0]
        print(str(ff), " features in ", str(number), " images")
        file_name = path_name + "/features/" + folder + ".npy"
        np.save(file_name, sets)
    print(set_train)

    k =100
    for folder,number in set_train.items():
        file_name = path_name + "/features/" + folder + ".npy"
        features = np.load(file_name)
        print("Now learning vocabulary ", folder, " ", end="")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(features, k, None, criteria, 20, flags)
        file_name = path_name + "/vocabulary/" + folder + ".npy"
        np.save(file_name, (labels, centers))
        print(" Done")