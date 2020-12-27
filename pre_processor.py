import numpy as np
import cv2, os, glob, gc
from skimage.morphology import skeletonize
from PIL import Image




def pre_process(img):
    #decrease resolving power
    original_img = cv2.resize(img,(np.int(img.shape[1]),np.int(img.shape[0])), interpolation=cv2.INTER_AREA)
    #print(original_img.shape)
    blur = cv2.GaussianBlur(original_img, (5, 5), 0)
    #turn into gray graph
    G = blur[:, :, 1]
    R = blur[:, :, 2]
    img_gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)


    #reduce noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

    kernel2 = np.ones((3,3), np.uint8)
    opening = cv2.dilate(opening, kernel2, iterations=1)

    blur = cv2.GaussianBlur(opening, (13, 13), 0)
    ret, binary = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    locations=[]
    for i in contours:
        location = cv2.boundingRect(i)
        locations.append(location)

    processed=extract_img(locations,binary,contours)


    processed_rgb=np.zeros((45,45,3)).astype("uint8")
    processed_rgb[:,:,0]=processed
    processed_rgb[:,:,1]=processed
    processed_rgb[:,:,2]=processed
    # cv2.imshow("ok",processed_rgb)
    # cv2.waitKey(0)



    return processed_rgb


def extract_img(locations,img,contour=None):

    # x, y, w, h = location
    x_list=[] #x coordinate of left top point
    y_list=[] #y coordinate of left top point
    x_rb_list=[] #x coordinate of right bottom point
    y_rb_list=[] #y coordinate of right bottom point

    for x0,y0,w0,h0 in locations:
        x_list.append(x0)
        y_list.append(y0)
        x_rb_list.append(x0+w0)
        y_rb_list.append(y0+h0)
    x_min=min(x_list)
    x_max=max(x_rb_list)
    y_min=min(y_list)
    y_max=max(y_rb_list)
    W=x_max-x_min
    H=y_max-y_min

    # only get things in contour
    if contour is None:
        extracted_img = img[y_min:y_max, x_min:x_max]
    else:
        mask = np.zeros(img.shape, np.uint8)
        for contour_i in contour:
            cv2.drawContours(mask, [contour_i], -1, 255, cv2.FILLED)
        # cv2.imshow("mask",mask)
        # cv2.waitKey(0)
        img_after_masked = cv2.bitwise_and(mask, img)
        extracted_img = img_after_masked[y_min:y_max, x_min:x_max]
    #extract img and turn it into binary graph
    black = np.zeros((45, 45), np.uint8)
    if (W > H):
        res = cv2.resize(extracted_img, (45, (int)(H * 45 / W)), interpolation=cv2.INTER_AREA)
        d = int(abs(res.shape[0] - res.shape[1]) / 2)
        black[d:res.shape[0] + d, 0:res.shape[1]] = res
    else:
        res = cv2.resize(extracted_img, ((int)(W * 45 / H), 45), interpolation=cv2.INTER_AREA)
        d = int(abs(res.shape[0] - res.shape[1]) / 2)
        black[0:res.shape[0], d:res.shape[1] + d] = res
    extracted_img = skeletonize(black)

    extracted_img = np.logical_not(extracted_img)
    extracted_img=extracted_img.astype(np.uint8)*255

    #cv2.imwrite("ok.jpg",extracted_img)
    return extracted_img








