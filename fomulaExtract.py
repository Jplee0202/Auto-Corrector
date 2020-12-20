import cv2
from identify import recognize
import numpy as np
LARGEST_NUMBER_OF_SYMBOLS = 500
IMG_SIZE = 100
def extract(img):
    """
    This function is used to extract a fomula from the image
    :param img: single cutted image
    :return: a tuple of (fomula extracted, whether there is an answer)
    """
    fomula = ''
    equal = '='
    width = len(img[0])
    height = len(img)
    symbol = ''
    possibility = 0
    ifAnswer = False
    for w1 in range(width):
        for w2 in range(w1 + 1, width):
            id = recognize(img[:, w1:w2])
            if possibility <= id[1]:
                symbol = id[0]
                possibility = id[1]
                leftSymbol = w1
                rightSymbol = w2
    possibility = 0
    for w1 in range(rightSymbol, width):
        for w2 in range(w1 + 1, width):
            id = recognize(img[:, w1:w2])
            if possibility <= id[1] and id[0] == '=':
                possibility = id[1]
                leftEqual = w1
                rightEqual = w2

    firstNum = recognize(img[:, 0:leftSymbol])[0]
    secondNum = recognize(img[:, rightSymbol:leftEqual])[0]
    fomula = firstNum + symbol + secondNum + equal


    for h in range(height):
        for w in range(rightEqual, width):
            if img[h, w, 0] != 255:
                ifAnswer = True

    return((fomula, ifAnswer))

def fomulaExtract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    print(len(contours))
    dict = {}
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        dict[(x, y, w, h)] = x
    dict = sorted(dict)
    print(dict)
    firstNumPos = dict[1]
    symbolPos = dict[2]
    secondNumPos = dict[3]
    firstNum = img[firstNumPos[1]:firstNumPos[1] + firstNumPos[3], firstNumPos[0]:firstNumPos[0] + firstNumPos[2]]
    symbol = img[symbolPos[1]:symbolPos[1] + symbolPos[3], symbolPos[0]:symbolPos[0] + symbolPos[2]]
    secondNum = img[secondNumPos[1]:secondNumPos[1] + secondNumPos[3],
                secondNumPos[0]:secondNumPos[0] + secondNumPos[2]]
    print(recognize(firstNum))
    recognize(symbol)
    recognize(secondNum)
img1 = cv2.imread('im0.png')
fomulaExtract(img1)