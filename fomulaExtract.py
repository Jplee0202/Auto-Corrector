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

    return ((fomula, ifAnswer))


def fomulaExtract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    cv2.imwrite('te.png', img)
    leftSymbol = 0
    rightSymbol = 0
    num = 0
    for contour in contours[1:]:
        x, y, w, h = cv2.boundingRect(contour)
        image = img[:, x:x + w]
        reg = recognize(image)
        if reg == '+' or reg == '-' or reg == 'x' or reg == 'd':
            leftSymbol = x
            rightSymbol = x + w
            symbol = reg
            break
    for contour in contours[1:]:
        x, y, w, h = cv2.boundingRect(contour)
        image = img[:, x:x + w]
        cv2.imwrite('test'+str(num)+'.png',image)
        num += 1
        reg = recognize(image)
        print(reg)
        if reg == '=':
            leftEqual = x
            rightEqual = x + w

    firstNum = recognize(img[:, 0:leftSymbol])
    print(firstNum)
    secondNum = recognize(img[:, rightSymbol:leftEqual])
    print(secondNum)


img1 = cv2.imread('im3.png')
fomulaExtract(img1)

