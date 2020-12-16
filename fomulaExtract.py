import cv2
import identify

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
            id = identify(img[:, w1:w2])
            if possibility <= id[1]:
                symbol = id[0]
                possibility = id[1]
                leftSymbol = w1
                rightSymbol = w2
    possibility = 0
    for w1 in range(rightSymbol, width):
        for w2 in range(w1 + 1, width):
            id = identify(img[:, w1:w2])
            if possibility <= id[1] and id[0] == '=':
                possibility = id[1]
                leftEqual = w1
                rightEqual = w2

    firstNum = identify(img[:, 0:leftSymbol])[0]
    secondNum = identify(img[:, rightSymbol:leftEqual])[0]
    fomula = firstNum + symbol + secondNum + equal


    for h in range(height):
        for w in range(rightEqual, width):
            if img[h, w, 0] is not 255:
                ifAnswer = True

    return((fomula, ifAnswer))

img = cv2.imread('im0.png')
extract(img)