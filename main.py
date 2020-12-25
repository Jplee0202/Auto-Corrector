from cut import imgCutter
from fomulaExtract import fomulaE
import cv2 as cv
import os

originImg = imgCutter(cv.imread('originImg.png', cv.COLOR_BGR2GRAY), 128)
files = os.listdir('cutted_img')
for f in files:
    if 'png' in f:
        img = cv.imread('cutted_img/' + f)
        fomula = fomulaE(img)
        if fomula['symbol'] == '+':
            ret = int(fomula['firstNum']) + int(fomula['secondNum'])
        elif fomula['symbol'] == '-':
            ret = int(fomula['firstNum']) - int(fomula['secondNum'])
        elif fomula['symbol'] == 'x':
            ret = int(fomula['firstNum']) * int(fomula['secondNum'])
        elif fomula['symbol'] == 'd':
            ret = int(fomula['firstNum']) / int(fomula['secondNum'])
        print('Answer for ' + fomula['firstNum'] + fomula['symbol'] + fomula['secondNum'] + 'is: ' + ret)
