from cut import imgCutter
from formulaExtract import formulaE
import cv2 as cv
import os

originImg = imgCutter(cv.imread('originImg.png', cv.COLOR_BGR2GRAY), 128)
files = os.listdir('cutted_img')
for f in files:
    if 'png' in f:
        img = cv.imread('cutted_img/' + f)
        formula = formulaE(img)
        if formula['symbol'] == '+':
            ret = int(formula['firstNum']) + int(formula['secondNum'])
        elif formula['symbol'] == '-':
            ret = int(formula['firstNum']) - int(formula['secondNum'])
        elif formula['symbol'] == 'x':
            ret = int(formula['firstNum']) * int(formula['secondNum'])
        elif formula['symbol'] == 'd':
            ret = int(formula['firstNum']) / int(formula['secondNum'])
        print('Answer for ' + formula['firstNum'] + formula['symbol'] + formula['secondNum'] + 'is: ' + ret)
