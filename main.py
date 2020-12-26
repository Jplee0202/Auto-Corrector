from cut import imgCutter
from formulaExtract import formulaE
import cv2 as cv
import os

originImg = imgCutter(cv.imread('originalImg.png', cv.COLOR_BGR2GRAY), 128)
files = os.listdir('cutted_img')
files = sorted(files)
for f in files:
    if 'png' in f:
        img = cv.imread('cutted_img/' + f)
        formula = formulaE(img)
        tf = 'wrong!'
        if formula['symbol'] == '+':
            ret = int(formula['firstNum']) + int(formula['secondNum'])
        elif formula['symbol'] == '-':
            ret = int(formula['firstNum']) - int(formula['secondNum'])
        elif formula['symbol'] == 'x':
            formula['symbol'] = '*'
            ret = int(formula['firstNum']) * int(formula['secondNum'])
        elif formula['symbol'] == 'd':
            formula['symbol'] = '/'
            ret = int(formula['firstNum']) / int(formula['secondNum'])
        if formula['resultTF']:
            if int(ret) == int(formula['result']):
                tf = "Right!"
                print('Answer for', formula['firstNum'], formula['symbol'], formula['secondNum'], 'is: ', ret, "  ", tf)
            else:
                tf = "Wrong!"
                print('Answer for', formula['firstNum'], formula['symbol'], formula['secondNum'], 'is: ', ret, "  ", tf,"Your answer is:",formula['result'])

        else:
            tf = "You do not write an answer"
            print('Answer for', formula['firstNum'], formula['symbol'], formula['secondNum'], 'is: ', ret,"  ",tf)


for f in files:
    os.remove('cutted_img/' + f)
