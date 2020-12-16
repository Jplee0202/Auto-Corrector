import cv2 as cv
num = 0
def imgCutter(originImg, div):
    '''

    :param originImg: Input the original image with gray-scale
    :param div: Gradation of gray level
    :return: The cutted images in the root dictionary

    '''

    global num
    width = len(originImg[0])
    height = len(originImg)
    upD = False
    downD = False
    rightD = False
    leftD = False
    upLine = 0
    downLine = 0
    leftLine = 0
    rightLine = 0
    for h in range(height):
        if upD:
            break
        for w in range(width):
            if originImg[h, w, 0] < div:
                if not upD:
                    upLine = h
                    upD = True
    if not upD:
        return
    for h in range(upLine, height):
        if downD:
            break
        blank = True
        for w in range(width):
            if originImg[h, w, 0] < div:
                blank = False
                break
        if blank:
            downD = True
            downLine = h

    for w in range(width):
        if leftD:
            break
        for h in range(upLine, downLine):
            if originImg[h, w, 0] < div:
                if not leftD:
                    leftLine = w
                    leftD = True

    for w in range(width - 1, leftLine - 1, -1):
        if rightD:
            break
        blank = True
        for h in range(upLine, downLine):
            if originImg[h, w, 0] < div:
                blank = False
                break
        if not blank:
            rightD = True
            rightLine = w
    string = str(num)
    cv.imwrite('im'+string+'.png', originImg[upLine:downLine, leftLine:rightLine])
    num += 1
    imgCutter(originImg[downLine:, :], 20)

originImg = imgCutter(cv.imread('originImg.png', cv.COLOR_BGR2GRAY), 128)