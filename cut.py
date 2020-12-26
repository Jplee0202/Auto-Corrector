import cv2 as cv
num = 0
def imgCutter(originImg, div):
    '''
    This function is used to cut a single image with a fomula from an origin image
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
    cv.imwrite('cutted_img/img'+string+'.png', originImg[upLine-5:downLine+5, leftLine-5:rightLine+5])
    num += 1
    imgCutter(originImg[downLine:, :], 20)

def cutWhite(originImg, div):
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

    for w in range(leftLine, width):
        if rightD:
            break
        blank = False
        for h in range(upLine, downLine):
            if originImg[h, w, 0] < div:
                blank = True
        if not blank:
            rightD = True
            rightLine = w
    return originImg[upLine - 2:downLine + 2, leftLine - 2:rightLine + 2]
