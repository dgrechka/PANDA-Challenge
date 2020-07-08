import numpy as np
import cv2
import math

def GetNonBlackArea(image):
    rowsAggregated = np.amax(image,axis=(0,2))
    rowIndices = np.where(rowsAggregated != 0)[0]
    if len(rowIndices) == 0:
        print("WARN: entire black image in TrimBlackPaddings")
        return image # entire black image
    firstRow,lastRow = rowIndices[0], rowIndices[-1]

    colsAggregated = np.amax(image, axis=(1,2))
    colIndices = np.where(colsAggregated != 0)[0]
    if len(colIndices) == 0:
        print("WARN: entire black image in TrimBlackPaddings")
        return image # entire black image

    firstCol, lastCol = colIndices[0], colIndices[-1]
    return firstCol, firstRow, lastCol, lastRow

def TrimBlackPaddings(image, precomputedBounds = None):
    if precomputedBounds is None:
        firstCol, firstRow, lastCol, lastRow = GetNonBlackArea(image)
    else:
        firstCol, firstRow, lastCol, lastRow = precomputedBounds

    return image[firstCol:(lastCol+1), firstRow:(lastRow+1), :]


def RotateWithoutCrop(image, angleDeg):
    angleDeg = angleDeg % 360.0
    #print("cropping initial")
    image = TrimBlackPaddings(image)

    if abs(angleDeg) < 1e-6:
        return image
    else:
        h,w,_ = image.shape
        diag = math.sqrt(h*h + w*w)
        diagInt = int(diag)
        padH = diagInt - h
        padW = diagInt - w

        if diagInt > 32768:
            print("WARN: image size is more than 32768 in RotateWithoutCrop. Will not rotate and return the image as is‬")
            return image

        #print("padding")
        paddedImage = np.pad(image, (
            (padH // 2, padH // 2),
            (padW // 2, padW // 2),
            (0,0)))

        paddedH, paddedW, _ = paddedImage.shape

        #print("paddedH {0}; paddedW {1}; diag {2}".format(paddedH, paddedW, diagInt))

        center = int(diag * 0.5)
        #print("affine transofrming")
        rot = cv2.getRotationMatrix2D((center,center), angleDeg, 1)
        #print("angle {2:.2f}\t\ttarget rotation size is {0},{1}".format(paddedW, paddedH,angleDeg))
        rotated = cv2.warpAffine(paddedImage, rot, (paddedW,paddedH))

        return TrimBlackPaddings(rotated)
    