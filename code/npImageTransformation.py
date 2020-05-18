import numpy as np
import cv2
import math

def TrimBlackPaddings(image):
    rowsAggregated = np.amax(image,axis=(0,2))
    rowIndices = np.where(rowsAggregated != 0)[0]
    firstRow,lastRow = rowIndices[0], rowIndices[-1]

    colsAggregated = np.amax(image, axis=(1,2))
    colIndices = np.where(colsAggregated != 0)[0]
    firstCol, lastCol = colIndices[0], colIndices[-1]


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
        print("target rotation size is {0},{1}".format(paddedW, paddedH))
        rotated = cv2.warpAffine(paddedImage, rot, (paddedW,paddedH))

        return TrimBlackPaddings(rotated)
    