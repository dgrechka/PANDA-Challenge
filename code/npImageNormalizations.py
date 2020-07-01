import numpy as np
import math

def getImageMean_withoutPureBlack(image):
    black = np.array([0,0,0])
    cleared = np.where(image==black,np.NaN,image)
    result = np.nanmean(cleared)
    if np.isnan(result):
        result = 0.0 # pure black
    return result

def getImageMean_withoutBlack(image,blackThreshold):
    #print("image shape is {0}".format(image.shape))
    imageBrightness = np.nanmax(image, axis=-1)

    blackMap = imageBrightness < blackThreshold
    blackMap = np.expand_dims(blackMap,axis=-1)
    blackMap = np.tile(blackMap, (1,1,3))
    if not(np.any(blackMap)):
        # nonNan exists
        cleared = np.where(blackMap,np.NaN,image)
        return np.nanmean(cleared)
    else:
        return np.NaN # pure black
    
def getImageContrast_withoutPureBlack(image, regTerm=0.0, precomputedMu = None):
    """(0,0,0) pixels are excluded from contrast evaluation"""
    if precomputedMu is None:
        mu = getImageMean_withoutPureBlack(image)
    else:
        mu = precomputedMu
    diff = image - mu
    return getImageContrast(image,regTerm=regTerm, precomputedDiff=diff)

def getImageContrast(image, regTerm=0.0, precomputedDiff = None):
    """Entire image contrast as defined in Goodfellow et al. 2016 "Deep learning" p.442"""
    if precomputedDiff is None:
        mu = getImageMean_withoutPureBlack(image)
        diff = image - mu
    else:
        diff = precomputedDiff
    
    squaredDiff = diff*diff
    meanSqDiff = np.mean(squaredDiff)
    return math.sqrt(regTerm + meanSqDiff)

def GCN(image,lambdaTerm = 0.0, eps=1e-8, precomputedContrast = None, precomputedMean = None):
    """Global Contrast Normalization as defined in Goodfellow et al. 2016 "Deep learning" p.442"""
    if precomputedMean is None:
        mu = np.mean(image)
    else:
        mu = precomputedMean
    
    diff = image - mu
    if precomputedContrast is None:
        initialContrast = getImageContrast(image, lambdaTerm, diff)
    else:
        initialContrast = precomputedContrast
    return diff / max(eps, initialContrast)

def GCNtoRGB_uint8(gcnImage, cutoffSigmasRange = 3.0):
    """gcnImage is considered to have 0 mean and stddev == 1.0"""
    rescaled = (gcnImage + cutoffSigmasRange)/(2.0 * cutoffSigmasRange) * 255.0
    return np.round(np.minimum(np.maximum(rescaled,0.0),255.0)).astype(np.uint8)
