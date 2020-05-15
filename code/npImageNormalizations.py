import numpy as np
import math

def getImageMean_withoutPureWhite(image):
    white = np.array([255,255,255])
    cleared = np.where(image==white,np.NaN,image)
    return np.nanmean(cleared)
    

def getImageContrast_withoutPureWhite(image, regTerm=0.0):
    """(255,255,255) pixels are exluded from contrast evaluation"""
    mu = getImageMean_withoutPureWhite(image)
    diff = image - mu
    return getImageContrast(image,regTerm=regTerm, precomputedDiff=diff)

def getImageContrast(image, regTerm=0.0, precomputedDiff = None):
    """Entire image contrast as defined in Goodfellow et al. 2016 "Deep learning" p.442"""
    if precomputedDiff is None:
        mu = np.mean(image)
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
