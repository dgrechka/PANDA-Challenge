import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import itertools
import threading
import multiprocessing
import random
import math
import matplotlib.pyplot as plt # for debugging previews only
from libExtractTile import getNotEmptyTiles
from tfQuadraticWeightedKappa import QuadraticWeightedKappa
import tfDataProcessing as tfdp
from modelA import constructModel
from skimage import io


cytoImagePath = sys.argv[1]
outPath = sys.argv[2]
if len(sys.argv)>3 :
    truncateCount = int(sys.argv[3])
else:
    truncateCount = 10000000

print("TFRecords path is {0}".format(cytoImagePath))

print("Generating previews for not more than {0} elements of the dataset".format(truncateCount))

trFilenames = os.listdir(cytoImagePath)
trFilenames.sort()
trFilenames = [fname for fname in trFilenames if fname.endswith(".tfrecords")]
#trFilenames = [fname for fname in trFilenames if fname[33:-10]=="0"] # only tiles without rotations
fullTrFilenames = [os.path.join(cytoImagePath,fname) for fname in trFilenames]
fullOutFilenames = ["{0}.jpeg".format(os.path.join(outPath,fname[0:-10])) for fname in trFilenames]

initiallyFound = len(fullTrFilenames)

print("Found {0} tfrecords files in source dir".format(initiallyFound))

toProcessIn = []
toProcessOut = []
for i in range(0,len(fullTrFilenames)):
    inFullPath = fullTrFilenames[i]
    outFullPath = fullOutFilenames[i]
    if not(os.path.exists(outFullPath)):
        toProcessIn.append(inFullPath)
        toProcessOut.append(outFullPath)
fullTrFilenames = toProcessIn
fullOutFilenames = toProcessOut
toGenerateCount = len(fullTrFilenames)
print("{0} previews already exist. {1} to generate".format(initiallyFound - toGenerateCount, toGenerateCount))

truncateCount = min(truncateCount, len(trFilenames))

def trImageTransform(imagePack):
    return tfdp.bigImageFromTiles(tfdp.coerceSeqSize(imagePack,36), 6)
    #return tfdp.bigImageFromTiles(tfdp.augment(tfdp.coerceSeqSize(imagePack,36)), 6)

trImagesDs = tfdp.getTfRecordDataset(fullTrFilenames) \
    .map(tfdp.extractTilePackFromTfRecord,deterministic=True).map(trImageTransform,deterministic=True).take(truncateCount)

i = 0
for sample in trImagesDs.as_numpy_iterator():
    outFile = fullOutFilenames[i]
    io.imsave(outFile,sample)
    print("{0} is ready ({1} out of {2})".format(outFile,i+1, truncateCount))
    i+=1
print("Done")









