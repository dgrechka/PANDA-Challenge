import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import itertools
import threading
import random
from libExtractTile import getNotEmptyTiles
from skimage import io


cytoImagePath = sys.argv[1]
labelsPath = sys.argv[2]
outputPath =sys.argv[3]

random.seed(8247236824)

labelsDf = pd.read_csv(labelsPath, engine='python')

idents = labelsDf.iloc[:,0]
labels = labelsDf.iloc[:,2]

N = len(idents)

tileSize = 1024
batchSize = 4
shuffleBufferSize = 32
prefetchSize = 16
trainSequenceLength = 64
seed = 35372932

tileIndexCache = dict()
tileIndexCacheLock = threading.Semaphore(1)

def getDataSet(idents,labels):
    def samplesGenerator():
        N = len(idents)
        for i in range(0,N): 
            ident = idents[i]
            label = labels[i]
            toOpen = os.path.join(cytoImagePath,"{0}.tiff".format(ident))
            print("openning {0}".format(toOpen))
            image = io.imread(toOpen)
            
            tileIndexCacheLock.acquire()
            if ident in tileIndexCache:
                precomputedTileIndeces = tileIndexCache[ident]
            else:
                precomputedTileIndeces = None
            tileIndexCacheLock.release()

            indices,tiles = getNotEmptyTiles(image,tileSize, precomputedTileIndeces)

            tileIndexCacheLock.acquire()
            tileIndexCache[ident] = indices
            tileIndexCacheLock.release()

            npTiles = np.stack(tiles,axis=0)

            yield (npTiles, label) 

    return tf.data.Dataset.from_generator( 
        samplesGenerator, 
        (tf.uint8, tf.uint8), 
        (tf.TensorShape([None,tileSize,tileSize,3]), tf.TensorShape([]))) 


tr_ds = getDataSet(idents,labels)

def coerceSeqSize(imagePack, label):
  imagePackShape = tf.shape(imagePack)
  T = imagePackShape[0]

  # if T is less than trainSequenceLength we need to duplicate the layers
  seqRepCount = tf.cast(tf.math.ceil(trainSequenceLength / T),tf.int32)
  notTooShort = tf.tile(imagePack, [seqRepCount, 1, 1, 1])
  # if T is greater than trainSequenceLength we need to truncate it
  notTooLong = notTooShort[0:trainSequenceLength,:,:,:]
  return notTooLong, label


# TODO: shuffle slices
tr_ds = tr_ds \
    .map(coerceSeqSize, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .repeat() \
    .shuffle(shuffleBufferSize,seed=seed) \
    .batch(batchSize, drop_remainder=True) \
    .prefetch(prefetchSize)

# At this point the data ingest first pipeline is ready
# TODO: add model and optimizer

print(list(tr_ds.take(3).as_numpy_iterator()))