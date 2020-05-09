import tensorflow as tf
import numpy as np
from skimage import io
import os
import threading
from libExtractTile import getNotEmptyTiles


feature_description = {
      'image': tf.io.FixedLenFeature([], tf.string),    
      'size': tf.io.FixedLenFeature([4], tf.int64),
  }

def getTfRecordDataset(tfRecordPaths):
    # random.shuffle(tfRecordPaths) # good chance to shuffle whole DS
    print("Constructing TfRecords dataset with {0} files".format(len(tfRecordPaths)))
    return tf.data.TFRecordDataset(tfRecordPaths,compression_type="GZIP")


def extractTilePackFromTfRecord(tfrecord_data):
    example = tf.io.parse_single_example(tfrecord_data, feature_description)
    #print(example)
    size = example['size']
    S,H,W,C = size[0],size[1],size[2],size[3]
    imageInt8 = tf.io.decode_raw(example['image'], tf.uint8)
    imageReshaped = tf.reshape(imageInt8,(S,W,H,C))

    return imageReshaped
  
def getTiffTrainDataSet(cytoImagePath, idents, labels):
    def samplesGenerator():
        N = len(idents)
        for i in range(0,N): 
            ident = idents[i]
            label = labels[i]
            toOpen = os.path.join(cytoImagePath,"{0}.tiff".format(ident))
            
            yield (toOpen, ident, label) 

    return tf.data.Dataset.from_generator( 
        samplesGenerator, 
        (tf.string, tf.string, tf.uint8), 
        (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))) 

tileIndexCache = dict()
tileIndexCacheLock = threading.Semaphore(1)


def loadTiffImage(imagePath,ident,label,tileSize=1024):
    def loadAsTilePack(path,ident):
        path = path.numpy().decode("utf-8")
        ident = ident.numpy().decode("utf-8")
        #print("openning {0}".format(path))
        image = io.imread(path)

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

        T = len(indices)
        if T == 0: # guard againt empty tile set (all of the tiles are white!)
            print("sample {0} does not contain suitable tiles! requires investigation!".format(ident))
            return np.empty((1,1024,1024,3)), 1, False
        else:
            npTiles = np.stack(tiles,axis=0)
            #print(npTiles)
            return npTiles, T, True
    
    imagePack,tileCount,valid = tf.py_function(func=loadAsTilePack, inp=[imagePath,ident], Tout=(tf.uint8, tf.int32, tf.bool)) 
    imagePack = tf.reshape(imagePack,[tileCount,tileSize,tileSize,3])
    resLabel = tf.cond(valid, lambda: label, lambda: tf.constant(255,dtype=tf.uint8))
    return imagePack, resLabel
  
def isValidPack(imagePack, label):
    return label <= 5

# usage example
# trDs = getTiffTrainDataSet(trIdents,trLabels) \
#    .map(loadTiffImage , num_parallel_calls=tf.data.experimental.AUTOTUNE) \
#    .filter(isValidPack) \