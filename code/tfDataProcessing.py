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

def coerceSeqSize(imagePack, trainSequenceLength):
  imagePackShape = tf.shape(imagePack)
  T = imagePackShape[0]

  # if T is less than trainSequenceLength we need to duplicate the layers
  seqRepCount = tf.cast(tf.math.ceil(trainSequenceLength / T), tf.int32)
  notTooShort = \
    tf.cond(seqRepCount > 1, \
        lambda : tf.tile(tf.random.shuffle(imagePack), [seqRepCount, 1, 1, 1]), \
        lambda : imagePack)
  
  # if T is greater than trainSequenceLength we need to truncate it
  notTooLong = tf.random.shuffle(notTooShort)[0:trainSequenceLength,:,:,:]
  shapeSet = tf.reshape(notTooLong,
    [
        trainSequenceLength,
        imagePackShape[1],
        imagePackShape[2],
        imagePackShape[3]
    ])
  return shapeSet

def downscale(imagePack, nnTileSize):
    resized = tf.image.resize(
        imagePack,
        [nnTileSize,nnTileSize], antialias=True,
        method=tf.image.ResizeMethod.AREA
        )
    return resized

def augment(imagePack):
    def augmentSingle(image):
        augSwitches = tf.cast(tf.math.round(tf.random.uniform([3],minval=0.0, maxval=1.0)),dtype=tf.bool)
        image = tf.cond(augSwitches[0], lambda: tf.image.rot90(image), lambda: image)
        image = tf.cond(augSwitches[1], lambda: tf.image.flip_left_right(image), lambda: image)
        image = tf.cond(augSwitches[2], lambda: tf.image.flip_up_down(image), lambda:image)
        return image
    return tf.map_fn(augmentSingle, imagePack, back_prop=False)
    

# usage example
# trDs = getTiffTrainDataSet(trIdents,trLabels) \
#    .map(loadTiffImage , num_parallel_calls=tf.data.experimental.AUTOTUNE) \
#    .filter(isValidPack) \