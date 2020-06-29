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
    #print("Constructing TfRecords dataset with {0} files".format(len(tfRecordPaths)))
    #print("Constructing TfRecords dataset with from {0} files".format(tfRecordPaths))
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

def isup_to_smoothed_labels(label):
    label = tf.cast(label, dtype=tf.int32)
    # label smothing that accounts for order
    def case0(): return tf.constant([2/3,   2/9,    1/9,    0,      0,      0],dtype=tf.float32)
    def case1(): return tf.constant([1/6,   2/3,    1/9,    1/18,   0,      0],dtype=tf.float32)
    def case2(): return tf.constant([1/18,  1/9,    2/3,    1/9,    1/18,   0],dtype=tf.float32)
    def case3(): return tf.constant([0,     1/18,   1/9,    2/3,    1/9,    1/18],dtype=tf.float32)
    def case4(): return tf.constant([0,     0,      1/18,   1/9,    2/3,    1/6],dtype=tf.float32)
    def case5(): return tf.constant([0,     0,      0,      1/9,    2/9,    2/3],dtype=tf.float32)
    return tf.switch_case(
        label,
        {
            0: case0,
            1: case1,
            2: case2,
            3: case3,
            4: case4,
            5: case5
        })

def coerceSeqSize(imagePack, trainSequenceLength):
  imagePackShape = tf.shape(imagePack)
  outputShape = [
        trainSequenceLength,
        imagePackShape[1],
        imagePackShape[2],
        imagePackShape[3]
    ]
  T = imagePackShape[0]

  availableIndices = tf.range(T)
  

  # if T is less than trainSequenceLength we need to duplicate the layers
  seqRepCount = tf.cast(tf.math.ceil(trainSequenceLength / T), tf.int32)
  notTooShortIndicies = \
    tf.cond(seqRepCount > 1, \
        lambda : tf.tile(tf.random.shuffle(availableIndices), [seqRepCount]), \
        lambda : availableIndices)
  
  # if T is greater than trainSequenceLength we need to truncate it
  notTooLongIndices = tf.random.shuffle(notTooShortIndicies)[0:trainSequenceLength]
  #notTooLong = tf.IndexedSlices(imagePack,notTooLongIndices, dense_shape = outputShape)
  notTooLong = tf.gather(imagePack, notTooLongIndices)
  shapeSet = tf.reshape(notTooLong,outputShape)
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
    return tf.map_fn(augmentSingle, imagePack)
    

# usage example
# trDs = getTiffTrainDataSet(trIdents,trLabels) \
#    .map(loadTiffImage , num_parallel_calls=tf.data.experimental.AUTOTUNE) \
#    .filter(isValidPack) \