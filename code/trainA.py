import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import itertools
import threading
import random
import math
import matplotlib.pyplot as plt # for debugging previews only
from libExtractTile import getNotEmptyTiles
from modelA import constructModel
from skimage import io


cytoImagePath = sys.argv[1]
labelsPath = sys.argv[2]
outputPath =sys.argv[3]


labelsDf = pd.read_csv(labelsPath, engine='python')

idents = labelsDf.iloc[:,0]
labels = labelsDf.iloc[:,2]

N = len(idents)

tileSize = 1024
nnTileSize = 224
batchSize = 1
shuffleBufferSize = 32
prefetchSize = 16
trainSequenceLength = 64
seed = 35372932
random.seed(seed)
tf.random.set_seed(seed+151)

tileIndexCache = dict()
tileIndexCacheLock = threading.Semaphore(1)

trSamples = N

def getDataSet(idents,labels):
    def samplesGenerator():
        N = len(idents)
        for i in range(0,N): 
            ident = idents[i]
            label = labels[i]
            toOpen = os.path.join(cytoImagePath,"{0}.tiff".format(ident))
            #print("openning {0}".format(toOpen))
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
            #print(npTiles)
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
  shapeSet = tf.reshape(notTooLong,
    [
        trainSequenceLength,
        imagePackShape[1],
        imagePackShape[2],
        imagePackShape[3]
    ])
  return shapeSet, label

def downscale(imagePack,label):
    resized = tf.image.resize(
        imagePack,
        [nnTileSize,nnTileSize], antialias=True,
        method=tf.image.ResizeMethod.AREA
        )
    return resized,label


# TODO: shuffle slices
# TODO: augment slices
# .shuffle(shuffleBufferSize,seed=seed+31) \
    
tr_ds = tr_ds \
    .map(downscale, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .map(coerceSeqSize, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .repeat() \
    .batch(batchSize, drop_remainder=False) \
    .prefetch(prefetchSize)

def previewSample(dsElem):
    imagePack,label = dsElem
    N,_,_,_ = imagePack.shape
    print("Pack size is {0}".format(N))
    #print(imagePack)
    cols = round(math.sqrt(N))
    rows = math.ceil(N/cols)

    plt.figure()
    plt.title("tile [0]")    
    plt.imshow(imagePack[0] / 255.0)


    plt.figure()
    plt.title("tile [1]")    
    plt.imshow(imagePack[1] / 255.0)


    fig, ax = plt.subplots(rows,cols)    
    fig.set_facecolor((0.3,0.3,0.3))

    print("label is {0}".format(label))

    idx = 1
    for row in range(0,rows):
        for col in range(0,cols):            
            row = (idx - 1) // cols
            col = (idx -1) % cols
            #ax[row,col].set_title("tile [{0},{1}]".format(tile_r,tile_c))    
            
            ax[row,col].axis('off')
            if idx-1 < N:
                im = np.squeeze(imagePack[idx-1,:,:,:])
                im = im / 255.0 
                # as data contains float range 0.0 - 255.-
                # to make imshow work properly we need to map into interval 0.0 - 1.0
                ax[row,col].imshow(im) 
            idx = idx + 1
    plt.show()  # display it

#testData = list(tr_ds.take(3).as_numpy_iterator())
#previewSample(testData[0])

model = constructModel(trainSequenceLength)
print("model constructed")

csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(outputPath,'training_log.csv'), append=True)
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose =1,
#                                patience=int(5), min_lr=1e-7)


callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    #tf.keras.callbacks.EarlyStopping(patience=int(10), monitor='val_loss',mode='min'),
    # Write TensorBoard logs to `./logs` directory
    #tf.keras.callbacks.TensorBoard(log_dir=experiment_output_dir, histogram_freq = 0, profile_batch=0),
    tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(outputPath,"weights.hdf5"),
            save_best_only=True,
            verbose=True,
            mode='min',
            save_weights_only=True,
            #monitor='val_root_recall'
            mintor='loss'
            ),
    tf.keras.callbacks.TerminateOnNaN(),
    csv_logger,
    #reduce_lr
  ]

loss = tf.keras.losses.LogCosh(
    #reduction=losses_utils.ReductionV2.AUTO,
    name='logcosh'
)


model.compile(
          #optimizer=tf.keras.optimizers.SGD(momentum=.5,nesterov=True, clipnorm=1.),
          optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
          #optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
          loss=loss,
          # metrics=[
          #     #auroc,
          #     dsc
          #     #tf.keras.metrics.MeanIoU(num_classes=2)
          #     ]
          )
print("model compiled")
print(model.summary())

model.fit(x = tr_ds, \
      #validation_data = va_ds,
      #validation_steps = 1024//batch_size,
      #initial_epoch=initial_epoch,
      verbose = 1,
      callbacks=callbacks,
      shuffle=False, # dataset is shuffled explicilty
      steps_per_epoch= int(math.ceil(trSamples / batchSize)),
      epochs=5)


print("Done")

