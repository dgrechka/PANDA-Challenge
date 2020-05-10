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
from tfQuadraticWeightedKappa import QuadraticWeightedKappa
import tfDataProcessing
from modelA import constructModel
from skimage import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} 3 - error, 0 - debug
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

cytoImagePath = sys.argv[1]
labelsPath = sys.argv[2]
valRowsPath = sys.argv[3]
checkpointPath = sys.argv[4]
outputPath = sys.argv[5]

tileSize = 1024
nnTileSize = 224
batchSize = 1
shuffleBufferSize = 32
prefetchSize = 16
trainSequenceLength = 15
seed = 35372932
epochsToTrain = 10
random.seed(seed)
tf.random.set_seed(seed+151)

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

def RemoveInvalidLabels(dataFrame):
    print("{0} images before removing whole white".format(len(dataFrame)))
    # whole white image
    #corruptedIdx = dataFrame[dataFrame['image_id'] == "3790f55cad63053e956fb73027179707"].index
    filteredDf = dataFrame[dataFrame['image_id'] != "3790f55cad63053e956fb73027179707"]
    print("{0} images after removing whole white".format(len(filteredDf)))
    return filteredDf


labelsDf = pd.read_csv(labelsPath, engine='python')

valIdxDf = pd.read_csv(valRowsPath, engine='python')
valIdx = valIdxDf.iloc[:,0]
#   print(valIdxDf)

vaLabelsDf  = labelsDf.iloc[list(valIdx),:]
trLabelsDf = labelsDf[~labelsDf.index.isin(vaLabelsDf.index)]

vaLabelsDf = RemoveInvalidLabels(vaLabelsDf)
trLabelsDf = RemoveInvalidLabels(trLabelsDf)

# debug run of srinked DS
#trLabelsDf = trLabelsDf.iloc[0:500,]
#vaLabelsDf = vaLabelsDf.iloc[0:100,]


trIdents = list(trLabelsDf.iloc[:,0])
trTfRecordFileNames = [os.path.join(cytoImagePath,"{0}.tfrecords".format(x)) for x in trIdents]
trLabels = list(trLabelsDf.iloc[:,2])
vaIdents = list(vaLabelsDf.iloc[:,0])
vaTfRecordFileNames = [os.path.join(cytoImagePath,"{0}.tfrecords".format(x)) for x in vaIdents]
vaLabels = list(vaLabelsDf.iloc[:,2])

#print("tf idents")
#print(trIdents)

trSamplesCount = len(trLabelsDf)
vaSamplesCount = len(vaLabelsDf)

print("{0} training samples, {1} val sample, {2} samples in total".format(trSamplesCount, vaSamplesCount, len(labelsDf)))


def coerceSeqSize(imagePack, label):
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


trImagesDs = tfDataProcessing.getTfRecordDataset(trTfRecordFileNames) \
    .map(tfDataProcessing.extractTilePackFromTfRecord)
trLabelsDs = tf.data.Dataset.from_tensor_slices(trLabels)
    
trDs = tf.data.Dataset.zip((trImagesDs,trLabelsDs)) \
    .map(downscale, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .map(coerceSeqSize, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .repeat() \
    .shuffle(shuffleBufferSize,seed=seed+31) \
    .batch(batchSize, drop_remainder=False) \
    .prefetch(prefetchSize)

valImagesDs = tfDataProcessing.getTfRecordDataset(vaTfRecordFileNames) \
    .map(tfDataProcessing.extractTilePackFromTfRecord)
valLabelsDs = tf.data.Dataset.from_tensor_slices(vaLabels)
    
valDs = tf.data.Dataset.zip((valImagesDs,valLabelsDs)) \
    .map(downscale, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .map(coerceSeqSize, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
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
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_kappa', factor=0.1, verbose =1,
                                patience=int(3), min_lr=1e-7, mode='max')


callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    tf.keras.callbacks.EarlyStopping(patience=int(5), monitor='val_kappa',mode='max'),
    # Write TensorBoard logs to `./logs` directory
    #tf.keras.callbacks.TensorBoard(log_dir=experiment_output_dir, histogram_freq = 0, profile_batch=0),
    tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(outputPath,"weights.hdf5"),
            save_best_only=True,
            verbose=True,
            mode='max',
            save_weights_only=True,
            #monitor='val_root_recall'
            monitor='val_kappa' # as we pretrain later layers, we do not care about overfitting. thus loss instead of val_los
            ),
    tf.keras.callbacks.TerminateOnNaN(),
    csv_logger,
    #reduce_lr
  ]

loss = tf.keras.losses.LogCosh(
    #reduction=losses_utils.ReductionV2.AUTO,
    name='logcosh'
)

if os.path.exists(checkpointPath):
  print("Loading pretrained weights {0}".format(checkpointPath))
  model.load_weights(checkpointPath, by_name=True)
  print("Loaded pretrained weights {0}".format(checkpointPath))
else:
  print("Starting learning from scratch")

for i in range(len(model.layers)):
  model.layers[i].trainable = True



model.compile(
          #optimizer=tf.keras.optimizers.SGD(momentum=.5,nesterov=True, clipnorm=1.),
          optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
          #optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
          loss=loss,
          metrics=[QuadraticWeightedKappa()]
          )
print("model compiled")
print(model.summary())

model.fit(x = trDs, \
      validation_data = valDs,
      validation_steps = int(math.ceil(vaSamplesCount / batchSize)),
      #initial_epoch=initial_epoch,
      verbose = 1,
      callbacks=callbacks,
      shuffle=False, # dataset is shuffled explicilty
      steps_per_epoch= int(math.ceil(trSamplesCount / batchSize)),
      epochs=epochsToTrain)

print("Done")

