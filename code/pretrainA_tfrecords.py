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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} 3 - error, 0 - debug
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

trainTfRecordsPathEnv = "trainTfRecordsPath"

if not(trainTfRecordsPathEnv in os.environ):
    print("Can't find environmental variable {0}".format(trainTfRecordsPathEnv))

cytoImagePath = os.environ[trainTfRecordsPathEnv]
labelsPath = sys.argv[1]
valRowsPath = sys.argv[2]
outputPath = sys.argv[3]
trainSequenceLength = int(sys.argv[4])

print("TFRecords path is {0}".format(cytoImagePath))

batchSize = 2
shuffleBufferSize = 512
prefetchSize = multiprocessing.cpu_count() + 1
seed = 35372932
epochsToTrain = 4
random.seed(seed)
tf.random.set_seed(seed+151)

def RemoveInvalidLabels(dataFrame):
    print("{0} images before removing absent".format(len(dataFrame)))
    
    dfIdents = list(dataFrame['image_id'])
    filteredDf = dataFrame
    for ident in dfIdents:
        doSkip = False
        if ident == "b0a92a74cb53899311acc30b7405e101":
            doSkip = True # wierd labeld image
        if not os.path.exists(os.path.join(cytoImagePath,"{0}.tfrecords".format(ident))):
            print("Tiles for {0} are missing. Skipping this image".format(ident))
            doSkip = True
        if doSkip:
            filteredDf = filteredDf[filteredDf['image_id'] != ident]
    print("{0} images after removing absent".format(len(filteredDf)))
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


trImagesDs = tfdp.getTfRecordDataset(trTfRecordFileNames) \
    .map(tfdp.extractTilePackFromTfRecord)
trLabelsDs = tf.data.Dataset.from_tensor_slices(trLabels)

def trImageTransform(imagePack):
    return tfdp.augment(
                tfdp.coerceSeqSize(imagePack, \
                    trainSequenceLength))

def vaImageTransofrm(imagePack):
    return tfdp.coerceSeqSize(imagePack, \
                    trainSequenceLength)

trDs = tf.data.Dataset.zip((trImagesDs,trLabelsDs)) \
    .map(lambda im,lab: (trImageTransform(im),lab), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .repeat() \
    .shuffle(shuffleBufferSize,seed=seed+31) \
    .batch(batchSize, drop_remainder=False) \
    .prefetch(prefetchSize)

valImagesDs = tfdp.getTfRecordDataset(vaTfRecordFileNames) \
    .map(tfdp.extractTilePackFromTfRecord)
valLabelsDs = tf.data.Dataset.from_tensor_slices(vaLabels)
    
valDs = tf.data.Dataset.zip((valImagesDs,valLabelsDs)) \
    .map(lambda im,lab: (vaImageTransofrm(im),lab), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
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

model = constructModel(trainSequenceLength, DORate=0.3)
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
            monitor='loss' # as we pretrain later layers, we do not care about overfitting. thus loss instead of val_los
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
          optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4, clipnorm=1.),
          #optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
          loss=loss,
          metrics=[QuadraticWeightedKappa()]
          )
print("model compiled")
print(model.summary())

model.fit(x = trDs, \
      #validation_data = valDs,
      #validation_steps = int(math.ceil(vaSamplesCount / batchSize)),
      #initial_epoch=initial_epoch,
      verbose = 2,
      callbacks=callbacks,
      shuffle=False, # dataset is shuffled explicilty
      steps_per_epoch= int(math.ceil(trSamplesCount / batchSize)),
      epochs=epochsToTrain)

print("Done")

