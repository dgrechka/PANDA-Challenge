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

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} 3 - error, 0 - debug
#tf.get_logger().setLevel("ERROR")
#tf.autograph.set_verbosity(0)

trainTfRecordsPathEnv = "trainTfRecordsPath"

if not(trainTfRecordsPathEnv in os.environ):
    print("Can't find environmental variable {0}".format(trainTfRecordsPathEnv))
    exit(1)

cytoImagePath = os.environ[trainTfRecordsPathEnv]

print("TFRecords path is {0}".format(cytoImagePath))

labelsPath = sys.argv[1]
singleImagePerClusterPath = sys.argv[2]
valRowsPath = sys.argv[3]
checkpointPath = sys.argv[4]
outputPath = sys.argv[5]
trainSequenceLength = int(sys.argv[6])

batchSize = 2
shuffleBufferSize = 512
prefetchSize = multiprocessing.cpu_count() + 1
seed = 36543452
epochsToTrain = 160
random.seed(seed)
tf.random.set_seed(seed+151)

#gpu = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu[0], True)

def RemoveInvalidLabels(dataFrame):
    print("{0} images before removing absent".format(len(dataFrame)))
    
    dfIdents = list(dataFrame['image_id'])
    filteredDf = dataFrame
    for ident in dfIdents:
        doSkip = False
        if ident == "b0a92a74cb53899311acc30b7405e101":
            doSkip = True # wierd labeld image
        # if not os.path.exists(os.path.join(cytoImagePath,"{0}.tfrecords".format(ident))):
        #     print("Tiles for {0} are missing. Skipping this image".format(ident))
        #     doSkip = True
        if doSkip:
            filteredDf = filteredDf[filteredDf['image_id'] != ident]
    print("{0} images after removing absent".format(len(filteredDf)))
    return filteredDf


clusterDf = pd.read_csv(singleImagePerClusterPath, engine='python')
labelsDf = pd.read_csv(labelsPath, engine='python')

valIdxDf = pd.read_csv(valRowsPath, engine='python')
valIdx = valIdxDf.iloc[:,0]
#   print(valIdxDf)

vaClusterDf  = clusterDf.iloc[list(valIdx),:]
trClusterDf = clusterDf[~clusterDf.index.isin(vaClusterDf.index)]

vaClusters = set(vaClusterDf.iloc[:,1]) # image_cluster_id
trClusterS = set(trClusterDf.iloc[:,1])
print("{0} image clusters in train set, {1} image clusters in val set".format(len(trClusterS),len(vaClusters)))

vaRowNums = list()
rowIdx = 0
labelsDict = dict()
for row in labelsDf.itertuples():
    if row.image_cluster_id in vaClusters:
        vaRowNums.append(rowIdx)
    labelsDict[row.image_id] = int(row.isup_grade)
    rowIdx += 1

vaLabelsDf  = labelsDf.iloc[list(vaRowNums),:]
trLabelsDf = labelsDf[~labelsDf.index.isin(vaLabelsDf.index)]

print("{0} images in train set, {1} images in val set".format(len(trLabelsDf),len(vaLabelsDf)))

vaLabelsDf = RemoveInvalidLabels(vaLabelsDf)
trLabelsDf = RemoveInvalidLabels(trLabelsDf)

# debug run of srinked DS
#trLabelsDf = trLabelsDf.iloc[0:500,]
#vaLabelsDf = vaLabelsDf.iloc[0:100,]

trIdents = set(trLabelsDf.iloc[:,0])
vaIdents = set(vaLabelsDf.iloc[:,0])

trFilenames = os.listdir(cytoImagePath)
trFilenames = [fname for fname in trFilenames if fname.endswith(".tfrecords")]
print("Found {0} tfrecords files in source dir".format(len(trFilenames)))
trTfRecordFileNames = list()
trLabels = list()
vaTfRecordFileNames = list()
vaLabels = list()
for trFilename in trFilenames:
    imIdent = trFilename[0:32]
    rotIdx = int(trFilename[33:-10])
    fullPath = os.path.join(cytoImagePath,trFilename)
    label = labelsDict[imIdent]
    if imIdent in trIdents:
        trTfRecordFileNames.append(fullPath)
        trLabels.append(label)
    elif imIdent in vaIdents:
        if rotIdx != 0:
            continue
        vaTfRecordFileNames.append(fullPath)
        vaLabels.append(label)
    else:
        print("WARN: ident {0} is nither in training nor in validation set".format(imIdent))

trSamplesCount = len(trTfRecordFileNames)
vaSamplesCount = len(vaTfRecordFileNames)

print("{0} samples in training ds, {1} samples in validation ds".format(trSamplesCount, vaSamplesCount))
#print("tf idents")
#print(trIdents)

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

def trImageTransformWithLabel(im, lab):
    return trImageTransform(im),lab

def vaImageTransofrmWithLabel(im, lab):
    return (vaImageTransofrm(im),lab)

trDs = tf.data.Dataset.zip((trImagesDs,trLabelsDs)) \
    .map(trImageTransformWithLabel, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .repeat() \
    .shuffle(shuffleBufferSize,seed=seed+31) \
    .batch(batchSize, drop_remainder=False) \
    .prefetch(prefetchSize)

valImagesDs = tfdp.getTfRecordDataset(vaTfRecordFileNames) \
    .map(tfdp.extractTilePackFromTfRecord)
valLabelsDs = tf.data.Dataset.from_tensor_slices(vaLabels)
    
valDs = tf.data.Dataset.zip((valImagesDs,valLabelsDs)) \
    .map(vaImageTransofrmWithLabel , num_parallel_calls=tf.data.experimental.AUTOTUNE) \
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

#testData = list(trDs.take(1).repeat(2).as_numpy_iterator())
#previewSample(testData[0])
#previewSample(testData[1])
#exit(1)

model = constructModel(trainSequenceLength, DORate=0.3, l2regAlpha = 0.0)
print("model constructed")

csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(outputPath,'training_log.csv'), append=False)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_kappa', factor=0.1, verbose =1,
                                patience=int(8), min_lr=1e-7, mode='max')


callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    tf.keras.callbacks.EarlyStopping(patience=int(17), monitor='val_kappa',mode='max'),
    # Write TensorBoard logs to `./logs` directory
    #tf.keras.callbacks.TensorBoard(log_dir=outputPath, histogram_freq = 5, profile_batch=0),
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
    reduce_lr
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
          optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5, clipnorm=1.),
          #optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
          loss=loss,
          metrics=[QuadraticWeightedKappa(), tf.keras.metrics.MeanAbsoluteError(name="mae")]
          )
print("model compiled")
print(model.summary())



model.fit(x = trDs, \
      validation_data = valDs,
      validation_steps = int(math.floor(vaSamplesCount / batchSize)),
      #initial_epoch=initial_epoch,
      verbose = 2,
      callbacks=callbacks,
      shuffle=False, # dataset is shuffled explicilty
      steps_per_epoch= int(math.ceil(trSamplesCount / batchSize) / 1),
      epochs=epochsToTrain)

print("Done")

