import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import math
import multiprocessing
import tfDataProcessing as tfdp
from modelA import constructModel

trainTfRecordsPathEnv = "trainTfRecordsPath"

if not(trainTfRecordsPathEnv in os.environ):
    print("Can't find environmental variable {0}".format(trainTfRecordsPathEnv))
    exit(1)

cytoImagePath = os.environ[trainTfRecordsPathEnv]

print("TFRecords path is {0}".format(cytoImagePath))

checkpointPath = sys.argv[1]
labelsPath = sys.argv[2]
singleImagePerClusterPath = sys.argv[3]
valRowsPath = sys.argv[4]
outputPath = sys.argv[5]
outputConfusionPath = sys.argv[6]
sequenceLength = int(sys.argv[7])
prefetchSize = multiprocessing.cpu_count() + 1
batchSize = 4
truncateSize = 10000000 # usefull to debugging    
print("Processing no more than {0} samples".format(truncateSize))

clusterDf = pd.read_csv(singleImagePerClusterPath, engine='python')
valIdxDf = pd.read_csv(valRowsPath, engine='python')
valIdx = valIdxDf.iloc[:,0]
vaClusterDf  = clusterDf.iloc[list(valIdx),:]
trClusterDf = clusterDf[~clusterDf.index.isin(vaClusterDf.index)]

vaClusters = set(vaClusterDf.iloc[:,1]) # image_cluster_id
trClusterS = set(trClusterDf.iloc[:,1])
print("{0} image clusters in train set, {1} image clusters in val set".format(len(trClusterS),len(vaClusters)))

print("Labels file {0}".format(labelsPath))
labelsDf = pd.read_csv(labelsPath, engine='python')
isupMap = dict()
gleasonMap = dict()
sourceMap = dict()
isValidationMap = dict()
for row in labelsDf.itertuples():
    image_id = row.image_id
    isValidation = row.image_cluster_id in vaClusters
    isupMap[image_id] = int(row.isup_grade)
    gleasonMap[image_id] = row.gleason_score
    sourceMap[image_id] = row.data_provider
    isValidationMap[image_id] = isValidation
print("Labels loaded")



tfrFilenames = os.listdir(cytoImagePath)
print("{0} tfrecords files initially found".format(len(tfrFilenames)))
tfrFilenames = [fname for fname in tfrFilenames if fname[33:-10]=="0"]
print("{0} tfrecords are without rotations".format(len(tfrFilenames)))

if(len(tfrFilenames) > truncateSize):
    tfrFilenames = tfrFilenames[0:truncateSize]

tfrFilenameBases = [fname[0:32] for fname in tfrFilenames if fname.endswith(".tfrecords")]
tfrFullFilenames = [os.path.join(cytoImagePath,fname) for fname in tfrFilenames if fname.endswith(".tfrecords")]
print("Found {0} tfrecords files to pridict".format(len(tfrFilenames)))

def imageTransform(imagePack):
    return tfdp.coerceSeqSize(imagePack,sequenceLength)

imagesDs = tfdp.getTfRecordDataset(tfrFullFilenames) \
    .map(tfdp.extractTilePackFromTfRecord) \
    .map(imageTransform) \
    .batch(batchSize, drop_remainder=False) \
    .prefetch(prefetchSize)

model,backbone = constructModel(sequenceLength, DORate=0.3, l2regAlpha = 0.0)
print("model constructed")

backbone.trainable = False

if os.path.exists(checkpointPath):
  print("Loading pretrained weights {0}".format(checkpointPath))
  model.load_weights(checkpointPath, by_name=True)
  print("Loaded pretrained weights {0}".format(checkpointPath))
else:
  print("Pretrained weights file does not exist: {0}".format(checkpointPath))
  exit(1)

print(model.summary())
print("predicting")

predicted = np.squeeze(model.predict(imagesDs,verbose=1))
print("predicted shape is {0}".format(predicted.shape))
predictedList = predicted.tolist()
isupList = []
providerList = []
gleasonList = []
isupDiffList = []
isValidationList = []
isupAbsDiffList = []

i = 0
for image_id in tfrFilenameBases:
    isup_truth = int(isupMap[image_id])
    isupList.append(isup_truth)
    providerList.append(sourceMap[image_id])
    gleasonList.append(gleasonMap[image_id])
    isValidationList.append(isValidationMap[image_id])


    isup_diff = predictedList[i] - isup_truth

    isupDiffList.append(isup_diff)
    isupAbsDiffList.append(abs(isup_diff))

    i+=1
    if i == truncateSize:
        break

resDf = pd.DataFrame.from_dict(
    {
        'file': tfrFilenameBases,
        'isup_grade_pred': predictedList,
        'isup_grade_truth': isupList,
        'isup_abs_diff': isupAbsDiffList,
        'isup_diff': isupDiffList,
        'provider': providerList,
        'gleason_score': gleasonList,
        'is_validation': isValidationList
    }
)

resDf.sort_values(by=['isup_abs_diff'], inplace=True, ascending=False)
resDf.to_csv(outputPath, index=False)

predictedRoundedList = [round(pred) for pred in predictedList]

y_actu = pd.Series(isupList, name='Actual')
y_pred = pd.Series(predictedRoundedList, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print("Confusion matrix")
print(df_confusion)
df_confusion.to_csv(outputConfusionPath)

print("Done")

