import os
import tensorflow as tf
import numpy as np
import pandas as pd
import tfDataProcessing
import sys
from tqdm import tqdm

tfRecordsDir = sys.argv[1]
outHistFile = sys.argv[2]

files = os.listdir(tfRecordsDir)
files = [x for x in files if x.endswith('.tfrecords')]
print("{0} files to analyze".format(len(files)))

fullTfRecordPaths = [os.path.join(tfRecordsDir,x) for x in files]

N = len(files)
N = 50

ds = tfDataProcessing.getTfRecordDataset(fullTfRecordPaths) \
    .map(tfDataProcessing.extractTilePackFromTfRecord) \
    .prefetch(16) \
    .take(N)


print("Analyzing tile count frequencies")

freqDict = dict()
for sample in tqdm(ds.as_numpy_iterator(), total=N,ascii=True):
    tileCount,_,_,_ = sample.shape
    if tileCount in freqDict:
        prevCount = freqDict[tileCount]
    else:
        prevCount = 0
    freqDict[tileCount] = prevCount + 1

freqDf = pd.DataFrame.from_dict(freqDict, orient='index', columns=["Counts"])    
freqDf.sort_index(inplace=True)
freqDf.to_csv(outHistFile)
print(freqDf)

