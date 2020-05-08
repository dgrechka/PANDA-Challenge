import sys
import os
import multiprocessing
from skimage import io
import tensorflow as tf
import numpy as np
import pandas as pd
from libExtractTile import getNotEmptyTiles

def savePackAsTFRecord(imageList,outputFilename):
    """Saves a list of HxWxC uin8 images into .tfrecords file and GZIPing them"""
    N = len(imageList)
    w,h,c = imageList[0].shape
    imagePack = np.stack(imageList,axis=0)
    
    imageFeature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[imagePack.tobytes()]))
    sizeFeature = tf.train.Feature(int64_list=tf.train.Int64List(value=[N,w,h,c]))

    featureDict = {
      'image': imageFeature,
      'size': sizeFeature,
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=featureDict))    
    with tf.io.TFRecordWriter(outputFilename,"GZIP") as writer:
        writer.write(example_proto.SerializeToString())
    print("Done with {0}\t-\t{1}\ttiles".format(outputFilename,N))
    return N

def ProcessTask(task):
    tiffPath = task['tiffPath']
    tfrecordsPath = task['tfrecordsPath']
    tileSize = task['tileSize']
    im = io.imread(tiffPath,plugin="tifffile")
    _,tiles = getNotEmptyTiles(im,tileSize)
    if len(tiles) > 0:
        return savePackAsTFRecord(tiles,tfrecordsPath)    
    else:
        return 0

if __name__ == '__main__':

    imagesPath = sys.argv[1]
    outPath = sys.argv[2]
    tileSize = int(sys.argv[3])

    print("tiff images path: {0}".format(imagesPath))
    print("out dir: {0}".format(outPath))
    print("tile size: {0}".format(tileSize))

    M = multiprocessing.cpu_count()
    p = multiprocessing.Pool(M)    
    print("Detected {0} CPU cores".format(M))

    files = os.listdir(imagesPath)
    tiffFiles = [x for x in files if x.endswith(".tiff")]

    #tiffFiles = tiffFiles[0:10]

    tasks = list()
    for tiffFile in tiffFiles:
        task = {
            'tiffPath': os.path.join(imagesPath,tiffFile),
            'tfrecordsPath': os.path.join(outPath,"{0}.tfrecords".format(tiffFile[:-5])),
            'tileSize': tileSize
        }
        tasks.append(task)
    print("Starting {0} conversion tasks".format(len(tasks)))

    tileCounts = p.map(ProcessTask,tasks)
    print("Analyzing tile count frequencies")

    freqDict = dict()
    for tileCount in tileCounts:
        if tileCount in freqDict:
            prevCount = freqDict[tileCount]
        else:
            prevCount = 0
        freqDict[tileCount] = prevCount + 1

    freqDf = pd.DataFrame.from_dict(freqDict, orient='index', columns=["Counts"])    
    freqDf.sort_index(inplace=True)
    print(freqDf)

    print("Done")
