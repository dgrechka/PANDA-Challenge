import sys
import os
import multiprocessing
from skimage import io
import tensorflow as tf
import numpy as np
import pandas as pd
import json
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

def ProcessTask(task):
    ident = task['ident']
    tiffPath = task['tiffPath']
    tfrecordsPath = task['tfrecordsPath']
    tileSize = task['tileSize']
    im = io.imread(tiffPath,plugin="tifffile")
    tilesIdx,tiles = getNotEmptyTiles(im,tileSize)
    if len(tiles) > 0:
        savePackAsTFRecord(tiles,tfrecordsPath)
    return tilesIdx,ident

if __name__ == '__main__':

    imagesPath = sys.argv[1]
    outPath = sys.argv[2]
    outHistFile = sys.argv[3]
    outIdxFile = sys.argv[4]
    tileSize = int(sys.argv[5])

    print("tiff images path: {0}".format(imagesPath))
    print("out dir: {0}".format(outPath))
    print("tile size: {0}".format(tileSize))

    M = multiprocessing.cpu_count()
    p = multiprocessing.Pool(M+1)    
    print("Detected {0} CPU cores".format(M))

    files = os.listdir(imagesPath)
    tiffFiles = [x for x in files if x.endswith(".tiff")]

    # uncomment for short (test) run
    # tiffFiles = tiffFiles[0:10]

    tasks = list()
    for tiffFile in tiffFiles:
        task = {
            'ident' : tiffFile[:-5],
            'tiffPath': os.path.join(imagesPath,tiffFile),
            'tfrecordsPath': os.path.join(outPath,"{0}.tfrecords".format(tiffFile[:-5])),
            'tileSize': tileSize
        }
        tasks.append(task)
    print("Starting {0} conversion tasks".format(len(tasks)))

    tilesIdx = p.map(ProcessTask,tasks)
    print("Analyzing tile count frequencies")

    freqDict = dict()
    tileIdxDict = dict()
    for tileIdx,ident in tilesIdx:
        tileIdxDict[ident] = tileIdx
        tileCount = len(tileIdx)
        if tileCount in freqDict:
            prevCount = freqDict[tileCount]
        else:
            prevCount = 0
        freqDict[tileCount] = prevCount + 1

    freqDf = pd.DataFrame.from_dict(freqDict, orient='index', columns=["Counts"])    
    freqDf.sort_index(inplace=True)
    freqDf.to_csv(outHistFile)
    print(freqDf)
    print("Tile count histogram written")

    with open(outIdxFile, 'w', encoding='utf-8') as f:
        json.dump(tileIdxDict, f, ensure_ascii=False)
    print("Tile index written")

    print("Done")
