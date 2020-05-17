import sys
import os
import multiprocessing
from skimage import io
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import cv2
from libExtractTile import getNotEmptyTiles
import npImageNormalizations as npImNorm
import npImageTransformation as npImTrans

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
    outImageSize = task['outImageSize']
    #print("reading image from disk {0}".format(ident))
    im = 255 - io.imread(tiffPath,plugin="tifffile")
    
    # initial downscaling (to speed up)
    initial_downscale_factor = 2
    h,w,_ = im.shape
    #print("initial downscale")
    im = cv2.resize(im, dsize=(w // initial_downscale_factor, h // initial_downscale_factor), interpolation=cv2.INTER_AREA)
    tileSize = tileSize // initial_downscale_factor

    N = 10
    rotStep = 360.0 / N
    gatheredTiles = []
    for i in range(0,N):
        effectiveDegree = rotStep*i
        #print("rotating for {0}".format(effectiveDegree))
        rotated = npImTrans.RotateWithoutCrop(im, effectiveDegree)
        #print("getting tiles")
        _,tiles = getNotEmptyTiles(rotated,tileSize)

        #print("normalizing")

        # normalizing with contrasts
        contrasts = []
        means = []
        for tile in tiles:        
            mu = npImNorm.getImageMean_withoutPureBlack(tile)
            means.append(mu)
            contrasts.append(npImNorm.getImageContrast_withoutPureBlack(tile, precomputedMu=mu))
        meanContrast = np.mean(contrasts)    
        meanMean = np.mean(means)
        for i in range(0,len(tiles)):
            tiles[i] = npImNorm.GCNtoRGB_uint8(npImNorm.GCN(tiles[i], lambdaTerm=0.0, precomputedContrast=meanContrast, precomputedMean=meanMean), cutoffSigmasRange=1.0)
        #print("resizing")

        if outImageSize != tileSize:
            resizedTiles = []
            for tile in tiles:
                resizedTiles.append(cv2.resize(tile, dsize=(outImageSize, outImageSize), interpolation=cv2.INTER_AREA))
            tiles = resizedTiles
        gatheredTiles.append(tiles)
    gatheredTiles = [item for sublist in gatheredTiles for item in sublist]
    if len(gatheredTiles) > 0:
        #print("encoding")
        savePackAsTFRecord(gatheredTiles,tfrecordsPath)
    #print("done")
    return len(gatheredTiles)

if __name__ == '__main__':

    imagesPath = sys.argv[1]
    outPath = sys.argv[2]
    outHistFile = sys.argv[3]
    #outIdxFile = sys.argv[4]
    tileSize = int(sys.argv[4])
    outImageSize = int(sys.argv[5])

    print("tiff images path: {0}".format(imagesPath))
    print("out dir: {0}".format(outPath))
    print("tile size: {0}".format(tileSize))

    M = multiprocessing.cpu_count()
    #M = 1
    p = multiprocessing.Pool(M + 1)    
    print("Detected {0} CPU cores".format(M))

    files = os.listdir(imagesPath)
    tiffFiles = [x for x in files if x.endswith(".tiff")]

    # uncomment for short (test) run
    #tiffFiles = tiffFiles[0:20]

    tasks = list()
    for tiffFile in tiffFiles:
        task = {
            'ident' : tiffFile[:-5],
            'tiffPath': os.path.join(imagesPath,tiffFile),
            'tfrecordsPath': os.path.join(outPath,"{0}.tfrecords".format(tiffFile[:-5])),
            'tileSize': tileSize,
            'outImageSize': outImageSize
        }
        tasks.append(task)
    print("Starting {0} conversion tasks".format(len(tasks)))

    tilesCounts = p.map(ProcessTask,tasks)
    print("Analyzing tile count frequencies")

    freqDict = dict()
    for tilesCount in tilesCounts:                
        if tilesCount in freqDict:
            prevCount = freqDict[tilesCount]
        else:
            prevCount = 0
        freqDict[tilesCount] = prevCount + 1

    freqDf = pd.DataFrame.from_dict(freqDict, orient='index', columns=["Counts"])    
    freqDf.sort_index(inplace=True)
    freqDf.to_csv(outHistFile)
    print(freqDf)
    print("Tile count histogram written")

    #with open(outIdxFile, 'w', encoding='utf-8') as f:
    #    json.dump(tileIdxDict, f, ensure_ascii=False)
    #print("Tile index written")

    print("Done")
