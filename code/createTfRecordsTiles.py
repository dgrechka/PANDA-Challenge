import sys
import os
import multiprocessing
from skimage import io
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import math
#import matplotlib.pyplot as plt
import cv2
from libExtractTile import getNotEmptyTiles
import npImageNormalizations as npImNorm
import npImageTransformation as npImTrans
#import tfDataProcessing as tfdp

def savePackAsTFRecord(imageList,outputFilename):
    """Saves a list of HxWxC uin8 images into .tfrecords file and GZIPing them"""
    N = len(imageList)
    for image in imageList:
        if len(image.shape) != 3:
            print("image shape {0}".format(image.shape))
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
    #print("Done with {0}\t-\t{1}\ttiles".format(outputFilename,N))

def ProcessGenerateRecordTask(task):
    ident = task['ident']
    tiffPath = task['tiffPath']
    tfrecordsPath = task['tfrecordsPath']
    tileSize = task['tileSize']
    outImageSize = task['outImageSize']    
    minRequiredTiles = task['minRequiredTiles']
    rotationDegree = task['rotationDegree']
    #print("processing {0}".format(ident))
    #print("reading image from disk {0}".format(ident))
    
    im = 255 - io.imread(tiffPath,plugin="tifffile")
    
    # initial downscaling (to speed up)
    initial_downscale_factor = task['initial_downscale_factor']
    h,w,_ = im.shape
    #print("initial downscale")
    im = cv2.resize(im, dsize=(w // initial_downscale_factor, h // initial_downscale_factor), interpolation=cv2.INTER_AREA)
    tileSize = tileSize // initial_downscale_factor

    #quantiles = [1/10, 1/8, 1/6, 1/5, 1/4, 1/3, 1/2, 2/3, 3/4, 4/5, 5/6, 7/8, 9/10, 1.0]
    quantiles = [3/4, 4/5, 5/6, 7/8, 9/10, 1.0]    
    activeQuantileIdx = 0
    for i in range(0,M):
        tiles = []
        tfrecordsPathIdx = "{0}-{1}.tfrecords".format(tfrecordsPath[0:-10], i)
        effectiveDegree = rotStep*i
        
        #print("Starting task {0} ({1},{2})".format(ident,h,w))
        while len(tiles) < minRequiredTiles:
            activeQuantile = quantiles[activeQuantileIdx]
            #print("rotating for {0}".format(effectiveDegree))
            if effectiveDegree != 0.0:
                #print("rotating to {0}".format(effectiveDegree))
                rotated = npImTrans.RotateWithoutCrop(im, effectiveDegree)
            else:
                rotated = im
            #print("getting tiles")
            _,tiles = getNotEmptyTiles(rotated, tileSize, emptyCuttOffQuantile=activeQuantile)

            #print("got {0} tiles ".format(len(tiles)))
            # if len(tiles) == 0:
            #     #print("angle {0} for {1} results in 0 tiles. skipping this angle".format(effectiveDegree, ident))
            #     sys.stdout.write("0")
            #     sys.stdout.flush()
            #     continue

            #print("normalizing")

            if len(tiles) < minRequiredTiles:
                if activeQuantileIdx == (len(quantiles) - 1):
                    if len(tiles) == 0:
                        print("WARN: Image {0} resulted in 0 tiles. producing blank (black) single tile TfRecords file".format(ident))
                        tiles = [ np.zeros((outImageSize,outImageSize,3),dtype=np.uint8) ]
                    break
                else:
                    activeQuantileIdx += 1
                    sys.stdout.write("[q{0:.2f}]".format(quantiles[activeQuantileIdx]))
                    sys.stdout.flush()
                    continue
            

        # normalizing with contrasts
        contrasts = []
        means = []
        #print("normalzing {0} tiles".format(len(tiles)))
        for tile in tiles:        
            mu = npImNorm.getImageMean_withoutPureBlack(tile)
            contrast = npImNorm.getImageContrast_withoutPureBlack(tile, precomputedMu=mu)
            means.append(mu)
            contrasts.append(contrast)
        meanContrast = np.mean(contrasts)    
        meanMean = np.mean(means)
        if meanMean > 0.0:
            for j in range(0,len(tiles)):
                tiles[j] = npImNorm.GCNtoRGB_uint8(npImNorm.GCN(tiles[j], lambdaTerm=0.0, precomputedContrast=meanContrast, precomputedMean=meanMean), cutoffSigmasRange=1.0)
        #print("resizing")

        if outImageSize != tileSize:
            resizedTiles = []
            for tile in tiles:
                resizedTiles.append(cv2.resize(tile, dsize=(outImageSize, outImageSize), interpolation=cv2.INTER_AREA))
            tiles = resizedTiles
        sys.stdout.write(".")
        sys.stdout.flush()

        #print("saving {0}".format(len(tiles)))
        savePackAsTFRecord(tiles,tfrecordsPathIdx)
        sys.stdout.write("({0}:{1})".format(i,len(tiles)))
        sys.stdout.flush()
    #print("done")

    # debug preview
        # N = len(gatheredTiles)
        # cols = round(math.sqrt(N))
        # rows = math.ceil(N/cols)

        # #plt.figure()
        # #r,c = tileIdx[N-1]
        # #plt.title("tile [{0},{1}]".format(r,c))    
        # #plt.imshow(gatheredTiles[N-1])

        # fig, ax = plt.subplots(rows,cols)    
        # fig.set_facecolor((0.3,0.3,0.3))

        # idx = 1
        # for row in range(0,rows):
        #     for col in range(0,cols):            
        #         row = (idx - 1) // cols
        #         col = (idx -1) % cols
        #         #ax[row,col].set_title("tile [{0},{1}]".format(tile_r,tile_c))    
                
        #         ax[row,col].axis('off')
        #         if idx-1 < N:
        #             ax[row,col].imshow(gatheredTiles[idx-1]) 
        #         idx = idx + 1
        # plt.show()  # display it

    return len(tiles)

if __name__ == '__main__':

    imagesPath = sys.argv[1]
    outPath = sys.argv[2]
    outHistFile = sys.argv[3]
    #outIdxFile = sys.argv[4]
    tileSize = int(sys.argv[4])
    outImageSize = int(sys.argv[5])
    minRequiredTilesCount = int(sys.argv[6])
    rotationStepsCount = int(sys.argv[7])
    initial_downscale_factor = int(sys.argv[8])

    print("tiff images path: {0}".format(imagesPath))
    print("out dir: {0}".format(outPath))
    print("hist file: {0}".format(outHistFile))
    print("tile size: {0}".format(tileSize))
    print("out image size: {0}".format(outImageSize))
    print("min Required Tiles Count: {0}".format(minRequiredTilesCount))
    print("rotation steps count: {0}".format(rotationStepsCount))
    print("initial downscale factor: {0}".format(initial_downscale_factor))

    M = multiprocessing.cpu_count()
    
    print("Detected {0} CPU cores".format(M))
    #M = 1
    #M //= 2 # opencv uses multithreading somehow. So we use less workers that CPU cores available
    M = 3

    p = multiprocessing.Pool(M)
    print("Created process pool of {0} workers".format(M))

    files = os.listdir(imagesPath)
    tiffFiles = [x for x in files if x.endswith(".tiff")]

    # uncomment for short (test) run
    #tiffFiles = tiffFiles[0:20]

    tasks = list()
    existsList = list()

    rotStep = 360.0 / rotationStepsCount

    for tiffFile in tiffFiles:
        tfPath = os.path.join(outPath,"{0}.tfrecords".format(tiffFile[:-5]))

        allFound = True
        for i in range(0,rotationStepsCount):
            if(not(os.path.exists("{0}-{1}.tfrecords".format(tfPath[0:-10],i)))):
                allFound = False
                break
        if allFound:
            existsList.append(tfPath)
        else:
            task = {
                'ident' : tiffFile[:-5],
                'tiffPath': os.path.join(imagesPath,tiffFile),
                'tfrecordsPath': tfPath,
                'tileSize': tileSize,
                'outImageSize': outImageSize,
                'minRequiredTiles': minRequiredTilesCount,
                'rotationStepsCount': rotationStepsCount,
                'initial_downscale_factor': initial_downscale_factor
            }
            tasks.append(task)
    
    print("Existing tfRecords file count: {0}".format(len(existsList)))

    def ExtractPackSize(imPack):
        return tf.shape(imPack)[0]

    # existingTilesCounts = \
    #     tfdp.getTfRecordDataset(existsList) \
    #     .map(tfdp.extractTilePackFromTfRecord) \
    #     .map(ExtractPackSize)
    print("Starting {0} conversion tasks".format(len(tasks)))

    tilesCounts = p.map(ProcessGenerateRecordTask,tasks)
    #tilesCounts = [ProcessGenerateRecordTask(x) for x in tasks] # single threaded debugging
    # print("Analyzing tile count frequencies")

    # tilesCounts = tilesCounts.extend(existingTilesCounts)

    # freqDict = dict()
    # for tilesCount in tilesCounts:                
    #     if tilesCount in freqDict:
    #         prevCount = freqDict[tilesCount]
    #     else:
    #         prevCount = 0
    #     freqDict[tilesCount] = prevCount + 1

    # freqDf = pd.DataFrame.from_dict(freqDict, orient='index', columns=["Counts"])    
    # freqDf.sort_index(inplace=True)
    # freqDf.to_csv(outHistFile)
    # print(freqDf)
    # print("Tile count histogram written")

    #with open(outIdxFile, 'w', encoding='utf-8') as f:
    #    json.dump(tileIdxDict, f, ensure_ascii=False)
    #print("Tile index written")

    print("Done")
