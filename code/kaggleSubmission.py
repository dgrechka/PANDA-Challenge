import time
import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf
import math
import os
from skimage import io
import cv2

isTestSetRun = False
input_dir = '/mnt/ML/Panda/officialData/train_images/'
checkpoint_path = 'experiments/dgrechka/35c/train2/fold_1/weights.hdf5'
out_file = 'submission.csv'
backboneFrozen = True
batchSize = 4
trainSequenceLength = 64
tileSizeSetting = 1024
outImageSize = 224
prefetchSize = multiprocessing.cpu_count() + 1
DORate = 0.3


cpuCores = multiprocessing.cpu_count()
print("Detected {0} CPU cores".format(cpuCores))

inputFiles = os.listdir(input_dir)
inputIdents = [x[0:-5] for x in inputFiles if x.endswith(".tiff")]

if not isTestSetRun:
    inputIdents.sort()
    inputIdents = inputIdents[0:100]

imageCount = len(inputIdents)
print("Found {0} files for inference".format(imageCount))
fullPaths = [os.path.join(input_dir,"{0}.tiff".format(x)) for x in inputIdents]

def GetNonBlackArea(image):
    rowsAggregated = np.amax(image,axis=(0,2))
    rowIndices = np.where(rowsAggregated != 0)[0]
    if len(rowIndices) == 0:
        print("WARN: entire black image in TrimBlackPaddings")
        return image # entire black image
    firstRow,lastRow = rowIndices[0], rowIndices[-1]

    colsAggregated = np.amax(image, axis=(1,2))
    colIndices = np.where(colsAggregated != 0)[0]
    if len(colIndices) == 0:
        print("WARN: entire black image in TrimBlackPaddings")
        return image # entire black image

    firstCol, lastCol = colIndices[0], colIndices[-1]
    return firstCol, firstRow, lastCol, lastRow

def TrimBlackPaddings(image, precomputedBounds = None):
    if precomputedBounds is None:
        firstCol, firstRow, lastCol, lastRow = GetNonBlackArea(image)
    else:
        firstCol, firstRow, lastCol, lastRow = precomputedBounds

    return image[firstCol:(lastCol+1), firstRow:(lastRow+1), :]

def RotateWithoutCrop(image, angleDeg):
    angleDeg = angleDeg % 360.0
    #print("cropping initial")
    image = TrimBlackPaddings(image)

    if abs(angleDeg) < 1e-6:
        return image
    else:
        h,w,_ = image.shape
        diag = math.sqrt(h*h + w*w)
        diagInt = int(diag)
        padH = diagInt - h
        padW = diagInt - w

        if diagInt > 32768:
            print("WARN: image size is more than 32768 in RotateWithoutCrop. Will not rotate and return the image as is‬")
            return image

        #print("padding")
        paddedImage = np.pad(image, (
            (padH // 2, padH // 2),
            (padW // 2, padW // 2),
            (0,0)))

        paddedH, paddedW, _ = paddedImage.shape

        #print("paddedH {0}; paddedW {1}; diag {2}".format(paddedH, paddedW, diagInt))

        center = int(diag * 0.5)
        #print("affine transofrming")
        rot = cv2.getRotationMatrix2D((center,center), angleDeg, 1)
        #print("angle {2:.2f}\t\ttarget rotation size is {0},{1}".format(paddedW, paddedH,angleDeg))
        rotated = cv2.warpAffine(paddedImage, rot, (paddedW,paddedH))

        return TrimBlackPaddings(rotated)

def getNotEmptyTiles(image, tileSize, precomputedTileIndices=None, emptyCuttOffQuantile = 3/4, emptyCutOffMaxThreshold = 25):
    '''Returns the list of non-empty tile indeces (tile_row_idx,tile_col_idx) and corresponding list of tile npArrays).
    Each index list element specifies the zero based index (row_idx,col_idx) of square tile which contain some data (non-empty)'''

    # negate image (as lots of further processing operations add zero paddings)
    # we need black background

    h,w,_ = image.shape
    vertTiles = math.ceil(h / tileSize)
    horTiles = math.ceil(w / tileSize)
    indexResult = []
    dataResult = []
    tileIntensity  = []

    def extractTileData(row_idx,col_idx):
        tileYstart = row_idx*tileSize
        tileYend = min((row_idx+1)*tileSize,h)
        tileXstart = col_idx*tileSize
        tileXend = min((col_idx+1)*tileSize,w)
        tile = image[
            tileYstart:tileYend,
            tileXstart:tileXend,
            :] # all 3 color channels
        return tile

    def coerceTileSize(tile):
        """In case tile is smaller than requested size, it is padded with white content"""
        # we may need to pad the edge tile to full requested tile size
        th,tw,_ = tile.shape
        xPad = tileSize - tw
        yPad = tileSize - th
        if (xPad>0) | (yPad>0):
            # we pad to the end
            tile = np.pad(tile,[[0,yPad],[0,xPad],[0,0]],constant_values = 0)
        return tile

    if precomputedTileIndices == None:
        # we analyze all tiles for the content
        for row_idx in range(0,vertTiles):
            for col_idx in range(0,horTiles):
                tile = extractTileData(row_idx, col_idx)
                # if the tile contains pure white pixels (no information) we skip it (do not return)
                #print("row {0} col {1} min_v {2}".format(row_idx,col_idx,tileMin))
                if not(emptyCutOffMaxThreshold is None):
                    tileMax = np.nanmax(tile)
                    if tileMax < emptyCutOffMaxThreshold: # too black! there is no tissue areas
                        continue
                if not(emptyCuttOffQuantile is None):
                    tileQuantile = np.quantile(tile, emptyCuttOffQuantile)
                    if tileQuantile < 15: # too many black pixels portion
                        continue

                tile = coerceTileSize(tile)

                indexResult.append((row_idx,col_idx))
                dataResult.append(tile)
                
                tileMean = np.nanmean(tile)
                tileIntensity.append(tileMean)
            
        
        # sorting the tiles according to intensity
        resIdx = []
        resData = []
        for (idxElem,dataElem,_) in sorted(zip(indexResult, dataResult, tileIntensity), key=lambda pair: -pair[2]):
            resIdx.append(idxElem)
            resData.append(dataElem)
        indexResult = resIdx
        dataResult = resData
    else:
        # do not analyse all possible tiles for return
        # just return requested tiles (possibly padded)
        for row_idx,col_idx in precomputedTileIndices:
            tile = extractTileData(row_idx, col_idx)
            tile = coerceTileSize(tile)
            dataResult.append(tile)
        indexResult = precomputedTileIndices

    return indexResult,dataResult

def getImageMean_withoutPureBlack(image):
    black = np.array([0,0,0])
    cleared = np.where(image==black,np.NaN,image)
    result = np.nanmean(cleared)
    if np.isnan(result):
        result = 0.0 # pure black
    return result

def getImageMean_withoutBlack(image,blackThreshold):
    #print("image shape is {0}".format(image.shape))
    imageBrightness = np.nanmax(image, axis=-1)

    blackMap = imageBrightness < blackThreshold
    blackMap = np.expand_dims(blackMap,axis=-1)
    blackMap = np.tile(blackMap, (1,1,3))
    if not(np.any(blackMap)):
        # nonNan exists
        cleared = np.where(blackMap,np.NaN,image)
        return np.nanmean(cleared)
    else:
        return np.NaN # pure black
    
def getImageContrast_withoutPureBlack(image, regTerm=0.0, precomputedMu = None):
    """(0,0,0) pixels are excluded from contrast evaluation"""
    if precomputedMu is None:
        mu = getImageMean_withoutPureBlack(image)
    else:
        mu = precomputedMu
    diff = image - mu
    return getImageContrast(image,regTerm=regTerm, precomputedDiff=diff)

def getImageContrast(image, regTerm=0.0, precomputedDiff = None):
    """Entire image contrast as defined in Goodfellow et al. 2016 "Deep learning" p.442"""
    if precomputedDiff is None:
        mu = getImageMean_withoutPureBlack(image)
        diff = image - mu
    else:
        diff = precomputedDiff
    
    squaredDiff = diff*diff
    meanSqDiff = np.mean(squaredDiff)
    return math.sqrt(regTerm + meanSqDiff)

def GCN(image,lambdaTerm = 0.0, eps=1e-8, precomputedContrast = None, precomputedMean = None):
    """Global Contrast Normalization as defined in Goodfellow et al. 2016 "Deep learning" p.442"""
    if precomputedMean is None:
        mu = np.mean(image)
    else:
        mu = precomputedMean
    
    diff = image - mu
    if precomputedContrast is None:
        initialContrast = getImageContrast(image, lambdaTerm, diff)
    else:
        initialContrast = precomputedContrast
    return diff / max(eps, initialContrast)

def GCNtoRGB_uint8(gcnImage, cutoffSigmasRange = 3.0):
    """gcnImage is considered to have 0 mean and stddev == 1.0"""
    rescaled = (gcnImage + cutoffSigmasRange)/(2.0 * cutoffSigmasRange) * 255.0
    return np.round(np.minimum(np.maximum(rescaled,0.0),255.0)).astype(np.uint8)

def getTiles(filename, rotDegree):
    t1 = time.time()
    #im = 255 - io.imread(filename,plugin="tifffile")
    multiimage = io.MultiImage(filename)
    #print("multiimage of {0} elements".format(len(multiimage)))
    #for i in range(0,len(multiimage)):
    #    print("level {0}, shape {1}".format(i,multiimage[i].shape))
    im = 255 - multiimage[1]
    t2 = time.time()
    print("tiff decoding took {0:.3f} sec".format(t2-t1))
    t1 = t2
    h,w,_ = im.shape
    #im = cv2.resize(im, dsize=(w // initial_downscale_factor, h // initial_downscale_factor), interpolation=cv2.INTER_AREA)
    tileSize = tileSizeSetting // 4

    if rotDegree != 0.0:
        rotated = RotateWithoutCrop(im, rotDegree)
    else:
        rotated = im
    _,tiles = getNotEmptyTiles(rotated, tileSize, emptyCuttOffQuantile=None, emptyCutOffMaxThreshold=10)

    t2 = time.time()
    print("non empty tile extraction took {0:.3f} sec".format(t2-t1))
    t1 = t2
    

    if len(tiles) > trainSequenceLength:
        tiles = tiles[0:trainSequenceLength]

    if len(tiles) == 0:
        print("WARN: Image {0} resulted in 0 tiles. producing blank (black) single tile TfRecords file".format(ident))
        tiles = [ np.zeros((outImageSize,outImageSize,3),dtype=np.uint8) ]
    else:
        # filtering out non green
        nonRedTiles = []
        for tile in tiles:
            red = np.mean(tile[:,:,0]) # supposing RGB, not BGR
            green = np.mean(tile[:,:,1])
            #print("[R:{0}\tG:{1}; ratio {2}".format(red,green, green/red))
            if green / red >= 1.2: # green must be at least 1.5 times more than red (to remove white marker tiles)
                nonRedTiles.append(tile)
        if len(nonRedTiles)>0:
            tiles = nonRedTiles
        else:
            print("WARN: omited non-green tiles filtering, as the filtering impose the empty tile set")

        t2 = time.time()
        print("red filtering took {0:.3f} sec".format(t2-t1))
        t1 = t2
    

        # normalizing with contrasts
        contrasts = []
        means = []
        #print("normalzing {0} tiles".format(len(tiles)))
        for tile in tiles:        
            mu = getImageMean_withoutBlack(tile, 40.0)
            if not np.isnan(mu):
                contrast = getImageContrast_withoutPureBlack(tile, precomputedMu=mu)
                means.append(mu)
                contrasts.append(contrast)
        if len(means)>0: # whole image is not entirly black
            meanContrast = np.mean(contrasts)    
            meanMean = np.mean(means)
            if meanMean > 0.0:
                for j in range(0,len(tiles)):
                    tiles[j] = GCNtoRGB_uint8(GCN(tiles[j], lambdaTerm=0.0, precomputedContrast=meanContrast, precomputedMean=meanMean), cutoffSigmasRange=2.0)
    
    t2 = time.time()
    print("GCN took {0:.3f} sec".format(t2-t1))
    t1 = t2
    

    if outImageSize != tileSize:
        resizedTiles = []
        for tile in tiles:
            resizedTiles.append(cv2.resize(tile, dsize=(outImageSize, outImageSize), interpolation=cv2.INTER_AREA))
        tiles = resizedTiles

    t2 = time.time()
    print("resize took {0:.3f} sec".format(t2-t1))
    t1 = t2
    
    return tiles

def constructModel(seriesLen, DORate=0.2, l2regAlpha = 1e-3):
    imageSize = outImageSize

    netInput = tf.keras.Input(shape=(seriesLen, imageSize, imageSize, 3), name="input")
    denseNet = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(imageSize, imageSize, 3),
        # it should have exactly 3 inputs channels,
        # and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value
        pooling=None)  # Tx8x8x1024 in case of None pooling
    
    converted = tf.keras.applications.densenet.preprocess_input(netInput)
    print("converted input shape {0}".format(converted.shape))
    cnnOut = tf.keras.layers.TimeDistributed(denseNet, name='cnns')(converted)  # Tx7x7x1024 in case of None pooling
    print("cnn out shape {0}".format(cnnOut.shape))
    cnnPooled = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((7, 7)), name='cnnsPooled')(
        cnnOut)  # Tx1x1x1024
    cnnPooledReshaped = tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((1024,)), name='cnnsPooledReshaped')(
        cnnPooled)  # Tx1024
    cnnPooledReshapedDO = tf.keras.layers.Dropout(rate=DORate, name='cnnsPooledReshapedDO')(
        cnnPooledReshaped)  # Tx1024
    perSliceDenseOut = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(192,
        activation="selu",
        kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha)), name='perSliceDenseOut')(
        cnnPooledReshapedDO)  # 128.   1024*128  parameters
    perSliceDenseOutDO = tf.keras.layers.Dropout(rate=DORate, name='perSliceDenseOutDO')(
        perSliceDenseOut)
    perSliceDenseOut2 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(96,
        activation="selu",
        kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha)), name='perSliceDenseOut2')(
        perSliceDenseOutDO)  # 128.   1024*128  parameters
    perSliceDenseOutDO2 = tf.keras.layers.Dropout(rate=DORate, name='perSliceDenseOutDO2')(
        perSliceDenseOut2)
    #gru1 = tf.keras.layers.GRU(128, return_sequences=True)
    #gru1back = tf.keras.layers.GRU(128, return_sequences=True, go_backwards=True)
    #gru1out = tf.keras.layers.Bidirectional(gru1, backward_layer=gru1back, name='rnn1')(perSliceDenseOutDO)
    #gru1outDO = tf.keras.layers.Dropout(rate=DORate, name='rnn1DO')(gru1out)

    #, batch_input_shape=(1, seriesLen, 128)
    # , implementation=1

    rnnOut = \
        tf.keras.layers.GRU(
            64, dropout=DORate,
            kernel_regularizer = tf.keras.regularizers.L1L2(l2=l2regAlpha),
            recurrent_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha),
            return_sequences=True)(perSliceDenseOutDO2)
    rnnOutDO = tf.keras.layers.Dropout(rate=DORate,name='rnnDO')(rnnOut)
    rnnOut2 = \
        tf.keras.layers.GRU(
            48, dropout=DORate,
            kernel_regularizer = tf.keras.regularizers.L1L2(l2=l2regAlpha),
            recurrent_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha))(rnnOutDO)
    rnnOutDO2 = tf.keras.layers.Dropout(rate=DORate,name='rnn2DO')(rnnOut2)
    # predOut = \
    #     tf.keras.layers.Dense(6,name="resLogits",activation="linear",
    #     kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha)
    #     )(rnnOutDO)
    predOut = \
        tf.keras.layers.Dense(1,name="unitRes",activation="sigmoid",
        kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha)
        )(rnnOutDO2)
    predOutScaled = \
        tf.keras.layers.Lambda(lambda x: x*5.0, name="scaledRes")(predOut)


    return tf.keras.Model(name="PANDA_A", inputs=netInput, outputs=predOutScaled), denseNet

def coerceSeqSize(imagePack, trainSequenceLength):
    imagePackShape = tf.shape(imagePack)
    outputShape = [
        trainSequenceLength,
        imagePackShape[1],
        imagePackShape[2],
        imagePackShape[3]
    ]
    T = imagePackShape[0]

    availableIndices = tf.range(T)


    # if T is less than trainSequenceLength we need to duplicate the layers
    seqRepCount = tf.cast(tf.math.ceil(trainSequenceLength / T), tf.int32)
    notTooShortIndicies = \
    tf.cond(seqRepCount > 1, \
        lambda : tf.tile(availableIndices, [seqRepCount]), \
        lambda : availableIndices)

    # if T is greater than trainSequenceLength we need to truncate it
    notTooLongIndices = notTooShortIndicies[0:trainSequenceLength]
    #notTooLong = tf.IndexedSlices(imagePack,notTooLongIndices, dense_shape = outputShape)
    notTooLong = tf.gather(imagePack, notTooLongIndices)
    shapeSet = tf.reshape(notTooLong,outputShape)
    return shapeSet

def augment(imagePack):
    def augmentSingle(image):
        augSwitches = tf.cast(tf.math.round(tf.random.uniform([3],minval=0.0, maxval=1.0)),dtype=tf.bool)
        image = tf.cond(augSwitches[0], lambda: tf.image.rot90(image), lambda: image)
        image = tf.cond(augSwitches[1], lambda: tf.image.flip_left_right(image), lambda: image)
        image = tf.cond(augSwitches[2], lambda: tf.image.flip_up_down(image), lambda:image)
        return image
    return tf.map_fn(augmentSingle, imagePack)


filenameDs = tf.data.Dataset.from_tensor_slices(fullPaths)

model,backbone = constructModel(trainSequenceLength, DORate=0.3, l2regAlpha = 0.0)
print("model constructed")
if backboneFrozen:
    backbone.trainable = False
else:
    backbone.trainable = True


def predict(checkpointPath, rotDegree):
    if os.path.exists(checkpointPath):
        print("Loading pretrained weights {0}".format(checkpointPath))
        model.load_weights(checkpointPath, by_name=True)
        print("Loaded pretrained weights {0}".format(checkpointPath))
    else:
        print("Pretrained weights file does not exist: {0}".format(checkpointPath))
        exit(1)

    def loadAsTilePackTf(pathTensor):
        def loadAsTilePack(path):
            path = path.numpy().decode("utf-8")
            tiles = getTiles(path, rotDegree)
            return len(tiles),tf.cast(tf.stack(tiles),tf.uint8)
        tensorLen,tensorPack = tf.py_function(loadAsTilePack, [pathTensor], Tout=(tf.int32,tf.uint8))
        return tf.reshape(tensorPack,[tensorLen, outImageSize, outImageSize, 3])


    def process(filename):
        return augment(coerceSeqSize(loadAsTilePackTf(filename),trainSequenceLength))

    #print("filenameDs:")
    #print(filenameDs)

    #tf.data.experimental.AUTOTUNE

    imagesDs = filenameDs \
        .map(process, num_parallel_calls=1) \
        .batch(batchSize, drop_remainder=False) \
        .prefetch(prefetchSize)

    t1 = time.time()
    predicted = np.round(np.squeeze(model.predict(imagesDs,verbose=1))).astype(np.int32)
    t2 = time.time()
    print("predicted shape is {0}".format(predicted.shape))
    print("predicted {0} samples in {1} sec. {2} sec per sample".format(imageCount,t2-t1,(t2-t1)/imageCount))
    return predicted

predicted = predict(checkpoint_path, 0.0)

predictedList = predicted.tolist()
resDf = pd.DataFrame.from_dict({'image_id':inputIdents , 'isup_grade': predictedList})
resDf.sort_values(by=['image_id'], inplace=True, ascending=True)
resDf.to_csv(out_file, index=False)

print('Done')