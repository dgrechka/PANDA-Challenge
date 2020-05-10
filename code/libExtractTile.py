import math
import numpy as np

def getNotEmptyTiles(image, tileSize, emptySpaceValue = 255, precomputedTileIndices=None):
    '''Returns the list of non-empty tile indeces (tile_row_idx,tile_col_idx) and corresponding list of tile npArrays).
    Each index list element specifies the zero based index (row_idx,col_idx) of square tile which contain some data (non-empty)'''
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
            tile = np.pad(tile,[[0,yPad],[0,xPad],[0,0]],constant_values = 255)
        return tile

    if precomputedTileIndices == None:
        # we analyze all tiles for the content
        for row_idx in range(0,vertTiles):
            for col_idx in range(0,horTiles):
                tile = extractTileData(row_idx, col_idx)
                # if the tile contains pure white pixels (no information) we skip it (do not return)
                tileMin = np.nanmin(tile)
                tileQuantile = np.quantile(tile,1/4)
                #print("row {0} col {1} min_v {2}".format(row_idx,col_idx,tileMin))
                if tileMin > 230: # too white! there is no dark areas
                    continue
                if tileQuantile > 240: # too many white pixels
                    continue

                tile = coerceTileSize(tile)

                indexResult.append((row_idx,col_idx))
                dataResult.append(tile)
                tileIntensity.append(tileQuantile)
        
        # sorting the tiles according to intensity
        resIdx = []
        resData = []
        for (idxElem,dataElem,_) in sorted(zip(indexResult, dataResult, tileIntensity), key=lambda pair: pair[2]):
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