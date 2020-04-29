import math
import numpy as np

def getNotEmptyTiles(image, tileSize, emptySpaceValue = 255):
    '''Returns the list of non-empty tile indeces (tile_row_idx,tile_col_idx) and corresponding list of tile npArrays).
    Each index list element specifies the zero based index (row_idx,col_idx) of square tile which contain some data (non-empty)'''
    h,w,_ = image.shape
    vertTiles = math.ceil(h / tileSize)
    horTiles = math.ceil(w / tileSize)
    indexResult = []
    dataResult = []
    for row_idx in range(0,vertTiles):
        for col_idx in range(0,horTiles):
            tileYstart = row_idx*tileSize
            tileYend = min((row_idx+1)*tileSize,h)
            tileXstart = col_idx*tileSize
            tileXend = min((col_idx+1)*tileSize,w)
            tile = image[
                tileYstart:tileYend,
                tileXstart:tileXend,
                :] # all 3 color channels
            
            # if the tile contains pure white pixels (no information) we skip it (do not return)
            tileMin = np.nanmin(tile)
            if tileMin == 255:
                continue

            # we may need to pad the edge tile to full requested tile size
            xPad = tileSize - (tileXend-tileXstart)
            yPad = tileSize - (tileYend-tileYstart)
            if (xPad>0) | (yPad>0):
                # we pad to the end
                tile = np.pad(tile,[[0,yPad],[0,xPad],[0,0]],constant_values = 255)
            indexResult.append((row_idx,col_idx))
            dataResult.append(tile)

    return indexResult,dataResult