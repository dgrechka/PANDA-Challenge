from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
if __name__ == "__main__":
    from libExtractTile import getNotEmptyTiles
else:
    from .libExtractTile import getNotEmptyTiles
import unittest
import math


class TestExtractTile(unittest.TestCase):
    def setUp(self):
        datasetPath = "data\\kaggleOfficial"
        citoImagesPath = os.path.join(datasetPath,"train_images")        
        self.toOpen = os.path.join(citoImagesPath,"00a7fb880dc12c5de82df39b30533da9.tiff")
        #print("file exists {0}".format(os.path.exists(self.toOpen)))
        #print("Attempting to open {0}".format(self.toOpen))
        self.im = io.imread(self.toOpen)
        self.shape = self.im.shape
        #print("Read. shape {0}".format(self.im.shape))

    def test_idx_list_of_tiles_returned(self):
        h,w,_ = self.shape
        tileSize = 1024
        idxList,_ = getNotEmptyTiles(self.im,tileSize)
        possibleTiles = math.ceil(h/tileSize) * math.ceil(w/tileSize)
        print("Got {0} non empty tiles out of {1} possible.".format(len(idxList),possibleTiles))
        self.assertLess(len(idxList),possibleTiles)

    def test_all_tiles_have_same_size(self):
        tileSize = 1024
        _,tiles = getNotEmptyTiles(self.im,tileSize)
        for tile in tiles:
            th,tw,tc = tile.shape
            self.assertEqual(th,tileSize,"height not equal to requested size")
            self.assertEqual(tw,tileSize,"width not equal to requested size")
            self.assertEqual(tc,3, "number of color channels not equal to requested size")



if __name__ == "__main__":
    print("test app")
    testCase = TestExtractTile()
    testCase.setUp()
    
    tileSize = 1024

    tileIdx,tiles = getNotEmptyTiles(testCase.im,tileSize)
    N = len(tileIdx)
    cols = 20
    rows = math.ceil(N/cols)

    plt.figure()
    plt.imshow(testCase.im)

    fig, ax = plt.subplots(rows,cols)    
    fig.set_facecolor((0.3,0.3,0.3))

    idx = 1
    for row in range(0,rows):
        for col in range(0,cols):            
            row = (idx - 1) // cols
            col = (idx -1) % cols
            #ax[row,col].set_title("tile [{0},{1}]".format(tile_r,tile_c))    
            
            ax[row,col].axis('off')
            if idx-1 < N:
                ax[row,col].imshow(tiles[idx-1]) 
            idx = idx + 1
    plt.show()  # display it

    #im.show()