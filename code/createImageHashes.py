from skimage import io
from PIL import Image
import sys
import os
import imagehash
import multiprocessing
import pandas as pd
import cv2
from npImageTransformation import TrimBlackPaddings

def ProcessTask(task):    
    ident = task['ident']
    tiffPath = task['tiffPath']
    hashSize = 16 # default is 8
    initial_downscale_factor = 4

    imNumpy = io.imread(tiffPath,plugin="tifffile")
    h,w,_ = imNumpy.shape
    imNumpy = cv2.resize(imNumpy, dsize=(w // initial_downscale_factor, h // initial_downscale_factor), interpolation=cv2.INTER_AREA)
    imNumpy = TrimBlackPaddings(imNumpy)
    h2,w2,_ = imNumpy.shape
    im = Image.fromarray(imNumpy,"RGB")

    a_hash = imagehash.average_hash(im, hash_size=hashSize)
    p_hash = imagehash.phash(im, hash_size=hashSize )
    d_hash = imagehash.dhash(im, hash_size=hashSize)
    w_hash = imagehash.whash(im, hash_size=hashSize)
    #w2_hash = imagehash.whash(im, mode='db4')
    print("{0}:\t{1}-{2}-{3}-{4}-{5}-{6}".format(ident,a_hash,p_hash,d_hash,w_hash,h2,w2))
    return (ident,a_hash,p_hash,d_hash,w_hash,h2,w2)


if __name__ == '__main__':

    imagesPath = sys.argv[1]
    outPath = sys.argv[2]

    print("Looking for images in {0}".format(imagesPath))
    print("Output table will be written to {0}".format(outPath))

    files = os.listdir(imagesPath)
    tiffFiles = [x for x in files if x.endswith(".tiff")]

    # uncomment for short (test) run
    # tiffFiles = tiffFiles[0:20]

    M = multiprocessing.cpu_count()
    
    print("Detected {0} CPU cores".format(M))
    #M //= 2 # opencv uses multithreading somehow. So we use less workers that CPU cores available
    M = 4

    p = multiprocessing.Pool(M)
    print("Created process pool of {0} workers".format(M))

    tasks = list()
    for tiffFile in tiffFiles:
        task = {
            'ident' : tiffFile[:-5],
            'tiffPath': os.path.join(imagesPath,tiffFile)
        }
        tasks.append(task)
    print("Starting {0} conversion tasks".format(len(tasks)))

    hashes = p.map(ProcessTask,tasks)

    print("reshaping")
    idents = list()
    a_hashes = list()
    p_hashes = list()
    d_hashes = list()
    w_hashes = list()
    heights = list()
    widths = list()
    #w2_hashes = list()
    for entry in hashes:
        ident,a,p,d,w,height,width = entry
        idents.append(ident)
        a_hashes.append(str(a))
        p_hashes.append(str(p))
        d_hashes.append(str(d))
        w_hashes.append(str(w))
        heights.append(int(height))
        widths.append(int(width))
    #    w2_hashes.append(str(w2))

    print("Saving as CSV")
    df = pd.DataFrame.from_dict(
        {
            'ident':idents,
            'Height': heights,
            'Width': widths,
            'Average': a_hashes,
            'Perceptual': p_hashes,
            'Difference': d_hashes,
            'HaarWavelet' : w_hashes,
            #'DaubechiesWavelet': w2_hashes
        }
    )
    df.to_csv(outPath, index=False)
    print("Done")
