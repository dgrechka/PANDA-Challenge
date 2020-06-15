import sys
import os
import multiprocessing
import numpy as np
from skimage import io
import cv2
from npImageTransformation import TrimBlackPaddings

max_out_side = int(sys.argv[3])

def GeneratePreview(task):
    tiffPath = task['tiffPath']
    jpegPath = task['jpegPath']
    
    im = 255 - io.imread(tiffPath,plugin="tifffile")

    im = TrimBlackPaddings(im)

    h,w,_ = im.shape
    max_init_size = max(h,w)
    
    if max_init_size > max_out_side:
        scale_factor = max_out_side / max_init_size
        im = cv2.resize(im, dsize=(int(round(w * scale_factor)), int(round(h * scale_factor))), interpolation=cv2.INTER_AREA)
    io.imsave(jpegPath,im)
    sys.stdout.write(".")
    sys.stdout.flush()
    return 1


if __name__ == '__main__':
    imagesPath = sys.argv[1]
    outPath = sys.argv[2]

    M = multiprocessing.cpu_count()
    
    print("Detected {0} CPU cores".format(M))
    M //= 2

    p = multiprocessing.Pool(M)
    print("Created process pool of {0} workers".format(M))

    files = os.listdir(imagesPath)
    tiffFiles = [x for x in files if x.endswith(".tiff")]

    # uncomment for short (test) run
    # tiffFiles = tiffFiles[0:20]

    tasks = list()
    existsList = list()

    for tiffFile in tiffFiles:
        jpegPath = os.path.join(outPath,"{0}.jpeg".format(tiffFile[:-5]))
        if(os.path.exists(jpegPath)):
            existsList.append(jpegPath)
        else:
            task = {
                'ident' : tiffFile[:-5],
                'tiffPath': os.path.join(imagesPath,tiffFile),
                'jpegPath': jpegPath
            }
            tasks.append(task)
    print('{0} previews are already generated. {1} to generate'.format(len(existsList),len(tasks)))
    gathered = p.map(GeneratePreview,tasks)
    print("Done")
    
