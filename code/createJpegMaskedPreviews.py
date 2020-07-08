import sys
import os
import multiprocessing
import numpy as np
import pandas as pd
from skimage import io
import cv2
from npImageTransformation import TrimBlackPaddings, GetNonBlackArea

max_out_side = int(sys.argv[6])

def ProcessTask(task):
    ident = task['ident']
    tiffPath = task['tiffPath']
    maskPath = task['maskPath']
    jpegPath = task['jpegPath']
    jpegMaskedPath = task['jpegMaskedPath']
    isKarolinska = task['isKarolinska']
    
    im = 255 - io.imread(tiffPath,plugin="tifffile")

    nonBlackArea = GetNonBlackArea(im)
    im = TrimBlackPaddings(im, precomputedBounds=nonBlackArea)

    h,w,_ = im.shape
    max_init_size = max(h,w)
    
    if max_init_size > max_out_side:
        scale_factor = max_out_side / max_init_size
        im = cv2.resize(im, dsize=(int(round(w * scale_factor)), int(round(h * scale_factor))), interpolation=cv2.INTER_NEAREST)
    
    io.imsave(jpegPath,im)

    mask = io.imread(maskPath, plugin="tifffile")
    mask = TrimBlackPaddings(mask, precomputedBounds=nonBlackArea)[:,:,0]

    if max_init_size > max_out_side:
        scale_factor = max_out_side / max_init_size    
        mask = np.round(cv2.resize(mask, dsize=(int(round(w * scale_factor)), int(round(h * scale_factor))), interpolation=cv2.INTER_NEAREST)).astype(dtype=np.uint8)
    
    #print("Mask shape is {0}".format(mask.shape))

    # Radboud: Prostate glands are individually labelled. Valid values are:
    # 0: background (non tissue) or unknown
    # 1: stroma (connective tissue, non-epithelium tissue)
    # 2: healthy (benign) epithelium
    # 3: cancerous epithelium (Gleason 3)
    # 4: cancerous epithelium (Gleason 4)
    # 5: cancerous epithelium (Gleason 5)

    # Karolinska: Regions are labelled. Valid values are:
    # 0: background (non tissue) or unknown
    # 1: benign tissue (stroma and epithelium combined)
    # 2: cancerous tissue (stroma and epithelium combined)

    backgroundIndicator = mask == 0
    if isKarolinska:
        benignIndicator = (mask == 1) | (mask == 2)
        malignantIndicator = (mask == 3) | (mask == 4) | (mask == 5)
    else:
        benignIndicator = mask == 1
        malignantIndicator = mask == 2
    mask = None
    backgoundCount = float(np.sum(backgroundIndicator))
    backgroundIndicator = None

    h,w,_ = im.shape
    
    highlightMagnitude = 200

    benignHighlight = benignIndicator.astype(np.int16) * highlightMagnitude
    im[:,:,2] = np.clip(im[:,:,2].astype(np.int16) + benignHighlight,0,255).astype(np.uint8) # blue channel
    benignHighlight = None

    malignantHighlight = malignantIndicator.astype(np.int16) * highlightMagnitude
    im[:,:,0] = np.clip(im[:,:,0].astype(np.int16) + malignantHighlight,0,255).astype(np.uint8) # red channel
    malignantHighlight = None

    totalCount = float(h*w)
    
    benignCount = float(np.sum(benignIndicator))
    malignantCount = float(np.sum(malignantIndicator))

    
    io.imsave(jpegMaskedPath,im)
    sys.stdout.write(".")
    sys.stdout.flush()
    return (ident, backgoundCount / totalCount, benignCount / totalCount, malignantCount / totalCount)


if __name__ == '__main__':
    imagesPath = sys.argv[1]
    labelsPath = sys.argv[2]
    maskPath = sys.argv[3]
    outDirPath = sys.argv[4]
    outStatsPath = sys.argv[5]

    M = multiprocessing.cpu_count()
    
    labelsDf = pd.read_csv(labelsPath, engine='python')
    print("loaded labels")
    isKarolinskaMap = dict()
    isupMap = dict()
    gleasonMap = dict()
    for row in labelsDf.itertuples():
        isKarolinskaMap[row.image_id] = (row.data_provider == "karolinska")
        isupMap[row.image_id] = row.isup_grade
        gleasonMap[row.image_id] = row.gleason_score
    print("Provider map is built")

    print("Detected {0} CPU cores".format(M))
    M //= 2

    p = multiprocessing.Pool(M)
    print("Created process pool of {0} workers".format(M))

    files = os.listdir(imagesPath)
    tiffFiles = [x for x in files if x.endswith(".tiff")]

    # uncomment for short (test) run
    #tiffFiles = tiffFiles[0:20]

    tasks = list()
    existsList = list()

    for tiffFile in tiffFiles:
        ident = tiffFile[:-5]
        jpegPath = os.path.join(outDirPath,"{0}.jpeg".format(ident))
        jpegMaskedPath = os.path.join(outDirPath,"{0}M.jpeg".format(ident))
        if(os.path.exists(jpegPath) & os.path.exists(jpegMaskedPath)):
            existsList.append(jpegPath)
        else:
            task = {
                'ident' : ident,
                'isKarolinska': isKarolinskaMap[ident],
                'tiffPath': os.path.join(imagesPath,tiffFile),
                'maskPath': os.path.join(maskPath, "{0}_mask.tiff".format(ident)),
                'jpegPath': jpegPath,
                'jpegMaskedPath': jpegMaskedPath
            }
            if os.path.exists(task['maskPath']):
                tasks.append(task)
    print('{0} previews are already generated. {1} to generate'.format(len(existsList),len(tasks)))
    
    #gathered = p.map(ProcessTask,tasks)
    gathered = [ProcessTask(x) for x in tasks] # single process processing for debugging

    print("Dumping stats")

    idents = list()
    bgPortions = list()
    benPortions = list()
    malPortions = list()
    isups = list()
    gleasons = list()
    for res in gathered:
        ident,bg,ben,mal = res
        idents.append(ident)
        isups.append(isupMap[ident])
        gleasons.append(gleasonMap[ident])
        bgPortions.append(bg)
        benPortions.append(ben)
        malPortions.append(mal)

    resDf = pd.DataFrame.from_dict(
    {
        'image_id': idents,
        'gleason_score': gleasons,
        'isup_grade': isups,
        'backgroundPortion': bgPortions,
        'benignPortion': benPortions,
        'malignantPortion':malPortions
    })
    resDf.to_csv(outStatsPath, index=False)
    print("Done")    
