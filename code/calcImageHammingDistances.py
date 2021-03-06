import imagehash
import pandas as pd
import numpy as np
import os
import sys
import math

hashesPath = sys.argv[1]
outPath = sys.argv[2]

df1 = pd.read_csv(hashesPath)

def hashArr(hashes):
    acc = None
    for hash in hashes:
        ar = imagehash.hex_to_hash(hash).hash
        if acc is None:
            acc = ar
        else:
            acc = np.concatenate((acc,ar))
    return acc



idents = list()
fullHashes = list()
widths = list()
heights = list()

print("Building full hashes")
for row in df1.itertuples():
    ident = row.ident
    aHash = row.Average
    pHash = row.Perceptual
    dHash = row.Difference
    wHash = row.HaarWavelet
    hashes = [aHash, pHash, dHash, wHash]
    fullHashes.append(hashArr(hashes))
    widths.append(int(row.Width))
    heights.append(int(row.Height))
    idents.append(ident)

def getAspectDiff(h1,w1,h2,w2):
    a1 = h1/w1
    a2 = h2/w2

    def NonLinear(v):
        sign = 1.0
        if v <= 1.0:
            v = 1.0 / v
            sign = -1.0
        return (math.exp(v - 1) - 1)  * sign
    return abs(NonLinear(a1) - NonLinear(a2))
    

#debuging code
#idents = idents[0:100]
#fullHashes = fullHashes[0:100]

N = len(idents)

idents1 = list()
idents2 = list()
hammingDists = list()
aspectDists = list()

print("Calculating distances")
for idx1 in range(0, N-1):
    hash1 = fullHashes[idx1]
    ident1 = idents[idx1]
    w1 = widths[idx1]
    h1 = heights[idx1]
    #print("{0}".format(ident1))
    for idx2 in range(idx1+1, N):
        ident2 = idents[idx2]
        hash2 = fullHashes[idx2]
        w2 = widths[idx2]
        h2 = heights[idx2]
        hamDist = np.sum(hash1 ^ hash2)
        hammingDists.append(hamDist)
        aspectDists.append(getAspectDiff(h1,w1,h2,w2))
        idents1.append(ident1)
        idents2.append(ident2)
        
print("Saving as CSV")
df2 = pd.DataFrame.from_dict(
    {
        'ident1': idents1,
        'ident2': idents2,
        'hammingDist': hammingDists,
        'aspectDist': aspectDists
    })
df2.to_csv(outPath, index=False)
print("Done")