import sys
import pandas as pd

trainLabelsPath = sys.argv[1]
closeImagesPath = sys.argv[2]
outputPath = sys.argv[3]

df1 = pd.read_csv(trainLabelsPath)
print("Loaded {0} total samples".format(len(df1)))

df2 = pd.read_csv(closeImagesPath)
print("Loaded {0} close image pairs".format(len(df2)))

clusterToNodes = dict()
nodeToCluster = dict()

def GetClusterByImage(imageId):
    if imageId in nodeToCluster:
        return nodeToCluster[imageId]
    else:
        return imageId # image itself is cluster

def GetClusterImages(clusterId):
    if clusterId in clusterToNodes:
        return clusterToNodes[clusterId]
    else:
        return [clusterId]

def MergeClusters(imageId1,imageId2):
    cluster1 = GetClusterByImage(imageId1)
    cluster2 = GetClusterByImage(imageId2)

    # merging cluster 2 into cluster 1
    cluster2Images = GetClusterImages(cluster2)
    if cluster1 in clusterToNodes:
        # non empty cluster 1
        cluster1Nodes = clusterToNodes[cluster1]
    else:
        cluster1Nodes = [cluster1]
    #print("merging {0} images into cluster of {1} images".format(len(cluster2Images),len(cluster1Nodes)))
    cluster1Nodes.extend(cluster2Images)
    for img in cluster2Images:
        nodeToCluster[img] = cluster1


print("clustering...")
for pair in df2.itertuples():
    node1 = pair.ident1
    node2 = pair.ident2
    hd = pair.hammingDist
    ad = pair.aspectDist
    merge = False
    if ((hd <= 130) & (ad <= 1e-3)):
        merge = True
    if ((hd <= 100) & (ad <= 0.2)):
        merge = True
    if ((hd <= 90) & (ad <= 1.0)):
        merge = True
    if merge:
        MergeClusters(node1,node2)

print("Generating result dataframe with cluster ID")
#image_id,data_provider,isup_grade,gleason_score
image_id = list()
data_provider = list()
isup_grade = list()
gleason_score = list()
image_cluster_id = list()
for row in df1.itertuples():
    image_id.append(row.image_id)
    data_provider.append(row.data_provider)
    isup_grade.append(row.isup_grade)
    gleason_score.append(row.gleason_score)
    image_cluster_id.append(GetClusterByImage(row.image_id))

cluster_id_set = set(image_cluster_id)
print("Got {0} clusters".format(len(cluster_id_set)))


resDf = pd.DataFrame.from_dict(
    {
        'image_id': image_id,
        'image_cluster_id': image_cluster_id,
        'data_provider': data_provider,
        'isup_grade':isup_grade,
        'gleason_score':gleason_score
    }
)
resDf.sort_values(by=['image_cluster_id'], inplace=True)
resDf.to_csv(outputPath, index=False)
print("Done")