#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

inputFile <- args[1]
outputFile <- args[2]
print(paste0('Loading ',inputFile))

df1 <- read.csv(inputFile, stringsAsFactors = T)

isInResult <- c()
knownClusters <- c()

for(i in (1:nrow(df1)))
{
  clusterID <- df1[i,"image_cluster_id"]
  if(clusterID %in% knownClusters) {
    isInResult <- append(isInResult, F)
  } else {
    knownClusters <- union(knownClusters, clusterID)
    isInResult <- append(isInResult, T)
  }
}

print(paste0(nrow(df1)," rows initially in the table"))
df2 <- df1[isInResult,]
print(paste0(nrow(df2), " rows are left"))

write.csv(df2, file= outputFile, row.names = F, quote = F)