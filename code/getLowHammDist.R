#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

threshold <- as.integer(args[3])
print(paste0('Loading ',args[1]))
df1 <- read.csv(args[1])
print(paste0('Leaving hamming distances <= ',threshold))

df2 <- df1[df1$hammingDist <= threshold ,]
print('Sorting')
df2 <- df2[order(df2$hammingDist,df2$aspectDist),]
print(paste0('Wrting results to ',args[2]))
write.csv(df2, file=args[2], row.names = F)
print('Done')
