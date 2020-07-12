#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

res1File <- args[0]
res2File <- args[2]
outFile <- args[3]

df1 <- read.csv(res1File)
df2 <- read.csv(res2File)

threshold <- 2.5

df1hard <- df1[df1$isup_abs_diff >= threshold,]
df2hard <- df2[df2$isup_abs_diff >= threshold,]

print(paste0("selecting samples with abs error >= ",threshold))

print(paste0(res1File," has ",nrow(df1hard)," hard samples out of ",nrow(df1)))
print(paste0(res2File," has ",nrow(df2hard)," hard samples out of ",nrow(df2)))

commonHard <- intersect(df1hard$file, df2hard$file)

df1hardCommon <- df1hard[df1hard$file %in% commonHard,]
df2hardCommon <- df2hard[df2hard$file %in% commonHard,]

hardMerged <- merge(df1hardCommon, df2hardCommon,
                    by=c("file","isup_grade_truth","provider","gleason_score"),
                    suffixes = c(".fold1",".fold3"))
hardMerged$isup_abs_diff_sum <- hardMerged$isup_abs_diff.fold1 + hardMerged$isup_abs_diff.fold3
hardMerged <- hardMerged[order(hardMerged$isup_abs_diff_sum, decreasing = T),]

head(hardMerged)

write.csv(hardMerged, file=outFile, row.names = F)