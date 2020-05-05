args <- commandArgs(trailingOnly = TRUE)

dfPath <- args[1]
outDir <- args[2]

K <- 5

set.seed(4451)

df1 <- read.csv(dfPath)
require(caret)
folds <- createFolds(df1$gleason_score,k=K)
for(i in 1:K) {
  fold <- folds[[i]] - 1 # zero based
  outPath <- file.path(outDir,paste0("fold_",i,"_val_rows.csv"))
  df2 <- data.frame(rowIdx=fold)
  write.csv(df2,file=outPath,row.names = F)
}