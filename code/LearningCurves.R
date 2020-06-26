require(ggplot2)
require(stringr)
require(tidyr)
require(gridExtra)

args <- commandArgs(trailingOnly = TRUE)

print(args)

inputPattern <- args[1]
outputFile <- args[2]

print(paste0("Input pattern is ",inputPattern))
inputPattern2 <- substr(inputPattern,0,nchar(inputPattern)-1)
print(paste0("Input pattern2 is ",inputPattern2))
inputMatches <- Sys.glob(inputPattern2,dirmark=T)
inputMatches <- inputMatches[endsWith(inputMatches,'/')]

print(paste0(length(inputMatches)," input pattern matches"))
print(inputMatches)

resDf <- NULL
for(inputMatch in inputMatches)  {
  slashes <- str_locate_all(inputMatch,'/')
  foldName <- substr(
    inputMatch,
    slashes[[1]][nrow(slashes[[1]]),1] + 1,
    nchar(inputMatch)-1
  )
  print(paste0("Fold name is ",foldName))
  curDf <- read.csv(file.path(inputMatch,'training_log.csv'))
  curDf$Fold <- foldName
  if(is.null(resDf))
    resDf <- curDf
  else
    resDf <- rbind(resDf,curDf)
}

#preparing for figures
resDf$Fold <- as.factor(resDf$Fold)

# gathering loss columns
resDf2 <- resDf %>% gather('Dataset','loss','loss','val_loss', factor_key=T)
levels(resDf2$Dataset)[levels(resDf2$Dataset) == 'loss'] <- 'Training'
levels(resDf2$Dataset)[levels(resDf2$Dataset) == 'val_loss'] <- 'Validation'

# we do not need to gather kappa, as the rows are already duplicated for training/validation entries
resDf3 <- resDf2
resDf3[resDf3$Dataset == 'Validation',"kappa"] <- resDf3[resDf3$Dataset == 'Validation',"val_kappa"]
resDf3 <- resDf3[,names(resDf3) != 'val_kappa']


lineTypes <- c("Training" = "solid","Validation"="dashed")
markerTypes <- c("Training" = 19,"Validation"=1)

p1 <- ggplot(resDf3) +
  geom_point(aes(x=epoch, y=loss, color=Fold, shape= Dataset)) +
  geom_smooth(aes(x=epoch, y=loss, fill=Fold, color=Fold, linetype = Dataset)) +
  scale_linetype_manual(values = lineTypes, name = "DataSet") +
  scale_shape_manual(values= markerTypes, name = "DataSet") +
  ggtitle("Learning curves") +
  ylim(0, 1)


p2 <- ggplot(resDf3) +
  geom_point(aes(x=epoch, y=kappa, color=Fold, shape= Dataset)) +
  geom_smooth(aes(x=epoch, y=kappa, fill=Fold, color=Fold, linetype = Dataset)) +
  scale_linetype_manual(values = lineTypes, name = "DataSet") +
  scale_shape_manual(values= markerTypes, name = "DataSet") +
  ylim(0, 1)
  #ggtitle("Learning curves - Kappa")


p3 <- grid.arrange(p1, p2, nrow=2)
#p3

ggsave(outputFile,p3, width = 30, height=20, units = "cm")
print("Figure saved")
