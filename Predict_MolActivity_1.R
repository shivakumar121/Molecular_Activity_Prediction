##### Initiate Libraries #####
library ("caret")
library ("dplyr")
#devtools::install_github("rstudio/keras")
#devtools::install_github("tidyverse/readr")
library(keras)
library (readr)
##### Read Data and create data splits #####
TrainingCSVDir <- "/media/MerckData/MerkDataHome/RawData/TrainingSet"
TestCSVDir <- "/media/MerckData/MerkDataHome/RawData/TestSet"
Train.df <- read_csv (paste0(TrainingCSVDir, "/ACT1_competition_training.csv"))
ValidationSet <- createDataPartition(y = Train.df$Act, times = 1, p = 0.75, list = F)
Train_Validation.df <- Train.df[-ValidationSet,]
Train.df <- Train.df[ValidationSet,]
