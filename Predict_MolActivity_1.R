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
Train.df <- read.csv (paste0(TrainingCSVDir, "/ACT1_competition_training.csv"))
##### Make the effect discrete ######
Act_Discrete <- Train.df$Act
Discrete_Temp <- ceiling (Train.df$Act) - round(Train.df$Act)
Index_Ones <- which(Discrete_Temp == 1)
Act_Discrete[Index_Ones] <- round(Act_Discrete[Index_Ones])
Act_Discrete[-Index_Ones] <- ceiling(Train.df$Act[-Index_Ones]) - 0.5
##### Split DataSet ############
ValidationSet <- createDataPartition(y = Act_Discrete, times = 1, p = 0.75, list = F)
Train_Validation.df <- Train.df[-ValidationSet,]
Train.df <- Train.df[ValidationSet,]
##### Define R-squared function#####
Rsquared <- function(x,y) {
  # Returns R-squared.
  # R2 = \frac{[\sum_i(x_i-\bar x)(y_i-\bar y)]^2}{\sum_i(x_i-\bar x)^2 \sum_j(y_j-\bar y)^2}
  # Arugments: x = solution activities
  #            y = predicted activities
  
  if ( length(x) != length(y) ) {
    warning("Input vectors must be same length!")
  }
  else {
    avx <- mean(x) 
    avy <- mean(y)
    num <- sum( (x-avx)*(y-avy) )
    num <- num*num
    denom <- sum( (x-avx)*(x-avx) ) * sum( (y-avy)*(y-avy) )
    return(num/denom)
  }
}
##### Create Keras model ##########
Act_Discrete_Validation <- Act_Discrete[ValidationSet]
One_Hot_Matrix <- matrix(nrow = nrow(Train.df), ncol = length(levels(factor(Act_Discrete))))
colnames(One_Hot_Matrix) <- levels(factor(Act_Discrete_Validation))
for(i in 1:length(levels(factor(Act_Discrete))))
{
  Temp_Index <- which(Act_Discrete_Validation == levels(factor(Act_Discrete_Validation))[i])
  One_Hot_Matrix[,i] <- 0
  One_Hot_Matrix[Temp_Index,i] <- 1
}
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 27931, activation = 'relu', input_shape = c(9491)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 27931, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 9, activation = 'softmax')
summary (model)
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_sgd(lr = 0.5, decay = 0.05),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x = as.matrix(Train.df[,3:ncol(Train.df)]), y = One_Hot_Matrix, 
  epochs = 20, batch_size = 128, 
  validation_split = 0.2
)

plot(history)






