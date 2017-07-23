##### Initiate Libraries #####
library ("caret")
library ("dplyr")
#devtools::install_github("rstudio/keras")
#devtools::install_github("tidyverse/rlang")
#devtools::install_github("tidyverse/tibble")
#devtools::install_github("tidyverse/readr")
#devtools::install_github("rstudio/tensorflow", force = T)
library(keras)
library (readr)
library (tensorflow)
#install_tensorflow(gpu = T)
config = tf$gpu_options$allocator_type = 'BFC'
#config$gpu_options$allocator_type = 'BFC'
tf$GPUOptions$ALLOCATOR_TYPE_FIELD_NUMBER = "BFC"
sess <- tf$Session
with(tf$Session(config = tf$GPUOptions$ALLOCATOR_TYPE_FIELD_NUMBER = "BFC") %as% sess)
with tf.Session(config = config) as s
##### Read Data and create data splits #####
TrainingCSVDir <- "/media/Merck_Data/MerkDataHome/RawData/TrainingSet"
TestCSVDir <- "/media/Merck_Data/MerkDataHome/RawData/TestSet"
#Train.df <- read.csv (paste0(TrainingCSVDir, "/ACT1_competition_training.csv"))
#saveRDS(Train.df, file = "Trian.df_Jul042017.rds")
Train.df <- readRDS("Trian.df_Jul042017.rds")
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
#saveRDS(object = Train.df, file = "TrainAfterValidation_Jul122017.rds")
Train.df <- readRDS(file = "TrainAfterValidation_Jul122017.rds")
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
  optimizer = optimizer_sgd(lr = 0.000005, decay = 0.0000001),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x = as.matrix(Train.df[,3:ncol(Train.df)]), y = One_Hot_Matrix, 
  epochs = 20, batch_size = 32, 
  validation_split = 0.2
)

plot(history)

###### PlayGround ###############
#tt <- prcomp(Train.df[,3:ncol(Train.df)])
#saveRDS(tt, file = "tt_Jul122017.rds")
#ttprin <- princomp(Train.df[,3:ncol(Train.df)])
#saveRDS(ttprin, file = "ttprin_Jul182017.rds")
tt <- readRDS(file="tt_Jul122017.rds")
Fit_PrincComp <- readRDS("ttprin_Jul182017.rds")

MyData <- data.frame(tt$x)
p1 <- ggplot(MyData)
p1 <- p1 + geom_point(aes(PC2,PC3))
p1
ttmeans= kmeans(x=tt$x[,1:2], centers = 6)
p1 <- ggplot(MyData)
p1 <- p1 + geom_point(aes(PC1,PC2), color = ttmeans$cluster)
p1
###############
MidAct <- max(Train.df$Act) - min(Train.df$Act)
ScaledAct <- MidAct - Train.df$Act
#ScaledAct <- ScaledAct / var(ScaledAct)
p1 <- ggplot(MyData)
p1 <- p1 + geom_point(aes(PC2,PC3, color = ScaledAct), alpha = 1, size = 0.5) 
p1
class(tt$x)
####### Plot princomp #####
MyData_PrinComp <- data.frame(Fit_PrincComp$scores)
p1 <- ggplot(MyData_PrinComp)
p1 <- p1 + geom_point(aes(Comp.2,Comp.3, color = ScaledAct), alpha = 1, size = 0.5) 
p1
class(tt$x)
#### Plot activity as a function of PCs #############
ActNormalized <- Train.df$Act - mean(Train.df$Act)
ActNormalized <- ActNormalized / (max(ActNormalized) - min(ActNormalized))
fit <- glm (ActNormalized ~ tt$x[,5648])
summary(fit)
EngPC <- exp(-(PC_Normalized)^2)
EngPC <- poly(PC_Normalized,3)
fit <- lm (ActNormalized ~ EngPC)
summary(fit)
p1 <- ggplot(MyData)
p1 <- p1 + geom_point(aes(PC_Normalized, ActNormalized))
p1 <- p1 + geom_abline(intercept = fit$coefficients[1], slope = fit$coefficients[2], color = "red")
p1

FitCoeff <- NULL
for (i in 1:ncol(tt$x))
{
  PC_Normalized <- tt$x[,i] - mean(tt$x[,i])
  PC_Normalized <- PC_Normalized / max(tt$x[,i]) - (min(tt$x[,i]))
  #PC_Normalized <- PC_Normalized/max(PC_Normalized)
  #PC_Normalized <- PC_Normalized - min(PC_Normalized)
  fit <- glm (ActNormalized ~ PC_Normalized, family = gaussian)
  FitCoeff <- c(FitCoeff,fit$coefficients[2])
}
saveRDS(object = FitCoeff, file = "FitCoeff_Jul122017.rds")
class(FitCoeff[1])
CloseToOne <- 1 - abs(FitCoeff)
which((CloseToOne == min(abs(CloseToOne))) | (CloseToOne == - min(abs(CloseToOne))))
fit <- glm (Train.df$Act ~ tt$x[,6])
summary(fit)

