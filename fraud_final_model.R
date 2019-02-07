## This file contains the code for running the final model for fraud detection
##
## It doesn't show the full analysis and explanations that are included in the
## report. You can find the report and other files in the following site:
##
## https://github.com/esterodr/fraud
##
## All the redundant code is not included here
##
## If you are going to reproduce the could, you should download the database
## available at:
##
## https://www.kaggle.com/ntnu-testimon/paysim1
##
## and unziping it on your working directory.
##
## Please note that this dataset is not saved in GitHub due to space limitations


## Loading packages
library(tidyverse)
library(caret)
library(rpart)

## Loading dataset
if(file.exists("fraud_data.Rda")) {
  load("fraud_data.Rda") 
} else {
  data <- read_csv("PS_20174392719_1491204439457_log.csv")
  save(data, file="fraud_data.Rda")
}

## Creating the validation set
set.seed(1)
test_index <- createDataPartition(y=data$isFraud, times=1,p=0.9,list=FALSE)
working_set <- data[test_index,]
validation_set <- data[-test_index,]
rm(test_index, data)

## Converting variables to factos
working_set$type <- as.factor(working_set$type)
working_set$isFraud <- as.factor(working_set$isFraud)
working_set$isFlaggedFraud <- as.factor(working_set$isFlaggedFraud)

## As there are only frauds on the types "TRANSFER" and "CASH_OUT"
## I will focus on these types of transactions
## See the report for further details
working_set <- working_set %>%
  filter(type %in% c("TRANSFER", "CASH_OUT")) %>%
  mutate(type = as.factor(as.character(type)))

## Create variables to deal to the aparent inconsistencies of the data
working_set <- working_set %>%
  mutate(error_Orig = newbalanceOrig - oldbalanceOrg + amount,
         error_Dest = newbalanceDest - oldbalanceDest - amount,
         zero_Orig = ifelse(oldbalanceOrg==0 & newbalanceOrig ==0 & amount!=0,1,0),
         zero_Dest = ifelse(oldbalanceDest==0 & newbalanceDest ==0 & amount!=0,1,0))
working_set$zero_Orig <- as.factor(working_set$zero_Orig)
working_set$zero_Dest <- as.factor(working_set$zero_Dest)

## Convert "step" variable in to a "day" and "hour" variable
working_set$hour <- working_set$step%%24
working_set$day <- findInterval(working_set$step, seq(0,743,24))

## Delete irrelevant variables
working_set <- working_set %>%
  select(-c(step, nameOrig, nameDest, isFlaggedFraud))

## Make all the former transformations also in the validation_set
validation_set$type <- as.factor(validation_set$type)
validation_set$isFraud <- as.factor(validation_set$isFraud)
validation_set$isFlaggedFraud <- as.factor(validation_set$isFlaggedFraud)
validation_set <- validation_set %>%
  filter(type %in% c("TRANSFER", "CASH_OUT")) %>%
  mutate(type = as.factor(as.character(type)))
validation_set <- validation_set %>%
  mutate(error_Orig = newbalanceOrig - oldbalanceOrg + amount,
         error_Dest = newbalanceDest - oldbalanceDest - amount,
         zero_Orig = ifelse(oldbalanceOrg==0 & newbalanceOrig ==0 & amount!=0,1,0),
         zero_Dest = ifelse(oldbalanceDest==0 & newbalanceDest ==0 & amount!=0,1,0))
validation_set$zero_Orig <- as.factor(validation_set$zero_Orig)
validation_set$zero_Dest <- as.factor(validation_set$zero_Dest)
validation_set$hour <- validation_set$step%%24
validation_set$day <- findInterval(validation_set$step, seq(0,743,24))
validation_set <- validation_set %>%
  select(-c(step, nameOrig, nameDest, isFlaggedFraud))

## Training the model with cp=0 (see the report for explanations)
##
## *Note:* The execution of the following code will take time the first time.
## The results are saved in the file "TD_f.Rda" (available at Github) to make
## the next execution faster.
if (file.exists("TD_f.Rda")) {
  load("TD_f.Rda")
} else {
  fit <- rpart(isFraud ~ ., data = working_set, control = rpart.control(cp = 0))
  save(fit, file="TD_f.Rda")
}

## Prune the model to avoid overfitting
pfit <- prune(fit, cp=0)

## Make predictions
y_hat <- predict(pfit, validation_set, type="class")

## Confusion Matrix
confusionMatrix(y_hat, validation_set$isFraud, positive = "1")

## F-score with beta=2 (see the report for justification)
F_meas(data=y_hat,reference = validation_set$isFraud, beta = 2, relevant = "1")
