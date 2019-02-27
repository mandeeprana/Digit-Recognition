#Name:Mandeep Rana
#IST 707
#HW 7

rm(list=ls()) 
#install.packages("klaR")
#install.packages("MASS")
#install.packages("ElemStatLearn")
#install.packages("randomForest")
#install.packages("class")
library(klaR)
library(MASS)
library(caret)
library(lattice)
library(ggplot2)
library(C50)
library(rpart)
library(arules)
library(e1071)
library(ElemStatLearn)
library(rpart.plot)
library(rattle)
library(randomForest)
library(class)
#reading data
setwd("/Users/manu/Desktop/Fall'18/IST 707/Wk8/")
imgtrain = read.csv("all/train.csv", header = TRUE, stringsAsFactors = FALSE)
imgtest = read.csv("all/test.csv", header = TRUE, stringsAsFactors = FALSE)

str(imgtrain)
str(imgtest)
#since the dataset is very large we randomly sample it to run the model on it
#imgtrain <- imgtrain[sample(nrow(imgtrain), 5000, replace = TRUE), ]

#creating validation set from training data and using that validation set as test set
train_valid <- createDataPartition(imgtrain$label, p = 0.8, list = FALSE)
imgtrain_valid <- imgtrain[train_valid,]
imgtest_valid <- imgtrain[-train_valid,]
table(imgtrain_valid$label)
table(imgtest_valid$label)

######### Random Forest ##############
#running rf model on validation set to test its results and evaluate model performance
labels1 <- as.factor(imgtrain_valid$label)
ptime <- proc.time()
rf_valid <- randomForest(imgtrain_valid[,-1], labels1, imgtest_valid[,-1], ntree = 25)
proc.time()-ptime
rf_valid
actual.labels <- as.factor(imgtest_valid$label)
confusionMatrix(actual.labels, rf_valid$test$predicted)
#running model on original test set for Kaggle submission
labels<-as.factor(imgtrain$label)
ptime <- proc.time()
rf <- randomForest(imgtrain[,-1], labels, imgtest, ntree = 25)
proc.time()-ptime
rf
#Kaggle submission
pred.rf <- data.frame(ImageId= 1:nrow(imgtest), Label= rf$test$predicted)
write.csv(pred.rf, "rf_submission.csv",row.names = FALSE)

############## KNN ##########
#Scaling the data and using PCA
imgtrain_valid.x<- imgtrain_valid[,-1]/255
imgtrain_valid.c<- scale(imgtrain_valid.x, center=TRUE, scale = FALSE)
trainMeans<-colMeans(imgtrain_valid.x)
trainMeansMatrix<-do.call("rbind",replicate(nrow(imgtest_valid),trainMeans,simplif=FALSE))
#generating covariance matrix
imgtrain_valid.conv <- cov(imgtrain_valid.x)
#running pca
imgtrain_valid.pca <- prcomp(imgtrain_valid.conv)
varEx<-as.data.frame(imgtrain_valid.pca$sdev^2/sum(imgtrain_valid.pca$sdev^2))
varEx<-cbind(c(1:784),cumsum(varEx[,1]))
colnames(varEx)<-c("Nmbr PCs","Cum Var")
VarianceExplanation<-varEx[seq(0,200,20),]
# Because we can capture 95+% of the variation in the training data
# using only the first 20 PCs, we extract these for use in the KNN classifier
rotate<-imgtrain_valid.pca$rotation[,1:20]
# matrix with 784 cols and convert it to a matrix with only 20 cols
trainFinal<-as.matrix(imgtrain_valid.c)%*%(rotate)
# We then create a loading matrix for the testing data after applying the same centering and scaling convention as we did for training set
imgtest_valid.x<-imgtest_valid[,-1]/255
testFinal<-as.matrix(imgtest_valid.x-trainMeansMatrix)%*%(rotate)

# Run the KNN predictor on the dim reduced datasets
predict<-knn(train=trainFinal,test=testFinal,cl=labels1,k=3)
predict
confusionMatrix(actual.labels, predict)

# runnning on original test data for kaggle submission
test.kaggle <- imgtest/255
test.kaggle.final <- as.matrix(test.kaggle-trainMeansMatrix)%*%(rotate)
pred.kaggle <- knn(train = trainFinal, test = test.kaggle.final, cl=labels1, k=3)
pred.knn <- data.frame(ImageId= 1:nrow(test.kaggle.final), Label= pred.kaggle)
write.csv(pred.knn, "knn_submission.csv",row.names = FALSE)

############ SVM #############
#running linear svm model on preprocessed data - PCA
model.svm <- svm(labels1~., data=trainFinal, kernel="linear",cost=10)
model.svm

pred.svm <- predict(model.svm, testFinal)
pred.svm

confusionMatrix(actual.labels, pred.svm)

#running linear svm model on original test data for kaggle submission
pred.linear <- predict(model.svm, test.kaggle.final)
pred.linear
pred.svm.linear <- data.frame(ImageId= 1:nrow(test.kaggle.final), Label= pred.linear)
write.csv(pred.svm.linear, "l_svm_submission.csv", row.names = FALSE)

#running non-linear svm on preprocessed data - PCA
model.svm.non <- svm(labels1~., data = trainFinal, kernel = "radial", cost =10)
model.svm.non

pred.svm.non <- predict(model.svm.non, testFinal)
pred.svm.non

confusionMatrix(actual.labels, pred.svm.non)

#running non-linear svm (radial) on original test data for kaggle submission
pred.svm.non <- predict(model.svm.non, test.kaggle.final)
pred.svm.non
pred.radial <- data.frame(ImageId= 1:nrow(test.kaggle.final), Label= pred.svm.non)
write.csv(pred.radial, "R_svm_submission.csv", row.names = FALSE)
