#Name:Mandeep Rana
#IST 707
#HW 6

rm(list=ls()) 
#install.packages("klaR")
#install.packages("MASS")
#install.packages("ElemStatLearn")
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
#NB
#reading data
setwd("/Users/manu/Desktop/Fall'18/IST 707/Wk7/")
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


#training nb and calculating time to run the model
imgtrain_valid$label<-factor(imgtrain_valid$label)
ptime <- proc.time()
nbmodel <- naiveBayes(imgtrain_valid$label ~ .,data=imgtrain_valid)
#nbmodel <- suppressWarnings(train(label~., data = imgtrain_valid, method = "nb"))
proc.time()-ptime
summary(nbmodel)
#predicting nb and evaluating using confusion matrix
predict.nb <- predict(nbmodel, newdata = imgtest_valid[,-1], type="class")
predict.nb
#calculating accuracy
table('Actual Class'=imgtest_valid$label, 'Predicted Class'=predict.nb)
error.rate <- sum(imgtest_valid$label != predict.nb)/nrow(imgtest_valid)
print(paste0("Accuracy:", 1-error.rate))

#Tuning nb using 3 fold cv
imgtrain_valid$label=factor(imgtrain_valid$label)
ptime.tune <- proc.time()
nbmodel.tune1<- naiveBayes(label ~., data=imgtrain_valid,
                           trControl=trainControl(method = "cv",number = 3),
                           tuneGrid=data.frame(fL=1, usekernel=FALSE, adjust=1))
proc.time()-ptime.tune
summary(nbmodel.tune1)
#nbmodel.tune <- suppressWarnings(train(imgtrain_valid$label~., data=imgtrain_valid,method="nb",metric="Accuracy",
#                                       trControl=trainControl(method = "cv",number = 3)))
#nbmodel.tune
#predicting tuned model and evaluating the model
predict.nb.tune <- suppressWarnings(predict(nbmodel.tune1, newdata=imgtest_valid[,-1], type="class"))
predict.nb.tune
#calculating accuracy
table('Actual Class'=imgtest_valid$label, 'Predicted Class'=predict.nb.tune)
error.rate1 <- sum(imgtest_valid$label != predict.nb.tune)/nrow(imgtest_valid)
print(paste0("Tune model Accuracy:", 1-error.rate1))


#DECISION TREES
dtptime <- proc.time()
#DT<-train(imgtrain_valid[,-1], imgtrain_valid$label ,method="rpart",metric="Accuracy")
DT<-rpart(label~., data=imgtrain_valid)
proc.time()-dtptime
print(DT)
prp(DT)
fancyRpartPlot(DT)

#predicting
predict.dt <- predict(DT, newdata=imgtest_valid[,-1], type="class")
predict.dt
imgtest_valid$label <- factor(imgtest_valid$label)
confusionMatrix(predict.dt, imgtest_valid$label)

#tuned decision tress
dtptime.tune <- proc.time()
DT.tune<-train(imgtrain_valid[,-1], imgtrain_valid$label ,metric="Accuracy",method="rpart",
               trControl=trainControl(method = "cv",number = 3),
               tuneGrid=expand.grid(cp=seq(0,0.1,0.01)))
proc.time()-dtptime.tune
DT.tune$finalModel
prp(DT.tune$finalModel)
fancyRpartPlot(DT.tune$finalModel)

#predicting tuned model
predict.dt.tune <- predict(DT.tune, newdata=imgtest_valid[,-1], type="raw")
predict.dt.tune
imgtest_valid$label <- factor(imgtest_valid$label)
confusionMatrix(predict.dt.tune, imgtest_valid$label)

#kaggle submission
library(data.table)
kaggle.pred <- predict(DT.tune, newdata = imgtest, type="raw")
kaggle.pred
kaggle.sub<-data.frame(kaggle.pred)
final<-setDT(kaggle.sub, keep.rownames=TRUE)[]
colnames(final)<-c("ImageId","Label")
write.csv(final,"Submission.csv",row.names = FALSE)



