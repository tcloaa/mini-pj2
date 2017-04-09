########################################################################
# MATH 6380J
# Mini Project 2
# Tree-based methods analysis for
# The Combinatorial Drug 20 Efficacy Data
# Author: Lo Tze Cheung, Xia Jiacheng, Dong Chenyang
########################################################################
data_original = read.csv('train.csv')

# train data
data = data_original[!is.na(data_original$Viability), ]

# kaggle predict data
kaggle = data_original[is.na(data_original$Viability), ]

library(MASS)
library(tree)
library(ISLR)
library(gbm)
library(randomForest)
library(glmnet)

set.seed(123)

# eliminate ID attribute
data = data[,2:22]

# specify number of folds
nFolds = 5

# generate array containing fold-number for each sample (row)
folds = rep_len(1:nFolds, nrow(data))
folds = sample(folds, nrow(data))

# actual cross validation


########################################################################
# 1. Simple Regression Tree
########################################################################
data = as.data.frame(data)
sum_of_mse_test = 0
sum_of_mse_train = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  tree1 = tree(Viability ~ ., data_train)
  tree1_predict = predict(tree1, newdata = data_test)
  sum_of_mse_test = sum_of_mse_test + mean((tree1_predict-data_test[,"Viability"])^2)
  tree1_train = predict(tree1, newdata = data_train)
  sum_of_mse_train = sum_of_mse_train + mean((tree1_train-data_train[,"Viability"])^2)
}
sum_of_mse_test/nFolds 
#[1] 0.02408119
sum_of_mse_train/nFolds
#[1] 0.009873585


# Print final regression tree
data = as.data.frame(data)
tree1 = tree(Viability ~ ., data)
summary(tree1)
plot(tree1)
text(tree1, pretty = 0)
title("Full Regression Tree")

# prune the tree
cv_data = cv.tree(tree1, k = 5)
plot(cv_data$size, cv_data$dev, type = 'b', xlab = "Tree Size", ylab = "Deviance")
# we need to minimize the deviance, best = 2
prune_tree1 = prune.tree(tree1, best = 2)
plot(prune_tree1)
text(prune_tree1, pretty = 0)
# D3 dominates
# kaggle test: 0.01655
prune_tree1_train = predict(prune_tree1, newdata = data)
mean((prune_tree1_train-data[,"Viability"])^2)
# [1] 0.01766474



sum_of_mse_test = 0
sum_of_mse_train = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  tree1 = tree(Viability ~ ., data_train)
  prune_tree1 = prune.tree(tree1, best = 2)
  tree1_predict = predict(prune_tree1, newdata = data_test)
  sum_of_mse_test = sum_of_mse_test + mean((tree1_predict-data_test[,"Viability"])^2)
  tree1_train = predict(prune_tree1, newdata = data_train)
  sum_of_mse_train = sum_of_mse_train + mean((tree1_train-data_train[,"Viability"])^2)
}
sum_of_mse_test/nFolds 
#[1] 0.01911661
sum_of_mse_train/nFolds
#[1] 0.01754231
# for pruned tree



########################################################################
# 2. Bagging
########################################################################
data = as.data.frame(data)
sum_of_mse_test = 0
sum_of_mse_train = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  bagging1 = randomForest(Viability ~ ., data_train, mtry = 20, ntree = 500)
  bagging1_predict = predict(bagging1, newdata = data_test)
  sum_of_mse_test = sum_of_mse_test + mean((bagging1_predict-data_test[,"Viability"])^2)
  bagging1_train = predict(bagging1, newdata = data_train)
  sum_of_mse_train = sum_of_mse_train + mean((bagging1_train-data_train[,"Viability"])^2)
}
sum_of_mse_test/nFolds
# [1] 0.0179825
sum_of_mse_train/nFolds
# [1] 0.003505421

bagging1 = randomForest(Viability ~ ., data, mtry = 20, importance = TRUE)
bagging1
importance(bagging1)
varImpPlot(bagging1)
bagging1_predict = predict(bagging1, newdata = predict)
# kaggle test: 0.01066 bagging
#No. of variables tried at each split: 20

# Mean of squared residuals: 0.01785492
# % Var explained: 18.59





########################################################################
# 3. Random Forest (mtry = 7)
########################################################################
data = as.data.frame(data)
sum_of_mse_test = 0
sum_of_mse_train = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  rf1 = randomForest(Viability ~ ., data_train, mtry = 7)
  rf1_predict = predict(rf1, newdata = data_test)
  sum_of_mse_test = sum_of_mse_test + mean((rf1_predict-data_test[,"Viability"])^2)
  rf1_train = predict(rf1, newdata = data_train)
  sum_of_mse_train = sum_of_mse_train + mean((rf1_train-data_train[,"Viability"])^2)
}
sum_of_mse_test/nFolds
# [1] 0.01788655
sum_of_mse_train/nFolds
# [1] 0.003922754

rf1 = randomForest(Viability ~ ., data, mtry = 7, importance = TRUE)
rf1_predict = predict(rf1, newdata = predict)
# kaggle test: 0.01079 random forest
#Mean of squared residuals: 0.01753995
#% Var explained: 20.02




########################################################################
# 4. Random Forest (mtry = 4)
########################################################################
data = as.data.frame(data)
sum_of_mse_test = 0
sum_of_mse_train = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  rf2 = randomForest(Viability ~ ., data_train, mtry = 4)
  rf2_predict = predict(rf2, newdata = data_test)
  sum_of_mse_test = sum_of_mse_test + mean((rf2_predict-data_test[,"Viability"])^2)
  rf2_train = predict(rf2, newdata = data_train)
  sum_of_mse_train = sum_of_mse_train + mean((rf2_train-data_train[,"Viability"])^2)
}
sum_of_mse_test/nFolds
# [1] 0.01799601
sum_of_mse_train/nFolds
# [1] 0.004591626

rf2 = randomForest(Viability ~ ., data, mtry = 4, importance = TRUE)
rf2_predict = predict(rf2, newdata = predict)
# var explained
#Number of trees: 500
#No. of variables tried at each split: 4

#Mean of squared residuals: 0.01785131
#% Var explained: 18.6


########################################################################
# 5. Boosting (depth = 1)
########################################################################
data = as.data.frame(data)
sum_of_mse_test = 0
sum_of_mse_train = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  boost1 = gbm(Viability ~ ., data = data_train, distribution = "gaussian", n.trees = 5000, interaction.depth = 1)
  boost1_predict = predict(boost1, newdata = data_test, n.trees = 5000)
  sum_of_mse_test = sum_of_mse_test + mean((boost1_predict-data_test[,"Viability"])^2)
  boost1_train = predict(boost1, newdata = data_train, n.trees = 5000)
  sum_of_mse_train = sum_of_mse_train + mean((boost1_train-data_train[,"Viability"])^2)
}
sum_of_mse_test/nFolds
# [1] 0.01707097
sum_of_mse_train/nFolds
# [1] 0.012269

boost1 = gbm(Viability ~ ., data = data, distribution = "gaussian", n.trees = 5000, interaction.depth = 1)
boost1_predict = predict(boost1, newdata = predict, n.trees = 5000)



########################################################################
# 6. Boosting (depth = 4)
########################################################################
data = as.data.frame(data)
sum_of_mse_test = 0
sum_of_mse_train = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  boost4 = gbm(Viability ~ ., data = data_train, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
  boost4_predict = predict(boost4, newdata = data_test, n.trees = 5000)
  sum_of_mse_test = sum_of_mse_test + mean((boost4_predict-data_test[,"Viability"])^2)
  boost4_train = predict(boost4, newdata = data_train, n.trees = 5000)
  sum_of_mse_train = sum_of_mse_train + mean((boost4_train-data_train[,"Viability"])^2)
}
sum_of_mse_test/nFolds
# [1] 0.01693235
sum_of_mse_train/nFolds
# [1] 0.01032302

boost4 = gbm(Viability ~ ., data = data, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
boost4_predict = predict(boost4, newdata = predict, n.trees = 5000)
# kaggle test: 0.01354




########################################################################
# 7. Lasso
########################################################################
data = as.matrix(data)
sum_of_mse_test = 0
sum_of_mse_train = 0
data = as.matrix(data)
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  lasso1 = glmnet(data_train[,1:20], data_train[,"Viability"], family = "gaussian", alpha = 1)
  lasso1_predict = predict(lasso1, newx = data_test[,1:20])
  sum_of_mse_test = sum_of_mse_test + mean((lasso1_predict-data_test[,"Viability"])^2)
  lasso1_train = predict(lasso1, newx = data_train[,1:20])
  sum_of_mse_train = sum_of_mse_train + mean((lasso1_train-data_train[,"Viability"])^2)
}
sum_of_mse_test/nFolds
# [1] 0.01416397
sum_of_mse_train/nFolds
# [1] 0.0103708

lasso1 = glmnet(data[,1:20], data[,"Viability"], family = "gaussian", alpha = 1)
plot(lasso1,xvar = "lambda",label=TRUE)
title("Lasso Regression")
cvfit = cv.glmnet(data[,1:20], data[,"Viability"],type.measure = "mse", nfolds = 5, alpha = 1)
coef(cvfit,s="lambda.min")
plot(cvfit)
# lambda.min 0.008644296
# 0.01352473
lasso.predict = predict(cvfit, newx = kaggle[,2:21])
write.table(lasso.predict, file ="submission.csv", sep = ",", qmethod = "double", row.names = TRUE) 



########################################################################
# 8. Ridge
########################################################################
data = as.matrix(data)
sum_of_mse_test = 0
sum_of_mse_train = 0
data = as.matrix(data)
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  ridge1 = glmnet(data_train[,1:20], data_train[,"Viability"], family = "gaussian", alpha = 0)
  ridge1_predict = predict(ridge1, newx = data_test[,1:20])
  sum_of_mse_test = sum_of_mse_test + mean((ridge1_predict-data_test[,"Viability"])^2)
  ridge1_train = predict(ridge1, newx = data_train[,1:20])
  sum_of_mse_train = sum_of_mse_train + mean((ridge1_train-data_train[,"Viability"])^2)
}
sum_of_mse_test/nFolds
# [1] 0.01836033
sum_of_mse_train/nFolds
# [1] 0.01586868

ridge1 = glmnet(data[,1:20], data[,"Viability"], family = "gaussian", alpha = 0)
plot(ridge1,xvar = "lambda",label=TRUE)
title("Ridge Regression")
cvfit = cv.glmnet(data[,1:20], data[,"Viability"],type.measure = "mse", nfolds = 5, alpha = 0)
coef(cvfit,s="lambda.min")
plot(cvfit)
# lambda.min = 0.02701961
# min cv error 0.01345389


