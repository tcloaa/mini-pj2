# MATH 6380J
# Mini Project 2
# The Combinatorial Drug 20 Efficacy Data
# Author: Lo Tze Cheung, Xia Jiacheng, Dong Chenyang

data_original = read.csv('train.csv')

# train data
data = data_original[!is.na(data_original$Viability), ]

# kaggle predict data
predict = data_original[is.na(data_original$Viability), ]

library(MASS)
library(tree)
library(ISLR)
library(gbm)
library(randomForest)

set.seed(123)

# eliminate ID attribute
data = data[,2:22]

# specify number of folds
nFolds = 5

# generate array containing fold-number for each sample (row)
folds = rep_len(1:nFolds, nrow(data))
folds = sample(folds, nrow(data))

# actual cross validation

# 1. Simple Regression Tree

sum_of_mse = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  tree1 = tree(Viability ~ ., data_train)
  tree1_predict = predict(tree1, newdata = data_test)
  sum_of_mse = sum_of_mse + mean((tree1_predict-data_test[,"Viability"])^2)
}
sum_of_mse/nFolds
# average mse of simple regression tree [1] 0.02408119

# Print final regression tree
tree1 = tree(Viability ~ ., data)
summary(tree1)
plot(tree1)
text(tree1, pretty = 0)

# prune the tree
cv_data = cv.tree(tree1)
plot(cv_data$size, cv_data$dev, type = 'b')
prune_tree1 = prune.tree(tree1, best = 7)
plot(prune_tree1)
text(prune_tree1, pretty = 0)


# 2. Bagging

sum_of_mse = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  bagging1 = randomForest(Viability ~ ., data_train, mtry = 20)
  bagging1_predict = predict(bagging1, newdata = data_test)
  sum_of_mse = sum_of_mse + mean((bagging1_predict-data_test[,"Viability"])^2)
}
sum_of_mse/nFolds
# average mse of bagging [1] 0.01798753
# kaggle test: 0.01066 bagging
bagging1 = randomForest(Viability ~ ., data, mtry = 20)
bagging1_predict = predict(bagging1, newdata = predict)



# 3. Random Forest (mtry = 7)

sum_of_mse = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  rf1 = randomForest(Viability ~ ., data_train, mtry = 7)
  rf1_predict = predict(rf1, newdata = data_test)
  sum_of_mse = sum_of_mse + mean((rf1_predict-data_test[,"Viability"])^2)
}
sum_of_mse/nFolds
# average mse of rf [1] 0.01788655
# kaggle test: 0.01079 random forest
rf1 = randomForest(Viability ~ ., data, mtry = 7)
rf1_predict = predict(rf1, newdata = predict)


# 4. Random Forest (mtry = 5)
sum_of_mse = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  rf2 = randomForest(Viability ~ ., data_train, mtry = 5)
  rf2_predict = predict(rf2, newdata = data_test)
  sum_of_mse = sum_of_mse + mean((rf2_predict-data_test[,"Viability"])^2)
}
sum_of_mse/nFolds
# average mse of rf [1] 0.01814866
rf2 = randomForest(Viability ~ ., data, mtry = 5)
rf2_predict = predict(rf2, newdata = predict)



# 5. Boosting (depth = 1)
sum_of_mse = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  boost1 = gbm(Viability ~ ., data = data_train, distribution = "gaussian", n.trees = 5000, interaction.depth = 1)
  boost1_predict = predict(boost1, newdata = data_test, n.trees = 5000)
  sum_of_mse = sum_of_mse + mean((boost1_predict-data_test[,"Viability"])^2)
}
sum_of_mse/nFolds
# average mse of boosting [1] 0.01706485
boost1 = gbm(Viability ~ ., data = data, distribution = "gaussian", n.trees = 5000, interaction.depth = 1)
boost1_predict = predict(boost1, newdata = predict, n.trees = 5000)




# 6. Boosting (depth = 4)
sum_of_mse = 0
for(k in 1:nFolds) {
  # actual split of the data
  fold = which(folds == k)
  data_train = data[-fold,]
  data_test = data[fold,]
  
  boost4 = gbm(Viability ~ ., data = data_train, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
  boost4_predict = predict(boost4, newdata = data_test, n.trees = 5000)
  sum_of_mse = sum_of_mse + mean((boost4_predict-data_test[,"Viability"])^2)
}
sum_of_mse/nFolds
# average mse of boosting [1] 0.0168837
boost4 = gbm(Viability ~ ., data = data, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
boost4_predict = predict(boost4, newdata = predict, n.trees = 5000)






