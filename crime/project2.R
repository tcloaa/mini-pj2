# MATH 6380J
# Mini Project 2
# The US Crime Data
# Author: Dong Chenyang, Xia Jiacheng, Phil Lo
# Modified from project1's code

# Preprocessing, centering and scaling first
data_original = read.csv('crimedata.csv')
data_original = na.omit(data_original)
data1 = scale(data_original[,6:22], center = TRUE, scale = TRUE)
data2 = scale(data_original[,29:35], center = TRUE, scale = TRUE)
crime_data = as.data.frame(cbind(data1,data_original[,23:28],data2))

set.seed(123)
dt = sort(sample(nrow(crime_data),0.9*nrow(crime_data)))
train = crime_data[dt,]
test = crime_data[-dt,]

library(rpart)
fit <- rpart(murder~rincpc + econgrow + unemp + citypop + a0_5 + a5_9	+ a10_14 + a15_19	+ a20_24 + a25_29	+ citybla +	cityfemh + sta_educ + sta_welf + price + sworn + civil + elecyear +	governor + term2 + term3 + termlim + mayor, data=train)
printcp(fit)
plotcp(fit)
summary(fit)
rsq.rpart(fit)
plot(fit, uniform=TRUE)
text(fit, use.n=TRUE, all=TRUE, cex=.8)
#tree_predict <- predict(tree.model, test)

