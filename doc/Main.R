##### Main File #####

### packages

library(ggplot2)
library(forecast)
library(plotly)
library(ggfortify)
library(tseries)
library(gridExtra)
library(readr)
library(MASS)
library(data.table)
library(caret)
library(dplyr)
library(randomForest)
library(lme4)
library(xgboost)
library(lubridate)
library(countrycode)
library(highcharter)
library(magrittr)
library(tidyverse)

### Time series model ###

## Load the Data

gtrain_new <- read_csv("../data/gtrain_ts.csv")

## Compute Daily Revenue

date_wise <- gtrain_new  %>%
  group_by(date)  %>%
  summarise(daily_transactionRevenue = sum(transactionRevenue, na.rm =TRUE))

## Model Preparation

acf(date_wise$daily_transactionRevenue)
pacf(date_wise$daily_transactionRevenue)

# Based on this plots, it could be observed that there is a seasonal pattern at lag 7, which makes sense since it could be a weekly pattern.

# Convert data to time series based on observation
TRDts <- ts(date_wise$daily_transactionRevenue, frequency = 7)
plot.ts(date_wise$daily_transactionRevenue, main = "Daily Transaction Revenue")

# Note that the third observation is 0, which could not be used in this test. Thus, we drop it for the test.
ntD=1:length(TRDts)
bcTransD = boxcox(TRDts[-3]~ntD[-3])
lambdaD = bcTransD$x[which(bcTransD$y == max(bcTransD$y))]

dataTrans=(TRDts)^(0.3)
ts.plot(dataTrans)
var(dataTrans)

# Reference: https://www.ime.usp.br/~abe/lista/pdfQWaCMboK68.pdf

## Model Selection

# Select model based on AIC
auto.arima(dataTrans)
fit1 <- arima(dataTrans, order=c(1,0,1), seasonal=list(order=c(2,1,0), period=7))

# Check assumptions for residuals
# Test for independence of residuals
Box.test(residuals(fit1), type="Ljung")
plot(residuals(fit1))

# Histogram
hist(residuals(fit1),main = "Histogram")
# q-q plot
qqnorm(residuals(fit1)) 
qqline(residuals(fit1),col ="blue")

## Cross Validation

# Cross validation test error RMSE
# We compare the RMSE obtained via time series cross-validation with the residual RMSE
data_new <- c(dataTrans)
fit_new <- function(x, h){
  forecast(arima(x, order=c(1,0,1), seasonal=list(order=c(2,1,0), period=7)), h=h)
}
e <- tsCV(data_new, fit_new, h=1)

# Cross Validation Error (Predicted Test Error)
sqrt(mean(e^2, na.rm=TRUE))

# Training Error
sqrt(mean(residuals(arima(data_new, order=c(1,0,1), seasonal=list(order=c(2,1,0), period=7)), h=1)^2, na.rm=TRUE))

## Prediction

# First attempt
fit_new = arima(ts(dataTrans), order=c(1,0,1), seasonal=list(order=c(2,1,0), period=7))
fcast_new <- forecast(fit_new,h=60)
plot(fcast_new, main=" ")

# Improvement of Forecasting
dataNew <- NULL
dataUpper <- matrix(nrow = 60,ncol = 2)
dataUpper[,1] <- c(367:426)
dataLower <- matrix(nrow = 60,ncol = 2)
dataLower[,1] <- c(367:426)
dataNew <- as.numeric(dataTrans) 
length(dataNew) <- 426

for(i in 1:60){
  fit_new = arima(ts(dataNew[(247+(i-1)):(366+(i-1))], frequency = 7), order=c(1,0,1), seasonal=list(order=c(2,1,0), period=7))
  fcast_new <- forecast(fit_new, h=1)
  dataNew[366+i] <- fcast_new$mean
  dataUpper[i,2] <- fcast_new$upper[2]
  dataLower[i,2] <- fcast_new$lower[2]
}

dataCI <- matrix(NA, nrow = 426, ncol = 4)
dataCI[,1] <- c(1:426)
dataCI[,2] <- dataNew
dataCI[367:426,3] <- dataUpper[,2]
dataCI[367:426,4] <- dataLower[,2]
colnames(dataCI) <- c("date", "Reveune", "upper", "lower")

ts.plot(dataNew, xlab = "", ylab = "", main = "Improved Forecasting")
lines(dataUpper, col="blue", lty = 3)
lines(dataLower, col="blue", lty = 3)

# ggplot(data.frame(dataCI), aes(x = date, y = Reveune)) + 
#   geom_line(color = "blue", size = 0.05) +
#   geom_ribbon(data=dataCI,aes(ymin=lower,ymax=upper),alpha=0.3) +
#   theme_minimal() +
#   ylim(low=0,high=20)

# To get the prediction of revenue, we need to transform data into the original form.

#(fcast_new$mean)^(10/3)
(dataNew[367:426])^(10/3)

### Classification Models ###

## Resample dataset to get relatively balanced dataset

gtrain <- read_csv("../data/gtrain.csv")
revenue_convert<-0
gtrain <- data.frame(gtrain,revenue_convert)
gtrain$revenue_convert[gtrain$transactionRevenue > 0]<-1

train_data_Y<-gtrain[gtrain$revenue_convert==1,]
train_data_N<-gtrain[gtrain$revenue_convert==0,]
sample_size_Y <- floor(0.75*nrow(train_data_Y))
sample_size_N <- floor(0.75*nrow(train_data_N))
#set.seed(222)
train.ind_Y <- sample(seq_len(nrow(train_data_Y)), size = sample_size_Y)
train.ind_N <- sample(seq_len(nrow(train_data_N)), size = sample_size_N)
train_Y <- train_data_Y[train.ind_Y,]
train_N <- train_data_N[train.ind_N,]
train_sample_N<-sample_n(train_N,10*nrow(train_Y))
#test_Y <- train_data_Y[-train.ind_Y,]
#test_N <- train_data_N[-train.ind_N,]
#test_sample_N <- sample_n(test_N, nrow(test_Y))
train_new <- rbind(train_Y,train_sample_N)
#test_new <- rbind(test_Y,test_sample_N)

test_new<-read_csv("../data/test_new.csv")
#train_new<-read_csv('../data/train_new.csv')
# train_new$channelGrouping<-as.factor(train_new$channelGrouping)
# train_new$operatingSystem<-as.factor(train_new$operatingSystem)
# 
# test_new$channelGrouping<-as.factor(test_new$channelGrouping)
# test_new$operatingSystem<-as.factor(test_new$operatingSystem)

## Logistic Regression

glm.fit<- glm(revenue_convert~.-X1-transactionRevenue-fullVisitorId-visitId-sessionId-country-operatingSystem, data =train_new , family = binomial)
summary(glm.fit)
coef(glm.fit)
glm.probs <- predict(glm.fit,test_new,type="response")
# test the predictions and the real data
glm.pred <- rep("0",nrow(test_new))
glm.pred[glm.probs>0.5] = "1"
conf<-table(glm.pred, test_new$revenue_convert)

log.acc<-(conf[1,1]+conf[2,2])/sum(conf[,])
log.prec <- conf[2,2]/sum(conf[2,])
# Recall (true pos/true pos + false neg)
log.rec <- conf[2,2]/sum(conf[,2])
# F1 Score
log.f1 <- 2*(log.rec*log.prec)/(log.rec+log.prec)

## XGBoost Classification

grid_default <- expand.grid(
  nrounds = 1000,
  max_depth = 10,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)
xgbtree <- train(factor(revenue_convert) ~ .-X1-transactionRevenue-fullVisitorId-visitId-sessionId,
                 data = train_new,
                 tuneGrid = grid_default,
                 method = "xgbTree",
                 trControl = trainControl(method = "cv"))
xgb.pred <- predict(xgbtree, newdata = test_new)
conf = table(xgb.pred, test_new$revenue_convert)
xgb.acc<-(conf[1,1]+conf[2,2])/sum(conf[,])
xgb.prec <- conf[2,2]/sum(conf[2,])
# Recall (true pos/true pos + false neg)
xgb.rec <- conf[2,2]/sum(conf[,2])
# F1 Score
xgb.f1 <- 2*(xgb.rec*xgb.prec)/(xgb.rec+xgb.prec)

## Random Forest

train_new$revenue_convert<-as.factor(train_new$revenue_convert)
test_new$revenue_convert<-as.factor(test_new$revenue_convert)
model_randomforest<-randomForest(revenue_convert~.-X1-transactionRevenue-fullVisitorId-visitId-sessionId,data =train_new, mtry=5,importance=TRUE, type='classification',na.action = na.pass) 
print(model_randomforest)

#test_new<-read_csv("test_new.csv")
pred <- predict(model_randomforest, test_new)

conf <- table(pred, test_new$revenue_convert,dnn=c("Prediction", "Actual"))
rf.acc<-(conf[1,1]+conf[2,2])/sum(conf[,])

## Comparison

# Precision (true pos/true pos + false pos)
prec <- conf[2,2]/sum(conf[2,])
# Recall (true pos/true pos + false neg)
rec <- conf[2,2]/sum(conf[,2])
# F1 Score
f1 <- 2*(rec*prec)/(rec+prec)

eval.sum <- data.frame(method = c("RF", "Logit", "Xgb"),
                       accuracy = c(rf.acc, log.acc, xgb.acc),
                       precision = c(prec, log.prec, xgb.prec),
                       recall = c(rec, log.rec, xgb.rec),
                       f1score = c(f1, log.f1, xgb.f1))

### Regression Models ###

tr <- read.csv("../data/train_new.csv")
tr <- tr[,-1]
tr <- tr[,-14]
summary(tr$transactionRevenue)

## LMM

m_lmm0 <- glmer(transactionRevenue ~ (1|fullVisitorId), data = tr)

bg_var <- summary(m_lmm0)$varcor$fullVisitorId[1]
resid_var <- attr(summary(m_lmm0)$varcor, "sc")^2

summary(m_lmm0)

m_lmm1 <- update(m_lmm0, transactionRevenue ~ pageviews + (1|fullVisitorId))
m_lmm2 <- update(m_lmm0, transactionRevenue ~ pageviews + hits + (1|fullVisitorId))
m_lmm3 <- update(m_lmm0, transactionRevenue ~ pageviews + hits + visitNumber + (1|fullVisitorId))
m_lmm4 <- update(m_lmm0, transactionRevenue ~ pageviews + hits + visitNumber + channelGrouping + (1|fullVisitorId))
m_lmm5 <- update(m_lmm0, transactionRevenue ~ pageviews + hits + visitNumber + channelGrouping + browser + (1|fullVisitorId))
m_lmm6 <- update(m_lmm0, transactionRevenue ~ pageviews + hits + visitNumber + channelGrouping + browser + operatingSystem + (1|fullVisitorId))
m_lmm7 <- update(m_lmm0, transactionRevenue ~ pageviews + hits + visitNumber + channelGrouping + browser + operatingSystem + country + (1|fullVisitorId))

anova(m_lmm0, m_lmm1, m_lmm2, m_lmm3, m_lmm4, m_lmm5, m_lmm6, m_lmm7)

pred_lmm <- predict(m_lmm7)
RMSE(tr$transactionRevenue, pred_lmm)

te <- read.csv("../data/test_new.csv")

RMSE(te$transactionRevenue, pred_lmm)

## XGBoost Model

train_new<-read_csv('../data/train_new.csv')
test_new<-read_csv("../data/test_new.csv")

paid <- train_new[train_new$revenue_convert == 1,]
paidtest <- test_new[test_new$revenue_convert == 1,]

xgbmodel <- model.matrix(transactionRevenue ~ .-X1-transactionRevenue-fullVisitorId-visitId-sessionId, data = paid)
xgbtest <- model.matrix(transactionRevenue ~ .-X1-transactionRevenue-fullVisitorId-visitId-sessionId, data = paidtest)

xgb.final <- xgboost(data = xgbmodel, 
                     label = paid$transactionRevenue, 
                     eta = 0.01,
                     max_depth = 4, 
                     nrounds = 1000,
                     eval_metric = "rmse",
                     objective = "reg:linear"
)
pred.xgb <- predict(xgb.final, newdata = xgbtest)
sqrt(mean((paidtest$transactionRevenue - pred.xgb)^2))

