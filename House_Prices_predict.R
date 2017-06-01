# 1. Overview
## 1.1 Objective
### Predict sales prices and practice feature engineering, RFs, and gradient boosting

## 1.2 VARIABLE DESCRIPTIONS:
### See data_description.txt

# 2. Set up Project
rm(list=ls())
set.seed(1234)

## 2.1 Set Working Directory
setwd('C:/Users/m339673/Desktop/House_Prices')

## 2.2 Load Libraries
library(randomForest)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(caret)
library(doBy)
library(dplyr)
library(Boruta)
library(corrplot)
library(glmnet)

## 2.3 Import Data
train<-read.csv("train.csv", stringsAsFactors=F)
test<-read.csv("test.csv", stringsAsFactors=F)


## 2.4 Set up my helper functions
### Calculate the mode of a character/numeric variable
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

### Calculate R Squared
rsq<-function(y,f){
  1-sum((y-f)^2)/sum((y-mean(y))^2)
}

### Calculate Adjusted R Squared

adjusted_rsq <- function (y,f,k){
  n<-length(f)
  r2<-rsq(y,f)
  return(1-((1-r2)*(n-1)/(n-k-1)))
}


### Calculate Root Mean Square Error
rmse<-function(y,f){
  sqrt(mean((y-f)^2))
}

## 2.5 Set seed
set.seed(1234)

# 3. Understand and manipulating the variables

## 3.1 Evaluate Outcome Variable
outcome<-'SalePrice'
hist(train[,outcome])
skewness(train[,outcome])
shapiro.test(train[,outcome])

## 3.2 Tranform Outcome
train$ln_SalePrice<-log(train[,outcome])
head(train$ln_SalePrice)
skewness(train$ln_SalePrice) #Reduces the skew
shapiro.test(train$ln_SalePrice) #A little better
outcome<-'ln_SalePrice'

## 3.3 Label categorical and Numeric variables
vars<-colnames(train)
catVars<- vars[sapply(train[,vars],class) %in% c('factor','character')]
numericVars<- vars[sapply(train[,vars],class) %in% c('numeric','integer')]
dropcols<-c("SalePrice","Id", "ln_SalePrice")
numericVars<-setdiff(numericVars, dropcols)



### Label known categorical variables and relabel Categorical and Numeric Variables
cat_vars <- c('MSSubClass','OverallQual','OverallCond', 'MoSold','YrSold')
  

train[cat_vars] <- lapply(train[cat_vars], function(x) as.character(x))
test[cat_vars] <- lapply(test[cat_vars], function(x) as.character(x))


vars<-colnames(train)
catVars<- vars[sapply(train[,vars],class) %in% c('factor','character')]
numericVars<- vars[sapply(train[,vars],class) %in% c('numeric','integer')]
dropcols<-c("SalePrice","Id", 'ln_SalePrice')
catVars<-setdiff(catVars, dropcols)
numericVars<-setdiff(numericVars, dropcols)


## 3.4 Look for NAs and impute values
summary(train[,numericVars])
train$LotFrontage[is.na(train$LotFrontage)] <- 0
test$LotFrontage[is.na(test$LotFrontage)] <- 0
train$MasVnrArea[is.na(train$MasVnrArea)] <- 0
test$MasVnrArea[is.na(test$MasVnrArea)] <- 0
train$BsmtFullBath[is.na(train$BsmtFullBath)] <- 0
test$BsmtFullBath[is.na(test$BsmtFullBath)] <- 0

## Set Garage Year Built to the Year the House was built
train$GarageYrBlt_Imp<-ifelse(is.na(train$GarageYrBlt), train$YearBuilt, train$GarageYrBlt )
test$GarageYrBlt_Imp<-ifelse(is.na(test$GarageYrBlt), test$YearBuilt, test$GarageYrBlt )

#Set Rare Variables of RoofMatl
train$RoofMatl[train$RoofMatl %in% c('ClyTile','Membran','Metal','Roll')]<-'Rare'
test$RoofMatl[test$RoofMatl %in% c('ClyTile','Membran','Metal','Roll')]<-'Rare'


#Set 150 equal to 120 of MSSubClass
train$MSSubClass[train$MSSubClass=='150']<-'120'
test$MSSubClass[test$MSSubClass=='150']<-'120'


#Set 2.5 Fin HouseStyle
train$HouseStyle[train$HouseStyle=='2.5Fin']<-'2Story'
test$HouseStyle[test$HouseStyle=='2.5Fin']<-'2Story'



#Set Other Variables of Exterior2nd
train$Exterior2nd[train$Exterior2nd %in% c('CBlock')]<-'Other'
test$Exterior2nd[test$Exterior2nd %in% c('CBlock')]<-'Other'


#Set Other Variables of Exterior1st
train$Exterior1st[train$Exterior1st %in% c('AsphShn','BrkComm','CBlock','ImStucc','Stone')]<-'Other'
test$Exterior1st[test$Exterior1st %in% c('AsphShn','BrkComm','CBlock','ImStucc','Stone')]<-'Other'


#Combine Building Types
train$BldgType[train$BldgType %in% c('Twnhs','TwnhsE')]<-'Twnhs'
test$BldgType[test$BldgType %in% c('Twnhs','TwnhsE')]<-'Twnhs'



### Set NAs for Categorical Values to 'Missing'
train[,catVars][is.na(train[,catVars])]<-'Missing'
test[,catVars][is.na(test[,catVars])]<-'Missing'


#Set the missing values for MSZoning to the Mode
train$MSZoning[train$MSZoning %in% c('Missing')]<-'RL'
test$MSZoning[test$MSZoning %in% c('Missing')]<-'RL'

#Set the missing values for Exterior1st to Other
train$Exterior1st[train$Exterior1st %in% c('Missing')]<-'Other'
test$Exterior1st[test$Exterior1st %in% c('Missing')]<-'Other'

#Set the missing values for Exterior2nd to Other
train$Exterior2nd[train$Exterior2nd %in% c('AsphShn', 'Brk Cmn', 'Stone', 'Missing')]<-'Other'
test$Exterior2nd[test$Exterior2nd %in% c('AsphShn', 'Brk Cmn', 'Stone', 'Missing')]<-'Other'


#Set Rare foundations to Other
train$Foundation[train$Foundation %in% c('Stone','Wood')]<-'Other'
test$Foundation[test$Foundation %in% c('Stone','Wood')]<-'Other'


#Throw out Utilities as a variable, due to not enough cases of one of the two classes
table(test$Utilities)
table(train$Utilities)


#Set missing KitchenQual to mode
train$KitchenQual[train$KitchenQual %in% c('Missing')]<-'TA'
test$KitchenQual[test$KitchenQual %in% c('Missing')]<-'TA'

#Set Missing Saletype to mode
train$SaleType[train$SaleType %in% c('Missing')]<-'WD'
test$SaleType[test$SaleType %in% c('Missing')]<-'WD'

#Set Missing Functional to mode
train$Functional[train$Functional %in% c('Missing')]<-'Typ'
test$Functional[test$Functional %in% c('Missing')]<-'Typ'


#Set Po value of QC to Fair due to sparseness
train$HeatingQC[train$HeatingQC %in% c('Po')]<-'Fa'
test$HeatingQC[test$HeatingQC %in% c('Po')]<-'Fa'

#Group OverallQual 1 and 2
train$OverallQual[train$OverallQual %in% c(1,2)]<-1.5
test$OverallQual[test$OverallQual %in% c(1,2)]<-1.5


#Group OverallCond 1 and 2
train$OverallCond[train$OverallCond %in% c(1,2)]<-1.5
test$OverallCond[test$OverallCond %in% c(1,2)]<-1.5

#Group Poor Basement Condition with Fair
train$BsmtCond[train$BsmtCond %in% c('Po')]<-'Fa'
test$BsmtCond[test$BsmtCond %in% c('Po')]<-'Fa'

#Group Poor Garage Qual with Fair
train$GarageQual[train$GarageQual %in% c('Po')]<-'Fa'
test$GarageQual[test$GarageQual %in% c('Po')]<-'Fa'



#Group neighborhoods into price groups by average sale price
nmeans <- summaryBy(SalePrice ~ Neighborhood, data=train, FUN=mean)

nmeans$Neighbdecile <- with(nmeans, cut(SalePrice.mean, 
                                breaks=quantile(SalePrice.mean, probs=seq(0,1, by=0.1), na.rm=TRUE), 
                                include.lowest=TRUE, labels=c("1","2","3","4","5","6","7","8","9","10")))

train <- merge(train,nmeans[c(1,3)],by=c("Neighborhood")) 
test <- merge(test,nmeans[c(1,3)],by=c("Neighborhood")) 



#Look at how to group MSSubClass
table(train$MSSubClass)
table(test$MSSubClass)

MSClassmeans <- summaryBy(SalePrice ~ MSSubClass, data=train, FUN=mean)

MSClassmeans

MSClassmeans$MSClassQuintiles <- with(MSClassmeans, cut(SalePrice.mean, 
                                        breaks=quantile(SalePrice.mean, probs=seq(0,1, by=0.20), na.rm=TRUE), 
                                        include.lowest=TRUE, labels=c("1","2","3","4","5")))

train <- merge(train,MSClassmeans[c(1,3)],by=c("MSSubClass")) 
test <- merge(test,MSClassmeans[c(1,3)],by=c("MSSubClass")) 


### Label known factor variables and relabel Categorical and Numeric Variables
factor_vars <- c(catVars)


train[factor_vars] <- lapply(train[factor_vars], function(x) as.factor(x))
test[factor_vars] <- lapply(test[factor_vars], function(x) as.factor(x))



### Ordered Factors
factor_vars_ord <- c('YrSold','OverallQual','OverallCond','NeighbDecile','MSClassQuintiles')

train[factor_vars_ord] <- lapply(train[factor_vars_ord], function(x) as.ordered(x))
test[factor_vars_ord] <- lapply(test[factor_vars_ord], function(x) as.ordered(x))

vars<-colnames(train)
catVars<- vars[sapply(train[,vars],class) %in% c('factor','character')]
numericVars<- vars[sapply(train[,vars],class) %in% c('numeric','integer')]
factVars <- setdiff(vars, c(catVars, numericVars))
dropcols<-c("SalePrice","Id", 'ln_SalePrice','Neighborhood',"MSSubClass")
catVars<-setdiff(catVars, dropcols)
numericVars<-setdiff(numericVars, dropcols)
factVars<-setdiff(factVars, dropcols)



############ Set some of the Quality variables to ordered factors
train$LotShape <- factor(train$LotShape, levels=c("IR3", "IR2", "IR1", 'Reg'), ordered=TRUE)
train$Utilities <- factor(train$Utilities, levels=c("NoSeWa",'AllPub'), ordered=TRUE)
train$LandSlope <- factor(train$LandSlope, levels=c("Sev", "Mod", "Gtl"), ordered=TRUE)
train$ExterQual <- factor(train$ExterQual, levels=c("Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
train$ExterCond <- factor(train$ExterCond, levels=c("Po", "Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
train$BsmtQual <- factor(train$BsmtQual, levels=c('Missing', "Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
train$BsmtCond <- factor(train$BsmtCond, levels=c('Missing', "Fa", "TA", 'Gd'), ordered=TRUE)
train$BsmtExposure <- factor(train$BsmtExposure, levels=c('Missing',"No", "Mn", "Av", 'Gd'), ordered=TRUE)
train$BsmtFinType1 <- factor(train$BsmtFinType1, levels=c('Missing',"Unf", "LwQ", "Rec", 'BLQ','ALQ','GLQ'), ordered=TRUE)
train$BsmtFinType2 <- factor(train$BsmtFinType2, levels=c('Missing',"Unf", "LwQ", "Rec", 'BLQ','ALQ','GLQ'), ordered=TRUE)
train$HeatingQC <- factor(train$HeatingQC, levels=c("Fa", "TA", "Gd", 'Ex'), ordered=TRUE)
train$Electrical <- factor(train$Electrical, levels=c('Missing', 'Mix',"FuseP", "FuseF", "FuseA", 'SBrkr'), ordered=TRUE)
train$KitchenQual <- factor(train$KitchenQual, levels=c("Fa", "TA", "Gd", 'Ex'), ordered=TRUE)
train$Functional <- factor(train$Functional, levels=c('Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'), ordered=TRUE)
train$FireplaceQu <- factor(train$FireplaceQu, levels=c('Missing',"Po", "Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
train$GarageFinish <- factor(train$GarageFinish, levels=c('Missing',"Unf",'RFn','Fin'), ordered=TRUE)
train$GarageQual <- factor(train$GarageQual, levels=c('Missing',"Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
train$GarageCond <- factor(train$GarageCond, levels=c('Missing',"Po", "Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
train$PavedDrive <- factor(train$PavedDrive, levels=c('N','P','Y'), ordered=TRUE)
train$PoolQC <- factor(train$PoolQC, levels=c('Missing','Fa','TA','Gd','Ex'), ordered=TRUE)
train$Fence <- factor(train$Fence, levels=c('Missing','MnWw','GdWo','MnPrv','GdPrv'), ordered=TRUE)



test$LotShape <- factor(test$LotShape, levels=c("IR3", "IR2", "IR1", 'Reg'), ordered=TRUE)
test$Utilities <- factor(test$Utilities, levels=c("NoSeWa",'AllPub'), ordered=TRUE)
test$LandSlope <- factor(test$LandSlope, levels=c("Sev", "Mod", "Gtl"), ordered=TRUE)
test$ExterQual <- factor(test$ExterQual, levels=c("Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
test$ExterCond <- factor(test$ExterCond, levels=c("Po", "Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
test$BsmtQual <- factor(test$BsmtQual, levels=c('Missing', "Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
test$BsmtCond <- factor(test$BsmtCond, levels=c('Missing', "Fa", "TA", 'Gd'), ordered=TRUE)
test$BsmtExposure <- factor(test$BsmtExposure, levels=c('Missing',"No", "Mn", "Av", 'Gd'), ordered=TRUE)
test$BsmtFinType1 <- factor(test$BsmtFinType1, levels=c('Missing',"Unf", "LwQ", "Rec", 'BLQ','ALQ','GLQ'), ordered=TRUE)
test$BsmtFinType2 <- factor(test$BsmtFinType2, levels=c('Missing',"Unf", "LwQ", "Rec", 'BLQ','ALQ','GLQ'), ordered=TRUE)
test$HeatingQC <- factor(test$HeatingQC, levels=c("Fa", "TA", "Gd", 'Ex'), ordered=TRUE)
test$Electrical <- factor(test$Electrical, levels=c('Missing','Mix',"FuseP", "FuseF", "FuseA", 'SBrkr'), ordered=TRUE)
test$KitchenQual <- factor(test$KitchenQual, levels=c("Fa", "TA", "Gd", 'Ex'), ordered=TRUE)
test$Functional <- factor(test$Functional, levels=c('Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'), ordered=TRUE)
test$FireplaceQu <- factor(test$FireplaceQu, levels=c('Missing',"Po", "Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
test$GarageFinish <- factor(test$GarageFinish, levels=c('Missing',"Unf",'RFn','Fin'), ordered=TRUE)
test$GarageQual <- factor(test$GarageQual, levels=c('Missing', "Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
test$GarageCond <- factor(test$GarageCond, levels=c('Missing',"Po", "Fa", "TA", 'Gd', 'Ex'), ordered=TRUE)
test$PavedDrive <- factor(test$PavedDrive, levels=c('N','P','Y'), ordered=TRUE)
test$PoolQC <- factor(test$PoolQC, levels=c('Missing','Fa','TA','Gd','Ex'), ordered=TRUE)
test$Fence <- factor(test$Fence, levels=c('Missing','MnWw','GdWo','MnPrv','GdPrv'), ordered=TRUE)




## 3.5 Find and eliminate correlated variables and linear combinations
dropcols<-c("GarageYrBlt",'Neighborhood',"MSSubClass")
numericVars<-setdiff(numericVars, dropcols)

correlations <- cor(train[,numericVars])
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

correlated_vars<-colnames(train[,numericVars][,findCorrelation(correlations, cutoff=0.75, verbose=FALSE)])



#### Remove the variables
train<-train[ , !(names(train) %in% correlated_vars)]
test<-test[ , !(names(test) %in% correlated_vars)]



vars<-colnames(train)
catVars<- vars[sapply(train[,vars],class) %in% c('factor','character')]
numericVars<- vars[sapply(train[,vars],class) %in% c('numeric','integer')]
factVars <- setdiff(vars, c(catVars, numericVars))

dropcols<-c("SalePrice","Id", 'ln_SalePrice','Neighborhood',"MSSubClass")
catVars<-setdiff(catVars, dropcols)
numericVars<-setdiff(numericVars, dropcols)
factVars <- setdiff(factVars,dropcols)
dropcols<-c("GarageYrBlt",'Neighborhood',"MSSubClass")
numericVars<-setdiff(numericVars, dropcols)


lincombos<-numericVars[findLinearCombos(train[,numericVars])$remove] ## Need to remove

#### Remove the variables
train<-train[ , !(names(train) %in% lincombos)]
test<-test[ , !(names(test) %in% lincombos)]

vars<-colnames(train)
catVars<- vars[sapply(train[,vars],class) %in% c('factor','character')]
numericVars<- vars[sapply(train[,vars],class) %in% c('numeric','integer')]
factVars <- setdiff(vars, c(catVars, numericVars))
dropcols<-c("SalePrice","Id", 'ln_SalePrice','Neighborhood',"MSSubClass")
catVars<-setdiff(catVars, dropcols)
numericVars<-setdiff(numericVars, dropcols)
factVars <- setdiff(factVars,dropcols)
dropcols<-c("GarageYrBlt",'Neighborhood',"MSSubClass")
numericVars<-setdiff(numericVars, dropcols)
allVars<-c(catVars,numericVars, factVars)


### Finally set all NA numeric values to 0

train[is.na(train)]<-0
test[is.na(test)] <-0



## 3.6 Run Boruta Analysis on remaining variables
sample.df <- train[allVars]
boruta.train <- Boruta(sample.df, train$SalePrice, doTrace = 0, maxRuns=100)
print(boruta.train)


final.boruta <- TentativeRoughFix(boruta.train)
plot(final.boruta, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(final.boruta$ImpHistory),function(i)
  final.boruta$ImpHistory[is.finite(final.boruta$ImpHistory[,i]),i])
names(lz) <- colnames(final.boruta$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(final.boruta$ImpHistory), cex.axis = 0.7)

allVars<-getSelectedAttributes(final.boruta, withTentative = FALSE)



catVars<- allVars[sapply(train[,allVars],class) %in% c('factor','character')]
numericVars<- allVars[sapply(train[,allVars],class) %in% c('numeric','integer')]
factVars <- setdiff(allVars, c(catVars, numericVars))


selVars<-union(c(catVars, factVars), numericVars)

# 5. Split into Train and Calibration sets
train_sampling_vector<-createDataPartition(train[,outcome],p=.80, list=FALSE)
train_new<-train[train_sampling_vector,]
train_new_features<-train_new[,selVars]
train_new_labels<-train$ln_SalePrice[train_sampling_vector]
Cal<-train[-train_sampling_vector,]
Cal_features<-Cal[,selVars]


#Linear Regression
fV <- paste(outcome,' ~ ',
            paste(selVars,collapse=' + '),sep='')


lm_model1<-lm(paste0(fV,'-OverallCond-RoofStyle-Exterior2nd-ExterQual-BsmtQual-BsmtCond-BsmtFinType1-BsmtFinType2-FireplaceQu-
                     GarageFinish-GarageQual-Fence-LotFrontage-MasVnrArea-GarageYrBlt_Imp-EnclosedPorch-BedroomAbvGr') ,data=train_new)
summary(lm_model1)


opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(lm_model1, las = 1)

#### Calculate Cook's Distance
d1 <- cooks.distance(lm_model1)
r <- MASS::stdres(lm_model1)
a <- cbind(train_new, d1, r)
a[d1 > .03, ]


### Potential outliers: 842,869,953,1032,1232,1312

rsq(train_new[,outcome], predict(lm_model1, newdata=train_new, interval='none')) #0.9060232
rsq(Cal[,outcome], predict(lm_model1, newdata=Cal, interval='none')) #0.8774585


rmse(train_new[,outcome], predict(lm_model1, newdata=train_new, interval='none')) #0.1221914
rmse(Cal[,outcome], predict(lm_model1, newdata=Cal, interval='none'))  # 0.140786


### Append predictions to test data
test$lmpredlog <- predict(lm_model1, newdata=test, interval='none')

test$lmpred <- exp(test$lmpredlog)



# Linear Model removing outliers: 842,869,953,1032,1232,1312
lm_model2<-lm(paste0(fV,'-LandContour-RoofStyle-Exterior2nd-LotShape-BsmtQual-BsmtCond-HeatingQC-GarageFinish-Fence-
                     LotFrontage-MasVnrArea-BsmtFullBath-BedroomAbvGr-EnclosedPorch-GarageYrBlt_Imp') ,
              data=train_new[!(rownames(train_new)) %in% c(842,869,953,1032,1232,1312),])
              
summary(lm_model2)


opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(lm_model2, las = 1)


### Fit Metrics
rsq(train_new[,outcome], predict(lm_model2, newdata=train_new, interval='none')) # 0.8977988
rsq(Cal[,outcome], predict(lm_model2, newdata=Cal, interval='none')) #0.8840225, Improvement from last fit


rmse(train_new[,outcome], predict(lm_model2, newdata=train_new, interval='none')) #0.127426
rmse(Cal[,outcome], predict(lm_model2, newdata=Cal, interval='none'))  # 0.1369635


### Append predictions to test data
test$lmpred2log <- predict(lm_model2, newdata=test, interval='none')

test$lmpred2 <- exp(test$lmpred2log)

#Stepwise Regression
lm_model_null <- lm(ln_SalePrice~1 ,
                              data=train_new[!(rownames(train_new)) %in% c(842,869,953,1032,1232,1312),])

lm_model_full <- lm(fV ,
                    data=train_new[!(rownames(train_new)) %in% c(842,869,953,1032,1232,1312),])

lm_model3 <-step(lm_model_null, scope=
                   list(lower=lm_model_null, upper=lm_model_full), direction="forward")


summary(lm_model3)


plot(lm_model3, las = 1)


### Fit Metrics
rsq(train_new[,outcome], predict(lm_model3, newdata=train_new, interval='none')) #0.8996019
rsq(Cal[,outcome], predict(lm_model3, newdata=Cal, interval='none')) # 0.8883771, Slightly better than last


rmse(train_new[,outcome], predict(lm_model3, newdata=train_new, interval='none')) #0.126297
rmse(Cal[,outcome], predict(lm_model3, newdata=Cal, interval='none'))  #0.1343677


### Append predictions to test data
test$lmpred3log <- predict(lm_model3, newdata=test, interval='none')

test$lmpred3 <- exp(test$lmpred3log)


#Ridge Regression
fV <- paste(outcome,' ~ ',
            paste(selVars,collapse=' + '),sep='')

#Add a dummy column to test for the predicted value
test$ln_SalePrice<- 1

#remove outliers
train_rem_outliers <- train_new[!(rownames(train_new)) %in% c(842,869,953,1032,1232,1312),]


train_new_mat <- model.matrix(as.formula(fV),train_rem_outliers)[,-1]
Cal_mat <- model.matrix(as.formula(fV), Cal)[,-1]
test_mat <-model.matrix(as.formula(fV), test)[,-1]


ridge.cv <-cv.glmnet(train_new_mat, train_rem_outliers$ln_SalePrice, alpha=0)

lambda_ridge <-ridge.cv$lambda.min

ridge_model <- glmnet(train_new_mat, train_rem_outliers$ln_SalePrice, alpha=0, lambda=lambda_ridge)

predict(ridge_model, type='coefficients', s=lambda_ridge)

ridge_model_pred_train <- predict(ridge_model, s=lambda_ridge, newx=train_new_mat)
ridge_model_pred_Cal <- predict(ridge_model, s=lambda_ridge, newx=Cal_mat)
ridge_model_pred_test <- predict(ridge_model, s=lambda_ridge, newx=test_mat)


### Fit Metrics
rsq(train_rem_outliers[,outcome], ridge_model_pred_train) #0.9290863
rsq(Cal[,outcome], ridge_model_pred_Cal) #  0.8839706, slightly worse than last time, overfits the data

rmse(train_rem_outliers[,outcome], ridge_model_pred_train) #0.1050705
rmse(Cal[,outcome], ridge_model_pred_Cal)  #0.1369942


### Append predictions to test data
test <- cbind(test, data.frame(ridge_model_pred_test))


colnames(test)[401] <- 'ridgepredlog'

test$ridgepred <- exp(test$ridgepredlog)


#LASSO, to eliminate variables

lasso.cv <-cv.glmnet(train_new_mat,train_rem_outliers$ln_SalePrice, alpha=1)

lambda_lasso <-lasso.cv$lambda.min

lasso_model <- glmnet(train_new_mat, train_rem_outliers$ln_SalePrice, alpha=1, lambda=lambda_lasso)

predict(lasso_model, type='coefficients', s=lambda_lasso)


lasso_model_pred_train <- predict(lasso_model, s=lambda_lasso, newx=train_new_mat)
lasso_model_pred_Cal <- predict(lasso_model, s=lambda_lasso, newx=Cal_mat)
lasso_model_pred_test <- predict(lasso_model, s=lambda_lasso, newx=test_mat)



### Fit Metrics
rsq(train_rem_outliers[,outcome], lasso_model_pred_train) #0.9348126
rsq(Cal[,outcome], lasso_model_pred_Cal) #0.8898695


rmse(train_rem_outliers[,outcome], lasso_model_pred_train) #0.100739
rmse(Cal[,outcome], lasso_model_pred_Cal)  #0.1334664



### Append predictions to test data
test<- cbind(test, data.frame(lasso_model_pred_test))


colnames(test)[403] <- 'lassopredlog'

test$lassopred <- exp(test$lassopredlog)


#Regression Tree

fV <- paste(outcome,'~ ',
            paste(selVars,collapse=' + '),sep='')

tmodel <- rpart(fV,data=train_rem_outliers)

opar <- par(mfrow = c(1,1), oma = c(0, 0, 1.1, 0))
prp(tmodel) 

reg_tree_pred_train<-predict(tmodel,newdata=train_rem_outliers)
reg_tree_pred_Cal<-predict(tmodel,newdata=Cal)
reg_tree_pred_test<-predict(tmodel,newdata=test)

### Fit Metrics
rsq(train_rem_outliers[,outcome], reg_tree_pred_train)#0.7633893
rsq(Cal[,outcome], reg_tree_pred_Cal) #0.6686612


rmse(train_rem_outliers[,outcome], reg_tree_pred_train) #0.1919256
rmse(Cal[,outcome], reg_tree_pred_Cal)#0.2315017, poor fit, but lets try a bunch of decision trees



# Random Forest

### Tune model to find optimal mtry value
mtry.tuned <- tuneRF(x=train_new[,selVars],
       y=train_new[,outcome],
       #mtryStart=2,
       ntreeTry=1000,
       stepFactor=2,
       improve=.001,
       trace=TRUE,
       plot=TRUE,
       doBest=FALSE)


mtry.min <- with(data.frame(mtry.tuned),mtry[OOBError == min(OOBError)])


fmodel<-randomForest(x=train_new[,selVars],
                     y=train_new[,outcome],
                     ntree=1000,
                     nodesize=100,
                     importance=T,
                     mtry=mtry.min)


varImpPlot(fmodel,type=1) #Neighborhood and MSSubClass are most important, Basements are also important in this data

fresults<-predict(fmodel,newdata=train_new[,selVars])
fresultsCal<-predict(fmodel,newdata=Cal[,selVars])
fresultstest<-predict(fmodel,newdata=test[,selVars])


### Fit Metrics
rsq(train_new[,outcome], fresults)#0.8306703
rsq(Cal[,outcome], fresultsCal) #0.7866842


rmse(train_new[,outcome], fresults) #0.16402
rmse(Cal[,outcome], fresultsCal)#0.1857503




# Bootstrap Regression with 5 samples

### Create folds for training
data_folds <- createFolds(train$ln_SalePrice, k=5)

res_train <- data.frame()
res_Cal <- data.frame()
res_test <- data.frame()

for (i in 1:length(data_folds)){

fV <- paste(outcome,' ~ ',
            paste(setdiff(selVars, c('RoofStyle','GarageType','OverallQual','OverallCond','GarageCond', 'GarageQual', 'Exterior1st','MSZoning','Exterior2nd',
                                     'Foundation')),collapse=' + '),sep='')


lm_model_boot<-lm(as.formula(fV) ,data=train[data_folds[[i]],])

preds_train <- predict(lm_model_boot, newdata=train_rem_outliers, interval='none')
preds_Cal <- predict(lm_model_boot, newdata=Cal, interval='none')
preds_test <- predict(lm_model_boot, newdata=test, interval='none')

if(i==1){
  res_train <- data.frame(preds_train )
  colnames(res_train) <- paste0('_',i)
  res_Cal <- data.frame(preds_Cal)
  colnames(res_Cal) <- paste0('_',i)
  res_test <- data.frame(preds_test )
  colnames(res_test) <- paste0('_',i)
}
else if (i>1) {
  res_train <-cbind(res_train, data.frame(preds_train ))
  colnames(res_train)[i] <- paste0('_',i)
  res_Cal <-cbind(res_Cal, data.frame(preds_Cal ))
  colnames(res_Cal)[i] <- paste0('_',i)
  res_test <-cbind(res_test, data.frame(preds_test ))
  colnames(res_test)[i] <- paste0('_',i)
}
}



res_train$avgpredlog <- rowMeans(res_train)
res_train$avgpred <- exp(res_train$avgpredlog)
res_Cal$avgpredlog <- rowMeans(res_Cal)
res_Cal$avgpred <- exp(res_Cal$avgpredlog)
res_test$avgpredlog <- rowMeans(res_test)
res_test$avgpred <- exp(res_test$avgpredlog)

### Fit Metrics
rsq(train_rem_outliers[,outcome], res_train$avgpredlog)#0.898518
rsq(Cal[,outcome], res_Cal$avgpredlog)#0.8721145

rmse(train_rem_outliers[,outcome],res_train$avgpredlog) #0.1256928
rmse(Cal[,outcome], res_Cal$avgpredlog)#0.1438231


### Append predictions to data
test$avgpred <- res_test$avgpred





#Try Bootstrapping with LASSO

data_folds <- createFolds(data.frame(train_new_mat), k=2)


res_train <- data.frame()
res_Cal <- data.frame()
res_test <- data.frame()

for (i in 1:length(data_folds)){

lasso.cv <-cv.glmnet(train_new_mat[data_folds[[i]],],train_rem_outliers[data_folds[[i]],]$ln_SalePrice, alpha=1)

lambda_lasso <-lasso.cv$lambda.min

lasso_model <- glmnet(train_new_mat[data_folds[[i]],], train_rem_outliers[data_folds[[i]],]$ln_SalePrice, alpha=1, lambda=lambda_lasso)


preds_train <- predict(lasso_model, s=lambda_lasso, newx=train_new_mat)
preds_Cal <- predict(lasso_model, s=lambda_lasso, newx=Cal_mat)
preds_test <- predict(lasso_model, s=lambda_lasso, newx=test_mat)


if(i==1){
  res_train <- data.frame(preds_train )
  colnames(res_train) <- paste0('_',i)
  res_Cal <- data.frame(preds_Cal)
  colnames(res_Cal) <- paste0('_',i)
  res_test <- data.frame(preds_test )
  colnames(res_test) <- paste0('_',i)
}
else if (i>1) {
  res_train <-cbind(res_train, data.frame(preds_train ))
  colnames(res_train)[i] <- paste0('_',i)
  res_Cal <-cbind(res_Cal, data.frame(preds_Cal))
  colnames(res_Cal)[i] <- paste0('_',i)
  res_test <-cbind(res_test, data.frame(preds_test ))
  colnames(res_test)[i] <- paste0('_',i)
}
}

res_train$avgpredlog <- rowMeans(res_train)
res_train$avgpred <- exp(res_train$avgpredlog)
res_Cal$avgpredlog <- rowMeans(res_Cal)
res_Cal$avgpred <- exp(res_Cal$avgpredlog)
res_test$avgpredlog <- rowMeans(res_test)
res_test$avgpred <- exp(res_test$avgpredlog)


### Fit Metrics
rsq(train_rem_outliers[,outcome], res_train$avgpredlog)#0.6955081
rsq(Cal[,outcome], res_Cal$avgpredlog)#0.653446

rmse(train_rem_outliers[,outcome],res_train$avgpredlog) #0.2177227
rmse(Cal[,outcome], res_Cal$avgpredlog)#0.2177227


### Append predictions to data
test$avgpredlasso <- res_test$avgpred






head(test)


##### Export the test dataset with predictions
final_submission_df<-test[,c('Id','avgpredridge')]

colnames(final_submission_df)[2]<-'SalePrice'

head(final_submission_df)

write.csv(final_submission_df, file = 'McCullough_HousePrices.csv', append=FALSE, sep = " ",
          eol = "\n", na = "NA", dec = ".", row.names = FALSE,
          col.names = TRUE, qmethod = c("escape", "double"))
