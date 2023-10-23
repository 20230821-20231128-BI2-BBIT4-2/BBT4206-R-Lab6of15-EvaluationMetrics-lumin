# Consider a library as the location where packages are stored.
# Execute the following command to list all the libraries available in your
# computer:
.libPaths()

# One of the libraries should be a folder inside the project if you are using
# renv

# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)



if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}


# There are several evaluation metrics that can be used to evaluate algorithms.
# The default metrics used are:
## (1) "Accuracy" for classification problems and
## (2) "RMSE" for regression problems

# STEP 1. Install and Load the Required Packages ----
## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## pROC ----
if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# 1. Accuracy and Cohen's Kappa ----
## 1.a. Load the dataset ----
data("BreastCancer")

## 1.b. Determine the Baseline Accuracy ----
BreastCancer_freq <- BreastCancer$Class
cbind(frequency =
        table(BreastCancer_freq),
      percentage = prop.table(table(BreastCancer_freq)) * 100)

## 1.c. Split the dataset ----
# Define a 75:25 train:test data split of the dataset

train_index <- createDataPartition(BreastCancer$Class,
                                   p = 0.75,
                                   list = FALSE)
BreastCancer_train <- BreastCancer[train_index, ]
BreastCancer_test <- BreastCancer[-train_index, ]

## 1.d. Train the Model ----
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)



set.seed(7)
BreastCancer_model_glm <-
  train(Class ~ ., data = BreastCancer_train, method = "glm",
        metric = "Accuracy", na.action = na.omit, trControl = train_control)

## 1.e. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----

print(BreastCancer_model_glm)

### Option 2: Compute the metric yourself using the test dataset ----


predictions <- predict(BreastCancer_model_glm, BreastCancer_test[, 1:11])
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         BreastCancer_test[, 1:11]$Class)
print(confusion_matrix)


### Option 3: Display a graphical confusion matrix ----

# Visualizing Confusion Matrix
fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")


# 2. RMSE, R Squared, and MAE ----

## 2.a. Load the dataset ----


## 2.b. Split the dataset ----
# Define a train:test data split of the dataset. 

set.seed(7)

train_index <-
  createDataPartition(demand_forecasting_dataset$`Target (Total orders)`,
                      p = 0.75, list = FALSE)
demand_forecasting_dataset_train <- demand_forecasting_dataset[train_index, ] # nolint
demand_forecasting_dataset_test <- demand_forecasting_dataset[-train_index, ]

## 2.c. Train the Model ----
# We apply bootstrapping with 1,000 repetitions
train_control <- trainControl(method = "boot", number = 1000)

# We then train a linear regression model to predict the value of Employed
# (the number of people that will be employed given the independent variables).
longley_model_lm <-
  train(Employed ~ ., data = longley_train,
        na.action = na.omit, method = "lm", metric = "RMSE",
        trControl = train_control)

## 2.d. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show an RMSE value of approximately 4.3898 and
# an R Squared value of approximately 0.8594
# (the closer the R squared value is to 1, the better the model).
print(longley_model_lm)

### Option 2: Compute the metric yourself using the test dataset ----
predictions <- predict(longley_model_lm, longley_test[, 1:6])

# These are the 6 values for employment that the model has predicted:
print(predictions)

#### RMSE ----
rmse <- sqrt(mean((longley_test$Employed - predictions)^2))
print(paste("RMSE =", rmse))

#### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((longley_test$Employed - predictions)^2)
print(paste("SSR =", ssr))

#### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((longley_test$Employed - mean(longley_test$Employed))^2)
print(paste("SST =", sst))

#### R Squared ----
# We then use SSR and SST to compute the value of R squared
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", r_squared))

#### MAE ----
# MAE measures the average absolute differences between the predicted and
# actual values in a dataset. MAE is useful for assessing how close the model's
# predictions are to the actual values.

# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.
absolute_errors <- abs(predictions - longley_test$Employed)
mae <- mean(absolute_errors)
print(paste("MAE =", mae))

# 3. Area Under ROC Curve ----
# Area Under Receiver Operating Characteristic Curve (AUROC) or simply
# "Area Under Curve (AUC)" or "ROC" represents a model's ability to
# discriminate between two classes.

# ROC is a value between 0.5 and 1 such that 0.5 refers to a model with a
# very poor prediction
# and an AUC of 1 refers to a model that predicts perfectly.

## 3.a. Load the dataset ----
data("BreastCancer")
## 3.b. Determine the Baseline Accuracy ----


BreastCancer_freq <- BreastCancer$Class
cbind(frequency =
        table(BreastCancer_freq),
      percentage = prop.table(table(BreastCancer_freq)) * 100)

## 3.c. Split the dataset ----
# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(BreastCancer$Class,
                                   p = 0.8,
                                   list = FALSE)
BreastCancer2_train <- BreastCancer[train_index, ]
BreastCancer2_test <- BreastCancer[-train_index, ]

## 3.d. Train the Model ----
# We apply the 10-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 10,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)



set.seed(7)
BreastCancer2_model_knn <-
  train(Class ~ ., data = BreastCancer2_train, method = "knn",
        metric = "ROC", na.action = na.omit, trControl = train_control)

## 3.e. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show a ROC value of approximately 0.76 (the closer to 1,
# the higher the prediction accuracy) when the parameter k = 9
# (9 nearest neighbours).

print(BreastCancer2_model_knn)

### Option 2: Compute the metric yourself using the test dataset ----
#### Sensitivity and Specificity ----
predictions <- predict(BreastCancer2_model_knn, BreastCancer2_test[, 1:11])
# These are the values for diabetes that the
# model has predicted:
print(predictions)
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         BreastCancer2_test[, 1:11]$Class)


print(confusion_matrix)

#### AUC ----
# The type = "prob" argument specifies that you want to obtain class
# probabilities as the output of the prediction instead of class labels.
predictions <- predict(BreastCancer2_model_knn, BreastCancer2_test[, 1:11],
                       type = "prob")

# These are the class probability values for diabetes that the
# model has predicted:
print(predictions)

# "Controls" and "Cases": In a binary classification problem, you typically
# have two classes, often referred to as "controls" and "cases."
# These classes represent the different outcomes you are trying to predict.
# For example, in a medical context, "controls" might represent patients without
# a disease, and "cases" might represent patients with the disease.

# Setting the Direction: The phrase "Setting direction: controls < cases"
# specifies how you define which class is considered the positive class (cases)
# and which is considered the negative class (controls) when calculating
# sensitivity and specificity.
roc_curve <- roc(BreastCancer2_test$Class, predictions$pos)

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for KNN Model", print.auc = TRUE,
     print.auc.x = 0.6, print.auc.y = 0.6, col = "blue", lwd = 2.5)
