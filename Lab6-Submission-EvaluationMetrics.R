# Execute the following command to list all the libraries available in your
# computer:
.libPaths()

# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)


if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

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
# Load the Sonar dataset
data("Sonar")

# Set a random seed for reproducibility
set.seed(123)

# Create a data frame with the target variable and features
df <- as.data.frame(Sonar)


## 1.b. Determine the Baseline Accuracy ----
# Determine the baseline accuracy
sonar_freq <- Sonar$Class
baseline_accuracy <- sum(sonar_freq == "M") / length(sonar_freq)
cat("Baseline Accuracy:", baseline_accuracy, "\n")

# Define the training control with 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5)

## 1.c. Split the dataset ----
# Split the dataset into a 75:25 train:test data split
n_train <- round(0.75 * nrow(df))
train_data <- df[1:n_train, ]
test_data <- df[(n_train + 1):nrow(df), ]

## 1.d. Train the Model ----
# Train a classification model (e.g., logistic regression) with 5-fold cross-validation
model <- train(Class ~ ., data = train_data, method = "glm", metric = "Accuracy", trControl = train_control)

# Display the model's performance using the metric calculated by caret when training the model
print(model)


### Option 2:
# The Confusion Matrix is a type of matrix which is used to visualize the
# predicted values against the actual Values. The row headers in the
# confusion matrix represent predicted values and column headers are used to
# represent actual values.

# Load the Sonar dataset
data("Sonar")

# Set a random seed for reproducibility
set.seed(123)

# Create a data frame with the target variable and features
df <- as.data.frame(Sonar)

# Define the training control (e.g., 5-fold cross-validation)
train_control <- trainControl(method = "cv", number = 5)

# Train a classification model (e.g., random forest, support vector machine, etc.)
model <- train(Class ~ ., data = df, method = "rf", trControl = train_control)

# Predict on the test set (in this example, using the same data for illustration)
predictions <- predict(model, newdata = df)

# Compute a confusion matrix and other classification metrics
confusion_matrix <- confusionMatrix(predictions, df$Class)
print(confusion_matrix)

### Option 3: Display a graphical confusion matrix ----

# Visualizing Confusion Matrix
# Assuming 'confusion_matrix' is the confusion matrix you've computed
library(ggplot2)

# Create a data frame from the confusion matrix
confusion_df <- as.data.frame(as.table(confusion_matrix))

# Plot the confusion matrix as a heatmap
ggplot(data = confusion_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  labs(x = "Actual", y = "Predicted", fill = "Frequency") +
  scale_fill_gradient(low = "lightblue", high = "red") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Confusion Matrix")


fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

# 2. RMSE, R Squared, and MAE ----
## 2.a. Load the dataset ----
data("mtcars")

summary("mtcars")

# Set a random seed for reproducibility
set.seed(123)

# Create a data frame with the target variable and features
df <- as.data.frame(mtcars)

# Define the training control with 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5)

## 2.b. Split the dataset ----
# Split the dataset into a 75:25 train:test data split
n_train <- round(0.75 * nrow(df))
train_data <- df[1:n_train, ]
test_data <- df[(n_train + 1):nrow(df), ]

# Train a linear regression model with 5-fold cross-validation
model <- train(mpg ~ ., data = train_data, method = "lm", trControl = train_control)

# Display the model's performance using the metric calculated by caret when training the model
print(model)

# Predict on the test set
predictions <- predict(model, newdata = test_data)

# Calculate RMSE
rmse <- sqrt(mean((test_data$mpg - predictions)^2))
cat("RMSE =", rmse, "\n")

# Calculate SSR
ssr <- sum((test_data$mpg - predictions)^2)
cat("SSR =", ssr, "\n")

# Calculate SST
sst <- sum((test_data$mpg - mean(test_data$mpg))^2)
cat("SST =", sst, "\n")

# Calculate R-squared
r_squared <- 1 - (ssr / sst)
cat("R-squared =", r_squared, "\n")

# Calculate MAE
absolute_errors <- abs(predictions - test_data$mpg)
mae <- mean(absolute_errors)
cat("MAE =", mae, "\n")




library(pROC)


# 3. Area Under ROC Curve ----

# Split the dataset into a training and testing set
set.seed(7)
train_index <- createDataPartition(Sonar$Class, p = 0.8, list = FALSE)
sonar_train <- Sonar[train_index, ]
sonar_test <- Sonar[-train_index, ]

# Train the KNN model
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
knn_model <- train(Class ~ ., data = sonar_train, method = "knn", metric = "ROC", trControl = train_control)

# Display the model's performance using the metric calculated by caret when training the model
print(knn_model)

# Compute the metric yourself using the test dataset
# Sensitivity and Specificity
predictions <- predict(knn_model, sonar_test[, -ncol(sonar_test)])
actual_labels <- sonar_test$Class  # Use the actual class labels from the test dataset

conf_matrix <- confusionMatrix(predictions, actual_labels)
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]

cat("Sensitivity (True Positive Rate) =", sensitivity, "\n")
cat("Specificity (True Negative Rate) =", specificity, "\n")

# AUC
predictions_prob <- as.numeric(predict(knn_model, sonar_test[, -ncol(sonar_test)], type = "prob")[, "M"])
roc_curve <- roc(ifelse(actual_labels == "M", 1, 0), predictions_prob)

cat("AUC (Area Under ROC Curve) =", auc(roc_curve), "\n")

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for Sonar Model (Class 'M')", col = "blue", lwd = 2.5)



# 4. Logarithmic Loss (LogLoss) ----
# Load the BreastCancer dataset
data(BreastCancer)

## 4.a. Handle Missing Values ----
# Load the BreastCancer dataset
data(BreastCancer)

## 4.a. Handle Missing Values ----
# Load the BreastCancer dataset
# Load the BreastCancer dataset
data(BreastCancer)

# Load the mice package for imputations
library(mice)

# Impute missing values using mice
imputed_data <- mice(BreastCancer, method = "pmm", m = 5)  # You can change "m" as needed

# Combine the imputed datasets
completed_data <- complete(imputed_data)

# 4.a. Train the Model
# We apply the 5-fold repeated cross-validation resampling method with 3 repeats.
train_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = mnLogLoss
)
set.seed(7)

# Train a classification model on the BreastCancer dataset.
# In this case, we use a decision tree (rpart) as an example model.
BreastCancer_model_rpart <- train(Class ~ ., data = completed_data, method = "rpart",
                                  metric = "logLoss", trControl = train_control)

# 4.b. Display the Model's Performance
# Use the metric calculated by caret when training the model.
print(BreastCancer_model_rpart)
