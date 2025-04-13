# Load necessary libraries
if (!require("xgboost")) install.packages("xgboost", dependencies = TRUE)
library(xgboost)

# Load Training Data
Labs <- scan(file = "C:\\Users\\kyrie\\Downloads\\Project Data\\Project Data\\ProjectTrainingData.csv", what = "xx", sep = ",", nlines = 1)
Data <- matrix(
  scan(file = "C:\\Users\\kyrie\\Downloads\\Project Data\\Project Data\\ProjectTrainingData.csv", what = "xx", sep = ",", skip = 1),
  ncol = length(Labs), byrow = TRUE
)
colnames(Data) <- Labs

# Convert to data.frame and handle numeric conversion
Data <- as.data.frame(Data)
numeric_cols <- c("C1", "banner_pos", "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21")
Data[numeric_cols] <- lapply(Data[numeric_cols], as.numeric)
Data[] <- lapply(Data, function(col) {
  if (is.character(col)) as.numeric(as.factor(col)) else as.numeric(col)
})

# Extract target variable 'click' and predictors
Data$click <- as.numeric(Data$click)
y_train <- Data$click
X_train <- Data[, setdiff(colnames(Data), "click")]

# Sample a subset of the data for training
set.seed(123) # For reproducibility
sample_size <- 500000 # Adjust this size based on available memory
sample_indices <- sample(1:nrow(X_train), size = sample_size)
X_train_sample <- X_train[sample_indices, ]
y_train_sample <- y_train[sample_indices]

# Load Test Data
Labs_test <- scan(file = "C:\\Users\\kyrie\\Downloads\\Project Data\\Project Data\\ProjectTestData.csv", what = "xx", sep = ",", nlines = 1)
TestData <- matrix(
  scan(file = "C:\\Users\\kyrie\\Downloads\\Project Data\\Project Data\\ProjectTestData.csv", what = "xx", sep = ",", skip = 1),
  ncol = length(Labs_test), byrow = TRUE
)
colnames(TestData) <- Labs_test
TestData <- as.data.frame(TestData)

# Convert Test Data to numeric
TestData[numeric_cols] <- lapply(TestData[numeric_cols], as.numeric)
TestData[] <- lapply(TestData, function(col) {
  if (is.character(col)) as.numeric(as.factor(col)) else as.numeric(col)
})

# Add 'id' column if missing
if (!"id" %in% colnames(TestData)) {
  TestData$id <- 1:nrow(TestData)
}

# Convert sampled data to xgboost DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(X_train_sample), label = y_train_sample)
dtest <- xgb.DMatrix(data = as.matrix(TestData))

# Define model parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train the model with xgboost
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 50, # Adjust rounds for testing
  watchlist = list(train = dtrain),
  verbose = 1
)

# Generate predictions for the test dataset
predictions <- predict(xgb_model, dtest)

# Prepare the submission file
submission <- data.frame(id = TestData$id, `P(click)` = predictions)
write.csv(submission, "ProjectSubmission-TeamX.csv", row.names = FALSE)

# Display confirmation message
cat("Submission file 'ProjectSubmission-TeamX.csv' created successfully.\n")

