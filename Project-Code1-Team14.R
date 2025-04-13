#install.packages("devtools") 
#library(devtools)
#devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')

# Load necessary libraries
library(data.table)
library(catboost)

# Define LogLoss function
LogLoss <- function(PHat, Click) {
  Y <- as.integer(Click)
  eps <- 1e-15
  out <- -mean(Y * log(pmax(PHat, eps)) + (1 - Y) * log(pmax(1 - PHat, eps)))
  return(out)
}

# Load datasets
training_data <- fread('ProjectTrainingData.csv')
test_data <- fread('ProjectTestData.csv')

# Sample 10% of the training data
set.seed(123)  # For reproducibility
sampled_data <- training_data[sample(.N, .N * 0.1)]

# Drop the id, device_ip, and device_id columns
sampled_data <- sampled_data[, !c("id", "device_ip", "device_id"), with = FALSE]
test_data <- test_data[, !c("id", "device_ip", "device_id"), with = FALSE]

# Transform hour into weekday and hour
sampled_data[, `:=`(
  weekday = as.numeric(format(as.POSIXct(hour, format = "%y%m%d%H"), "%u")),
  time = as.numeric(format(as.POSIXct(hour, format = "%y%m%d%H"), "%H"))
)]
test_data[, `:=`(
  weekday = as.numeric(format(as.POSIXct(hour, format = "%y%m%d%H"), "%u")),
  time = as.numeric(format(as.POSIXct(hour, format = "%y%m%d%H"), "%H"))
)]
sampled_data <- sampled_data[, !"hour", with = FALSE]
test_data <- test_data[, !"hour", with = FALSE]

# Split sampled data into training and validation sets
set.seed(123)
train_indices <- sample(seq_len(nrow(sampled_data)), size = 0.8 * nrow(sampled_data))
train_set <- sampled_data[train_indices]
validation_set <- sampled_data[-train_indices]

# Group low-frequency categorical values into 'others'
group_low_freq <- function(column, threshold = 0.01) {
  freq_table <- table(column)
  total <- sum(freq_table)
  low_freq <- names(freq_table[freq_table / total < threshold])
  column[column %in% low_freq] <- 'others'
  return(column)
}

categorical_cols <- c("C1", "site_id", "site_domain", "site_category", 
                      "app_id", "app_domain", "app_category", 
                      "device_model", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21")

for (col in categorical_cols) {
  train_set[[col]] <- group_low_freq(train_set[[col]])
  validation_set[[col]] <- group_low_freq(validation_set[[col]])
  test_data[[col]] <- group_low_freq(test_data[[col]])
  
  # Convert to factors
  train_set[[col]] <- as.factor(train_set[[col]])
  validation_set[[col]] <- as.factor(validation_set[[col]])
  test_data[[col]] <- as.factor(test_data[[col]])
}

# Ensure numerical columns are doubles
train_set <- train_set[, lapply(.SD, function(x) if (is.integer(x)) as.numeric(x) else x)]
validation_set <- validation_set[, lapply(.SD, function(x) if (is.integer(x)) as.numeric(x) else x)]
test_data <- test_data[, lapply(.SD, function(x) if (is.integer(x)) as.numeric(x) else x)]

# Prepare data for CatBoost
train_pool <- catboost.load_pool(data = train_set[, !"click", with = FALSE], 
                                 label = train_set$click)

validation_pool <- catboost.load_pool(data = validation_set[, !"click", with = FALSE], 
                                      label = validation_set$click)

# Hyperparameter tuning
param_grid <- expand.grid(
  iterations = 1000,
  depth = c(4, 6, 8),
  learning_rate = c(0.01, 0.1, 0.2),
  random_seed = 123
)

best_logloss <- Inf
best_params <- list()

for (i in 1:nrow(param_grid)) {
  params <- param_grid[i, ]
  model <- catboost.train(
    learn_pool = train_pool,
    test_pool = validation_pool,
    params = list(
      loss_function = 'Logloss',
      iterations = params$iterations,
      depth = params$depth,
      learning_rate = params$learning_rate,
      eval_metric = 'Logloss',
      random_seed = params$random_seed
    )
  )
  
  validation_predictions <- catboost.predict(model, validation_pool, prediction_type = 'Probability')
  validation_logloss <- LogLoss(validation_predictions, validation_set$click)
  
  if (validation_logloss < best_logloss) {
    best_logloss <- validation_logloss
    best_params <- params
  }
}

print(paste("Best LogLoss:", best_logloss))
print("Best Parameters:")
print(best_params)

# Train the final model with the best parameters
final_model <- catboost.train(
  learn_pool = train_pool,
  test_pool = validation_pool,
  params = list(
    loss_function = 'Logloss',
    iterations = best_params$iterations,
    depth = best_params$depth,
    learning_rate = best_params$learning_rate,
    eval_metric = 'Logloss',
    random_seed = best_params$random_seed
  )
)

# Predict on test data
test_pool <- catboost.load_pool(data = test_data)
test_predictions <- catboost.predict(final_model, test_pool, prediction_type = 'Probability')

# Save test predictions
fwrite(data.table(click_probability = test_predictions), 'TestPredictions.csv')
