library(dplyr)
library(readr)

cat("R container is running!\n")

# Step 23: load train.csv
train <- read_csv("data/train.csv", show_col_types = FALSE)
cat("Loaded train.csv:", nrow(train), "rows,", ncol(train), "columns\n")

# Explore
cat("\nFirst 5 rows:\n")
print(head(train, 5))
miss <- colSums(is.na(train))
cat("\nMissing values per column:\n")
print(miss[miss > 0])
cat("\nOverall survival rate:", round(mean(train$Survived), 3), "\n")
cat("\nSurvival rate by sex:\n")
print(train %>% group_by(Sex) %>% summarise(survival_rate = mean(Survived)))
cat("\nSurvival rate by passenger class:\n")
print(train %>% group_by(Pclass) %>% summarise(survival_rate = mean(Survived)))

# Adjust the data
age_median <- median(train$Age, na.rm = TRUE)
fare_median <- median(train$Fare, na.rm = TRUE)
embarked_mode <- names(which.max(table(train$Embarked)))
cat("\nFilling missing Age with median (", age_median, "), Fare with median (",
    round(fare_median, 2), "), Embarked with mode (", embarked_mode, ")\n", sep = "")
cat("Converting Sex, Pclass, Embarked to factors; adding FamilySize = SibSp + Parch + 1\n")

prepare <- function(df) {
  df %>%
    mutate(
      Age = ifelse(is.na(Age), age_median, Age),
      Fare = ifelse(is.na(Fare), fare_median, Fare),
      Embarked = factor(ifelse(is.na(Embarked), embarked_mode, Embarked),
                        levels = c("C", "Q", "S")),
      Sex = factor(Sex, levels = c("male", "female")),
      Pclass = factor(Pclass, levels = c(1, 2, 3)),
      FamilySize = SibSp + Parch + 1
    )
}

train_clean <- prepare(train)
cat("Missing values in model columns after cleaning:",
    sum(is.na(train_clean[, c("Pclass", "Sex", "Age", "FamilySize", "Fare", "Embarked")])), "\n")

# Build the model
set.seed(42)
idx <- sample(nrow(train_clean), size = floor(0.8 * nrow(train_clean)))
train_set <- train_clean[idx, ]
val_set <- train_clean[-idx, ]
cat("\nSplit train.csv into", nrow(train_set), "training rows and", nrow(val_set), "validation rows\n")

model <- glm(Survived ~ Pclass + Sex + Age + FamilySize + Fare + Embarked,
             data = train_set, family = binomial)
cat("Trained logistic regression model\nCoefficients:\n")
print(round(coef(model), 3))

accuracy <- function(model, df) {
  p <- predict(model, newdata = df, type = "response")
  mean(as.integer(p > 0.5) == df$Survived)
}

# Accuracy on training set
cat("\nTraining accuracy:", round(accuracy(model, train_set), 3), "\n")

# Load test.csv and predict
test <- read_csv("data/test.csv", show_col_types = FALSE)
cat("\nLoaded test.csv:", nrow(test), "rows,", ncol(test), "columns\n")
test_clean <- prepare(test)
test_pred <- as.integer(predict(model, newdata = test_clean, type = "response") > 0.5)
cat("Predicted", sum(test_pred), "survivors out of", length(test_pred), "test passengers\n")
cat("\nFirst 10 test set predictions:\n")
print(head(data.frame(PassengerId = test$PassengerId, Survived = test_pred), 10))

# test.csv has no Survived labels, so the held-out 20% of train.csv measures accuracy
cat("\nHeld-out validation accuracy:", round(accuracy(model, val_set), 3), "\n")