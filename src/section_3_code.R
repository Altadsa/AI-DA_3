library(randomForest)
library(rpart)
library(tree)
library(ipred)
library(ggplot2)
library(caret)
set.seed(3060)

datapath <- "C:\\Users\\spark\\Documents\\AI-DA_3\\src\\training_data\\features"
feature_col_names <- c("label", "index", "nr_pix", "height", "width", "span", "rows_with_5", "cols_with_5", "neigh1", "neigh5", 
                       "left2tile", "right2tile", "verticalness", "top2tile", "bottom2tile", "horizontalness", "cust_feature_1", "cust_feature_2", 
                       "nr_regions", "nr_eyes", "hollowness", "cust_feature_3")

#Load Training Data from files
csv_files <- list.files(path = datapath, pattern = ".csv", 
                        full.names = TRUE,
                        recursive = FALSE)

#Load Training Data into dataframe
training_data <- data.frame(matrix(ncol = 22, nrow = 0))
training_data <- do.call(rbind, 
                         lapply(csv_files, function(x) {
  csv_features <- read.csv(x, sep = "\t", header = FALSE, col.names = feature_col_names)
}))
colnames(training_data) <- feature_col_names


#Set the formula that will be used to build the models
features <- paste(colnames(training_data)[3:10], collapse = "+")
cl_formula <- as.formula(paste("label ~ ", features, sep = ""))

#3.1

train_sample <- training_data[sample(nrow(training_data)),]

#a. Out of bag
??bagging
bag_sizes <- c(25,50, 200, 400, 800)

oob_accuracies <- c()
for (bag_size in bag_sizes)
{
  bag_sample <- train_sample[sample(bag_size),]
  t_bag <- bagging(cl_formula, data = bag_sample, coob = TRUE)
  
  #t_bag()
  
  pred_vals <- predict(t_bag, train_sample)
  correct <- train_sample$label == pred_vals
  accuracy <- nrow(train_sample[correct,])/nrow(train_sample)
  print(sprintf("Accuracy using bag size %s: %s", bag_size, accuracy))
  oob_accuracies <- c(oob_accuracies, accuracy)
}



#b. Using 5 fold cross validation
kfolds <- 5
cv_accuracies <- c()
for (bag_size in bag_sizes)
{
  bag_sample$folds <- cut(seq(1,nrow(bag_sample)),breaks=kfolds,labels=FALSE)
  cv_accuracy <- 0
  for (n in 1:kfolds)
  {
    train_bag <- bag_sample[bag_sample$fold != n,]
    test_bag <- bag_sample[bag_sample$fold == n,]   
    t_bag <- bagging(cl_formula, data = train_bag, coob = FALSE)
    
    pred_vals <- predict(t_bag, test_bag)
    levels(pred_vals) <- levels(test_bag$label)
    correct <- test_bag$label == pred_vals
    accuracy <- nrow(test_bag[correct,])/nrow(test_bag)
    cv_accuracy <- cv_accuracy + accuracy
  }
  cv_accuracy <- cv_accuracy / kfolds
  print(sprintf("Accuracy using bag size %s and 5 fold cross validation: %s", bag_size, cv_accuracy))  
  cv_accuracies <- c(cv_accuracies, cv_accuracy)
}

model_results <- data.frame(bag.size = bag_sizes, oob.accuracy = oob_accuracies, cv.accuracy = cv_accuracies)

ggplot(model_results, aes(x = bag.size)) +
  geom_line(aes(y = oob.accuracy),
            col = "red") +
  geom_point(aes(y = oob.accuracy),
             col = "red") +
  geom_line(aes(y = cv.accuracy),
            col = "blue") +
  geom_point(aes(y = cv.accuracy),
             col = "blue") +
  xlab("Accuracy") +
  ggtitle("Model Accuracy against Bag sizes") +
  scale_fill_discrete(name = "",
                      breaks = c("red", "blue"),
                      labels = c("Out-of-Bag", "5-fold Cross Validation"))

#3.2 Classification with RandomForests
?randomForest
kfolds <- 5
n_trees <- seq(25,400,by = 25)
n_predictors <- c(2,4,6,8)

best_acc <- 0

f_sample <- train_sample
f_sample$folds = cut(seq(1,nrow(f_sample)),breaks=kfolds,labels=FALSE)
for (n_tree in n_trees)
{
  for (n_predictor in n_predictors)
  {
    cv_accuracy <- 0
    #pred_features <- sample(features)
    #pred_formula <- as.formula(paste("label ~ ", sample(pred_features, n_predictor, replace = TRUE), sep = ""))
    for (i in 1:kfolds)
    {
      train_data <- f_sample[f_sample$folds != i, ]
      val_data <- f_sample[f_sample$folds == i, ]
      
      r_forest <- randomForest(cl_formula, data = train_data, 
                               ntree = n_tree,
                               mtry = n_predictor)
      pred_vals <- predict(r_forest, val_data)
      levels(pred_vals) <- levels(val_data$label)
      correct <- val_data$label == pred_vals
      accuracy <- nrow(val_data[correct,])/nrow(val_data)
      cv_accuracy <- cv_accuracy + accuracy
    }
    cv_accuracy <- cv_accuracy / kfolds
    if (cv_accuracy > best_acc)
    {
      best_acc <- cv_accuracy
      print(sprintf("Accuracy using Random Forests with %s trees and %s predictors: %s", n_tree, n_predictor, cv_accuracy)) 
    }
  }
}

#3.3 

cv_sample_accuracies <- c()

for (n in 1:20)
{
  f_sample <- train_sample[sample(nrow(train_sample)), ]
  f_sample$folds = cut(seq(1,nrow(f_sample)),breaks=kfolds,labels=FALSE)
  cv_accuracy <- 0
  for (i in 1:kfolds)
  {
    train_data <- f_sample[f_sample$folds != i, ]
    val_data <- f_sample[f_sample$folds == i, ]
    
    r_forest <- randomForest(cl_formula, data = train_data, 
                             ntree = 325,
                             mtry = 2)
    pred_vals <- predict(r_forest, val_data)
    levels(pred_vals) <- levels(val_data$label)
    correct <- val_data$label == pred_vals
    accuracy <- nrow(val_data[correct,])/nrow(val_data)
    cv_accuracy <- cv_accuracy + accuracy
  }
  cv_accuracy <- cv_accuracy / kfolds
  cv_sample_accuracies <- c(cv_sample_accuracies, cv_accuracy)
}

print(mean(cv_sample_accuracies))
x <- seq(0,20, by = 1)
hx <- dnorm(x, mean(cv_sample_accuracies), sd(cv_sample_accuracies))
plot(x,hx)

ggplot(data = NULL, aes(x)) +
  geom_line(aes(y = hx))


#3.4

seven_feature_combin <- combn(colnames(training_data[3:10]), 7)

for (n in 1:ncol(seven_feature_combin))
{
  f_combo <- seven_feature_combin[,n]
  new_features <- paste(f_combo, collapse = "+")
  new_cl_formula <- as.formula(paste("label ~ ", new_features, sep = ""))
  cv_accuracy <- 0
  for (i in 1:kfolds)
  {
    train_data <- f_sample[f_sample$folds != i, ]
    val_data <- f_sample[f_sample$folds == i, ]
    
    r_forest <- randomForest(new_cl_formula, data = train_data, 
                             ntree = 325,
                             mtry = 2)
    pred_vals <- predict(r_forest, val_data)
    levels(pred_vals) <- levels(val_data$label)
    correct <- val_data$label == pred_vals
    accuracy <- nrow(val_data[correct,])/nrow(val_data)
    cv_accuracy <- cv_accuracy + accuracy
  }
  print(sprintf("CV Accuracy: %s", cv_accuracy / kfolds))
}






