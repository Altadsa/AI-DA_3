library(ggplot2)

#Load Feature Data
filepath <- "C:\\Users\\spark\\Documents\\AI-DA_3\\src\\40178464_features.csv"
feature_col_names <- c("label", "index", "nr_pix", "height", "width", "span", "rows_wth_5", "cols_with_5", "neigh1", "neigh5", 
                       "left2tile", "right2tile", "verticalness", "top2tile", "bottom2tile", "horizontalness", "horizontal3tile", "vertical3tile", 
                       "nr_regions", "nr_eyes", "hollowness", "image_fill")

feature_data <- read.csv(filepath, header = TRUE, sep = "\t", col.names = feature_col_names)
feature_data$label <- tolower(as.factor(feature_data$label))
feature_data$classification <- NA

set.seed(3060)

#Insert classification of Object (Living = 1 or Nonliving = 0)
define_object <- function()
{
  living <- c("banana", "cherry", "flower", "pear")
  nonliving <- c("envelope", "golfclub", "pencil", "wineglass")
  classifications <- c()
  for (i in 1:nrow(feature_data))
  {
    classification <- -1
    name <- feature_data[i,1]
    if (name %in% living)
    {
      classification <- 1
    }
    else if (name %in% nonliving)
    {
      classification <- 0
    }
    else
    {
      print("Undefined")
      return()
    }
    classifications <- c(classifications, classification)
  }
  return (classifications)
}

feature_data$classification <- define_object()

#Create a training Dataset
shuffled_set <- feature_data[sample(nrow(feature_data)),]
training_data <- shuffled_set[1:132,]
test_data <- shuffled_set[133:160,]

set_plot <- ggplot(training_data, aes(x=verticalness, fill = as.factor(classification))) +
  geom_histogram(binwidth = .2, alpha = .5, position = 'identity')
set_plot
ggsave('hist_verticalness.png', scale = 1, dpi = 400)


#1.1 Generate a Logistic Regression Model using glm function with the Training Data
log_model <- glm(classification ~ verticalness,
                 data = training_data,
                 family = 'binomial')
summary(log_model)

#Plot the fitted curve
x_range <- range(training_data$verticalness)
x_axis <- seq(x_range[1], x_range[2], length.out = 1000)

fitted_curve <- data.frame(verticalness = x_axis)
fitted_curve$classification <- predict(log_model, fitted_curve, type = "response")

curve_plot <- ggplot(training_data, aes(x=verticalness, y=classification)) + 
  geom_point(aes(colour = factor(classification)), 
             show.legend = T, position="dodge")+
  geom_line(data=fitted_curve, colour="orange", size=1)
curve_plot
ggsave('log_model_fitted_curve.png', scale = 1, dpi = 400)

#1.2 Calculate Accuracy of Model using the 160 items in the Feature Data

p_acc_table <- data.frame(p = character(), accuracy = character())
for (p in seq(0,1, by = 0.01))
{
    test_data$predicted_val = predict(log_model, test_data, type="response")
    test_data$predicted_class = 0
    test_data$predicted_class[test_data$predicted_val > p] = 1

    correct_predictions = test_data$predicted_class == test_data$classification
    correct = nrow(test_data[correct_predictions,])
    total = nrow(test_data)
    accuracy = correct/total
    #print(sprintf("Accuracy with a p>%s cut-off: %s", p, accuracy))
    p_acc_table <- rbind(p_acc_table, c(p, accuracy))
}
colnames(p_acc_table) <- c("p.value", "accuracy")

#1.3 Custom classifier
kfolds = 5

# combo_class_table <- data.frame(feature.a = "",
#                                 feature.b = "",
#                                 feature.c = "",
#                                 p.value = 0.0,
#                                 train.accuracy = 0.0,
#                                 test.accuracy = 0.0,
#                                 stringsAsFactors = FALSE)
# 
# combinations <- combn(colnames(feature_data)[3:22],3)
# for (i in 1:ncol(combinations))
# {
#   combo = combinations[,i]
#   print(sprintf("Testing combination %s", i))
#   for (p in seq(0,1, by = 0.01))
#   {
#     train_acc <- 0
#     test_acc <- 0
#     shuffled_set <- feature_data[sample(nrow(feature_data)),]
#     shuffled_set$folds <- cut(seq(1,nrow(shuffled_set)), breaks = kfolds, labels = FALSE)
#     for (i in 1:kfolds)
#     {
#       training_data <- shuffled_set[shuffled_set$folds != i,]
#       test_data <- shuffled_set[shuffled_set$folds == i,]
# 
#       fit <- glm(classification ~ eval(parse(text = combo[1])) + 
#                    eval(parse(text = combo[2])) + 
#                    eval(parse(text = combo[3])),
#                  data = training_data,
#                  family = 'binomial')
#       
#       #Calculate Accuracy over training data
#       training_data$predicted_val = predict(fit, newdata = training_data, type="response")
#       training_data$predicted_class = 0
#       training_data$predicted_class[training_data$predicted_val > p] = 1
#       
#       correct_predictions = training_data$predicted_class == training_data$classification
#       
#       f_accuracy = nrow(training_data[correct_predictions,])/nrow(training_data)
#       train_acc <- train_acc + f_accuracy      
#       
#       #Calculate Accuracy over test data
#       test_data$predicted_val = predict(fit, newdata = test_data, type="response")
#       test_data$predicted_class = 0
#       test_data$predicted_class[test_data$predicted_val > p] = 1
# 
#       correct_predictions = test_data$predicted_class == test_data$classification
# 
#       f_accuracy = nrow(test_data[correct_predictions,])/nrow(test_data)
#       test_acc <- test_acc + f_accuracy
#     }
#     
#     train_acc <- train_acc / kfolds
#     test_acc <- test_acc / kfolds
#     combo_class_table <- rbind(combo_class_table, c(combo[1], combo[2], combo[3], p, train_acc, test_acc))
#     #print(sprintf("Cross-Validated Accuracy with a p>%s cut-off: %s", p, cv_accuracy))
#   }
# }
# write.csv(combo_class_table, "combotable.csv")
p <- 0.39
train_acc <- 0
test_acc <- 0
shuffled_set <- feature_data[sample(nrow(feature_data)),]
shuffled_set$folds <- cut(seq(1,nrow(shuffled_set)), breaks = kfolds, labels = FALSE)
for (i in 1:kfolds)
{
  training_data <- shuffled_set[shuffled_set$folds != i,]
  test_data <- shuffled_set[shuffled_set$folds == i,]
  
  fit <- glm(classification ~ span + cols_with_5 + neigh5,
             data = training_data,
             family = 'binomial')
  
  #Predict accuracy over training data
  training_data$predicted_val = predict(fit, training_data, type="response")
  training_data$predicted_class = 0
  training_data$predicted_class[training_data$predicted_val > p] = 1
  
  correct_predictions = training_data[["predicted_class"]] == training_data[["classification"]]    
  
  f_accuracy <- nrow(training_data[correct_predictions,])/nrow(training_data)
  train_acc <- train_acc + f_accuracy
  
  #Predict accuracy over test data
  test_data$predicted_val = predict(fit, test_data, type="response")
  test_data$predicted_class = 0
  test_data$predicted_class[test_data$predicted_val > 0.38] = 1
  
  correct_predictions = test_data$predicted_class == test_data$classification
  
  f_accuracy = nrow(test_data[correct_predictions,])/nrow(test_data)
  test_acc <- test_acc + f_accuracy
}
train_acc <- train_acc / kfolds
test_acc <- test_acc / kfolds
print(sprintf("Cross-Validated Training Accuracy with a p>%s cut-off: %s", p, train_acc))
print(sprintf("Cross-Validated Test Accuracy with a p>%s cut-off: %s", p, test_acc))




