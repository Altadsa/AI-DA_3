library(ggplot2)
library(gmodels)
library(plyr)

#Load Feature Data
filepath <- "C:\\Users\\spark\\Documents\\AI-DA_3\\src\\40178464_features.csv"
feature_col_names <- c("label", "index", "nr_pix", "height", "width", "span", "rows_wth_5", "cols_with_5", "neigh1", "neigh5", 
                       "left2tile", "right2tile", "verticalness", "top2tile", "bottom2tile", "horizontalness", "horizontal3tile", "vertical3tile", 
                       "nr_regions", "nr_eyes", "hollowness", "image_fill")

feature_data <- read.csv(filepath, header = TRUE, sep = "\t", col.names = feature_col_names)
feature_data$label <- tolower(as.factor(feature_data$label))
feature_data$classification <- NA

living <- c("banana", "cherry", "flower", "pear")
nonliving <- c("envelope", "golfclub", "pencil", "wineglass")

set.seed(3060)

#Insert classification of Object (Living = 1 or Nonliving = 0)
define_object <- function()
{
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

vert_hist <- ggplot(training_data, aes(x=verticalness, fill = as.factor(classification))) +
  geom_histogram(binwidth = .2, alpha = .5, position = 'identity')
vert_hist
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
for (p in seq(0.01,0.99, by = 0.01))
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
write.csv(p_acc_table, "1.2_p_accuracy.csv")
#1.3 Custom classifier
kfolds = 5

#Highly advise against running this code, as it took a hyperthreaded 8 core processor over 20 mins to complete
#. I used this to determine which features yielded the highest prediction accuracy over the dataset.
#The results should be in an excel document for observation.

# combo_class_table <- data.frame(feature.a = "",
#                                 feature.b = "",
#                                 feature.c = "",
#                                 p.value = 0.0,
#                                 test.accuracy = 0.0,
#                                 stringsAsFactors = FALSE)
# 
# combinations <- combn(colnames(feature_data)[3:22],3)
# shuffled_set <- feature_data[sample(nrow(feature_data)),]
# shuffled_set$folds <- cut(seq(1,nrow(shuffled_set)), breaks = kfolds, labels = FALSE)
# for (i in 1:ncol(combinations))
# {
#   combo = combinations[,i]
#   print(sprintf("Testing combination %s", i))
#   for (p in seq(0.01, 0.99, by = 0.01))
#   {
#     test_acc <- 0
#     for (i in 1:kfolds)
#     {
#       training_data <- shuffled_set[shuffled_set$folds != i,]
#       test_data <- shuffled_set[shuffled_set$folds == i,]
# 
#       fit <- glm(classification ~ eval(parse(text = combo[1])) +
#                  eval(parse(text = combo[2])) +
#                  eval(parse(text = combo[3])),
#                data = training_data,
#                family = 'binomial')
# 
#       #Calculate Accuracy over test data
#       test_data$predicted_val = predict(fit, newdata = test_data, type ="response")
#       test_data$predicted_class = 0
#       test_data$predicted_class[test_data$predicted_val > p] = 1
# 
#       correct_predictions = test_data$predicted_class == test_data$classification
# 
#       f_accuracy = nrow(test_data[correct_predictions,])/nrow(test_data)
#       test_acc <- test_acc + f_accuracy
#     }
# 
#     test_acc <- test_acc / kfolds
#     combo_class_table <- rbind(combo_class_table, c(combo[1], combo[2], combo[3], p, test_acc))
#   }
#     #print(sprintf("Cross-Validated Accuracy with a p>%s cut-off: %s", p, cv_accuracy))
# 
# }
# write.csv(combo_class_table, "combotable.csv")


p <- 0.4
test_acc <- 0
shuffled_set <- feature_data[sample(nrow(feature_data)),]
shuffled_set$folds <- cut(seq(1,nrow(shuffled_set)), breaks = kfolds, labels = FALSE)

p_cont <- data.frame()

for (i in 1:kfolds)
{
  training_data <- shuffled_set[shuffled_set$folds != i,]
  test_data <- shuffled_set[shuffled_set$folds == i,]
  
  fit <- glm(classification ~ span + cols_with_5 + neigh5,
             data = training_data,
             family = 'binomial')
  
  #Predict accuracy over test data
  test_data$predicted_val = predict(fit, test_data, type="response")
  test_data$predicted_class = 0
  test_data$predicted_class[test_data$predicted_val > p] = 1
  
  correct_predictions = test_data$predicted_class == test_data$classification

  f_accuracy = nrow(test_data[correct_predictions,])/nrow(test_data)
  test_acc <- test_acc + f_accuracy

  #This code is for 1.5. It stores the test data labels and the predictions from the model
    
  fold_cont <- CrossTable(x = test_data$label,
                          y = test_data$predicted_class,
                          prop.chisq = F,
                          prop.r = T,
                          prop.c = F,
                          prop.t = F )
  
  #Add the results for this fold to the table and rbind it to the main table
  cont_t <- as.data.frame.matrix(fold_cont$t)
  cont_t <- cbind("label" = rownames(cont_t), cont_t)
  p_cont <- rbind(p_cont, cont_t)
  
}


cv_acc <- test_acc / kfolds


#1.4

#How many correct predictions must be made by the random model?
r_size <- nrow(shuffled_set)
r_successes <- cv_acc * nrow(shuffled_set)

r_binom <- pbinom(r_successes, r_size, 0.5)

x <- 0:200
hx <- dbinom(x, r_size, 0.5)
plot(x, hx,
     xlab = "Correct Predictions",
     ylab = "Probability Density",
     main = "Binomial Distribution for a Random classification model")
abline(v = r_successes, col = "blue")
ggplot(data = NULL, aes(x = x)) +
  geom_point(aes(y = hx)) +
  geom_vline(xintercept = r_successes,
             col = "blue") +
  xlab("Correct Predictions") +
  ylab("Probability Density") +
  ggtitle("Binomial Distribution for a Random classification model")
ggsave("binom_random_class.png", scale = 1, dpi = 400)


binom_sam <- shuffled_set
r_cl <- sample(c(0,1), 1, replace = TRUE)
correct <- nrow(binom_sam[binom_sam$classification == r_cl,])

#1.5

#Taking the table created in 1.3 in anticipation of this section, we clean the table rownames and get the total
#for each observation

rownames(p_cont) <- 1:nrow(p_cont)
p_cont <- ddply(p_cont, "label", numcolwise(sum))
p_cont$total <- rowSums(p_cont[,2:3])

write.csv(x = p_cont, file = "prediciton_test_results.csv")


#Detemrine which features might improve the accuracy of the model

#Get remaining features not used in previous model
# rem_features <- feature_col_names[!feature_col_names %in% c("label","index", "span", "cols_with_5", "neigh5")]

#Create a table to log the results of each test
# log_4_table <- data.frame(feature = "", p.value = 0.0, test.accuracy = 0,
                          # stringsAsFactors = FALSE)

#Perform 5 fold cross validation for different p-cutoffs to determine which values and features will show improved accuracy



# for (i in 1:length(rem_features))
# {
#   feature <- rem_features[i]  
#   print(sprintf("Test %s for %s", i, feature))
#   for (p in seq(0.01,0.99, by = 0.01))
#   {
#     test_acc <- 0
#     for (i in 1:kfolds)
#     {
#       training_data <- shuffled_set[shuffled_set$folds != i,]
#       test_data <- shuffled_set[shuffled_set$folds == i,]
#       
#       fit <- glm(classification ~ span + cols_with_5 + neigh5 +
#                    eval(parse(text = feature)),
#                  data = training_data,
#                  family = 'binomial')
#       
#       #Predict accuracy over test data
#       test_data$predicted_val = predict(fit, test_data, type="response")
#       test_data$predicted_class = 0
#       test_data$predicted_class[test_data$predicted_val > p] = 1
#       
#       correct_predictions = test_data$predicted_class == test_data$classification
#       
#       f_accuracy = nrow(test_data[correct_predictions,])/nrow(test_data)
#       test_acc <- test_acc + f_accuracy
#       
#     }
#     test_acc <- test_acc / kfolds
# 
#     if (test_acc > cv_acc)
#     {
#       log_4_table <- rbind(log_4_table, c(feature, p, test_acc))     
#     }
#   }
# }
# write.csv(log_4_table, "4_feature_log_model_accuracies.csv")


