install.packages("ggplot2")
library(ggplot2)

#Load Feature Data
filepath <- "C:\\Users\\spark\\Documents\\AI-DA_3\\src\\40178464_features.csv"
feature_col_names <- c("label", "index", "nr_pix", "height", "width", "span", "rows_wth_5", "cols_with_5", "neigh1", "neigh5", 
                       "left2tile", "right2tile", "verticalness", "top2tile", "bottom2tile", "horizontalness", "horizontal3tile", "vertical3tile", 
                       "nr_regions", "nr_eyes", "hollowness", "image_fill")

feature_data <- read.csv(filepath, header = TRUE, sep = "\t", col.names = feature_col_names)

#Insert classification of Object (Living = 1 or Nonliving = 0)
feature_data$classification <- NA
define_object <- function()
{
  living <- c("Banana", "Cherry", "Flower", "Pear")
  nonliving <- c("Envelope", "Golfclub", "Pencil", "Wineglass")
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
training_data <- shuffled_set[1:120,]
test_data <- shuffled_set[121:160,]

set_plot <- ggplot(training_data, aes(x=verticalness, fill = as.factor(classification))) +
  geom_histogram(binwidth = .2, alpha = .5, position = 'identity')
set_plot
ggsave('hist_verticalness.png', scale = 1, dpi = 400)


#Generate a Logistic Regression Model using glm function with the Training Data
log_model <- glm(classification ~ verticalness,
                 data = training_data,
                 family = 'binomial')
summary(log_model)

#Plot the fitted curve
x_range <- range(training_data[["verticalness"]])
x_axis <- seq(x_range[1], x_range[2], length.out = 1000)

fitted_curve <- data.frame(verticalness = x_axis)
fitted_curve[["classification"]] <- predict(log_model, fitted_curve, type = "response")

curve_plot <- ggplot(training_data, aes(x=verticalness, y=classification)) + 
  geom_point(aes(colour = factor(classification)), 
             show.legend = T, position="dodge")+
  geom_line(data=fitted_curve, colour="orange", size=1)
curve_plot
ggsave('log_model_fitted_curve.png', scale = 1, dpi = 400)
#Calculate Accuracy of Model using Training Data
training_data[["predicted_val"]] = predict(log_model, training_data, type="response")
training_data[["predicted_class"]] = 0
training_data[["predicted_class"]][training_data[["predicted_val"]] > 0.5] == 1

correct_predictions = training_data[["predicted_class"]] == training_data[["classification"]]

accuracy = nrow(training_data[correct_predictions,])/nrow(training_data)

#Calculate Accuracy of Model using Test Data
test_data[["predicted_val"]] = predict(log_model, test_data, type="response")
test_data[["predicted_class"]] = 0
test_data[["predicted_class"]][test_data[["predicted_val"]] > 0.5] == 1

correct_predictions = test_data[["predicted_class"]] == test_data[["classification"]]

accuracy = nrow(test_data[correct_predictions,])/nrow(test_data)

