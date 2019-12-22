library(ggplot2)

datapath <- "C:\\Users\\spark\\Documents\\AI-DA_3\\src\\training_data\\features"
feature_col_names <- c("label", "index", "nr_pix", "height", "width", "span", "rows_wth_5", "cols_with_5", "neigh1", "neigh5", 
                       "left2tile", "right2tile", "verticalness", "top2tile", "bottom2tile", "horizontalness", "horizontal3tile", "vertical3tile", 
                       "nr_regions", "nr_eyes", "hollowness", "image_fill")

set.seed(3060)

#Load Training Data from files
csv_files <- list.files(path = datapath, pattern = ".csv", 
                                full.names = TRUE,
                                recursive = FALSE)
test <- lapply(csv_files, function(x) {
    csv_features <- read.csv(x, sep = "\t", header = FALSE, col.names = feature_col_names)
})

#Load Training Data into dataframe
training_data <- data.frame(matrix(ncol = 22, nrow = 0))
training_data <- do.call(rbind, test)
colnames(training_data) <- feature_col_names

#2.1 KNN for features 1-8
library(class)


training_sample <- training_data[sample(nrow(training_data)),]
train_size <- 0.8 * nrow(training_sample)

train_x <- training_sample[1:train_size,]
test_x <- training_sample[(train_size+1):nrow(training_sample),]
cl <- train_x$label
odd_k <- c()
knn_acc <- c()
for (n in 1:59)
{
  if (n %% 2 == 1)
  {
    knn_pred <- knn(train_x[,3:10], 
                    test_x[3:10],
                    cl,
                    k = n)
    odd_k <- c(odd_k, n)
    accuracy <- mean(knn_pred == training_sample$label)
    knn_acc <- c(knn_acc, 1-accuracy)
  }
}
knn_table <- data.frame("k"= odd_k, "error.rate" = knn_acc)


#2.2 KNN using 5 fold Cross validation for same 8 features
kfolds = 5
cl_fs <- colnames(training_sample)[3:10]
cv_err <- c()
inv_k <-c()
for (n in 1:59)
{
  if (n %% 2 == 1)
  {
    training_sample$folds <- cut(seq(1,nrow(training_sample)), breaks = kfolds, labels = FALSE)
    cv_accuracy = 0
    for (i in 1:kfolds)
    {
      fold_train_items = training_sample[training_sample$folds != i,]
      fold_validation_items = training_sample[training_sample$folds == i,]
      
      knn_pred = knn(fold_train_items[,cl_fs],
                     fold_validation_items[,cl_fs],
                     fold_train_items$label,
                     k = n)
      correct_items = nrow(training_sample[knn_pred == training_sample$label,])
      test_accuracy = correct_items/nrow(training_sample)
      
      cv_accuracy = cv_accuracy + test_accuracy
    }
    cv_accuracy = cv_accuracy/kfolds
    cv_err <- c(cv_err, 1-cv_accuracy)
    inv_k <- c(inv_k, 1/n)
  }
}

knn_table <- cbind(knn_table, cv_err, inv_k)
colnames(knn_table)[3] <- "cv.error.rate"
colnames(knn_table)[4] <- "1.div.k"

k_ticks <- c(0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
err_plt <- ggplot(knn_table, aes(x = knn_table$`1.div.k`),
                  position_dodge()) +
  geom_line(aes(y = knn_table$cv.error.rate),
            colour = "red",
            lwd =1.25,
            lty = "longdash") +
  geom_line(aes(y = knn_table$error.rate),
            colour = "blue",
            lwd = 1.25,
            lty = "longdash") +  
  xlab("1/K") +
  ylab("Error Rate") + 
  geom_point(aes(y = knn_table$cv.error.rate, fill = "red"),
             col = "black",
             pch = 21,
             size = 2.5) +
  geom_point(aes(y = knn_table$error.rate, fill = "blue"),
             col = "black",
             pch = 21,
             size = 2.5) +
  scale_x_log10(breaks = k_ticks) +
  scale_fill_discrete(name = "",
                      breaks = c("red", "blue"),
                      labels = c("Cross-Val", "Original"))
err_plt

