install.packages('ggplot2')
library(ggplot2)

featurepath <- "C:\\Users\\spark\\Documents\\AI-DA_3\\src\\40178464_features.csv"
datapath <- "C:\\Users\\spark\\Documents\\AI-DA_3\\src\\training_data\\features"
feature_col_names <- c("label", "index", "nr_pix", "height", "width", "span", "rows_wth_5", "cols_with_5", "neigh1", "neigh5", 
                       "left2tile", "right2tile", "verticalness", "top2tile", "bottom2tile", "horizontalness", "horizontal3tile", "vertical3tile", 
                       "nr_regions", "nr_eyes", "hollowness", "image_fill")

#Load My Feature Data
f_data <- read.csv(featurepath, header = TRUE, sep = "\t", col.names = feature_col_names)
f_data$label <- tolower(as.factor(f_data$label))
#Load Training Data from files
csv_files <- list.files(path = datapath, pattern = ".csv", 
                                full.names = TRUE,
                                recursive = FALSE)
colnames(training_data) <- feature_col_names
test <- lapply(csv_files, function(x) {
    csv_features <- read.csv(x, sep = "\t", header = FALSE, col.names = feature_col_names)
    #rbind(training_data, data.frame(csv_features[1,]))
    #print(csv_features)
})

#Load Training Data into dataframe
training_data <- data.frame(matrix(ncol = 22, nrow = 0))
training_data <- do.call(rbind, test)
colnames(training_data) <- feature_col_names

set.seed(3060)

#2.1 KNN for features 1-8
library(class)

f_data <- f_data[sample(nrow(f_data)),]
# training_data <- training_data[sample(nrow(training_data)),]
train_size <- 0.8 * nrow(f_data)
test_size <- nrow(f_data)

train_x <- f_data[1:train_size,]
test_x <- f_data[(train_size+1):test_size,]
# train_x <- f_data
# test_x <- training_data
#N.B. When changing test data, the cl value must be taken from the test data
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
    knn_acc <- c(knn_acc, mean(knn_pred == training_data$label))
  }
}
knn_table <- data.frame("k"= odd_k, "cl error rate" = 1-knn_acc)


#2.2 KNN using 5 fold Cross validation for same 8 features
kfolds = 5
# training_data = training_data[sample(nrow(training_data)),]
f_data  = f_data[sample(nrow(f_data)),]
f_data$folds = cut(seq(1,nrow(f_data)), breaks = kfolds, labels = FALSE)

# training_data$folds = cut(seq(1,nrow(training_data)), breaks = kfolds, labels = FALSE)
#training_data$folds

# cl_fs <- colnames(training_data)[3:10]
cl_fs <- colnames(f_data)[3:10]

cv_err <- c()
inv_k <-c()
for (n in 1:59)
{
  if (n %% 2 == 1)
  {
    validation_error = 0
    for (i in 1:kfolds)
    {
      fold_train_items = f_data[f_data$folds != i,]
      fold_validation_items = f_data[f_data$folds == i,]
      
      knn_pred = knn(fold_train_items[,cl_fs],
                     fold_validation_items[,cl_fs],
                     fold_train_items$label,
                     k = n)
      correct_test_items = nrow(training_data[knn_pred == training_data$label,])
      test_accuracy = correct_test_items/nrow(training_data)
       
      # correct_train = knn_pred == fold_train_items$label
      # correct_train_items = nrow(fold_train_items[correct_train,])
      # train_accuracy = correct_train_items/nrow(fold_train_items)
      
      validation_error = validation_error + test_accuracy
    }
    validation_error = validation_error/kfolds
    cv_err <- c(cv_err, 1-validation_error)
    inv_k <- c(inv_k, 1/n)
    #knn_table = rbind(knn_table, c(round(1-validation_error,2), round(1/n,2)))
    #print(sprintf("Average Validation Error for K = %s: %s", n, validation_error))
    #print("----END----")
  }
}
knn_table <- cbind(knn_table, cv_err, inv_k)
colnames(knn_table)[3] <- "crv.cl.error.rate"
colnames(knn_table)[4] <- "1.div.k"
l_y_lim = min(min(knn_table$cl.error.rate), min(knn_table$crv.cl.error.rate))
u_y_lim = max(max(knn_table$cl.error.rate), max(knn_table$crv.cl.error.rate))
plot(knn_table$`1.div.k`, knn_table$crv.cl.error.rate, 
     xlim = c(0,1),
     ylim = c(l_y_lim, u_y_lim),
     xlab = "1/K",
     ylab = "Error Rate",
     xaxt = "n")
legend(locator(), legend = c("a", "b"),
       col = c("red", "blue"), lty = 1:2, cex = 0.8)
lines(knn_table$`1.div.k`, knn_table$crv.cl.error.rate,
      col = "red")
points(knn_table$`1.div.k`, knn_table$cl.error.rate)
lines(knn_table$`1.div.k`, knn_table$cl.error.rate,
      col = "blue")


line_col <- c("#ff0000", "#0000ff")
fill_col <- c("#ff00ff", "#00ffff")

k_ticks <- c(0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
err_plt <- ggplot(knn_table, aes(x = knn_table$`1.div.k`),
                  position_dodge()) +
  geom_line(aes(y = knn_table$crv.cl.error.rate),
            colour = "red",
            lwd =1.25,
            lty = "longdash") +
  geom_line(aes(y = knn_table$cl.error.rate),
            colour = "blue",
            lwd = 1.25,
            lty = "longdash") +  
  xlab("1/K") +
  ylab("Error Rate") + 
  geom_point(aes(y = knn_table$crv.cl.error.rate, fill = "red"),
             col = "black",
             pch = 21,
             size = 2.5) +
  geom_point(aes(y = knn_table$cl.error.rate, fill = "blue"),
             col = "black",
             pch = 21,
             size = 2.5) +
  scale_x_log10(breaks = k_ticks) +
  scale_fill_discrete(breaks = c("red", "blue"),
                      labels = c("Cross-Val", "Original"))
err_plt
?geom_line
