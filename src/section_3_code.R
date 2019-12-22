
library(ggplot2)
library(class)

set.seed(3060)

datapath <- "C:\\Users\\spark\\Documents\\AI-DA_3\\src\\training_data\\features"
feature_col_names <- c("label", "index", "nr_pix", "height", "width", "span", "rows_wth_5", "cols_with_5", "neigh1", "neigh5", 
                       "left2tile", "right2tile", "verticalness", "top2tile", "bottom2tile", "horizontalness", "horizontal3tile", "vertical3tile", 
                       "nr_regions", "nr_eyes", "hollowness", "image_fill")

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

#3.1

train_sample <- training_data[sample(nrow(training_data)),]

