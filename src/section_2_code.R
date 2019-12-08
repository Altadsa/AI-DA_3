install.packages('ggplot2')
library(ggplot2)

datapath <- "C:\\Users\\spark\\Documents\\AI-DA_3\\src\\training_data\\features"
feature_col_names <- c("label", "index", "nr_pix", "height", "width", "span", "rows_wth_5", "cols_with_5", "neigh1", "neigh5", 
                       "left2tile", "right2tile", "verticalness", "top2tile", "bottom2tile", "horizontalness", "horizontal3tile", "vertical3tile", 
                       "nr_regions", "nr_eyes", "hollowness", "image_fill")
training_features <- list.files(path = datapath, pattern = ".csv", 
                                full.names = TRUE,
                                recursive = FALSE)
training_data <- data.frame(matrix(ncol = 22, nrow = 0))
colnames(training_data) <- feature_col_names
test <- lapply(training_features, function(x) {
    csv_features <- read.csv(x, sep = "\t", header = FALSE, col.names = feature_col_names)
    #rbind(training_data, data.frame(csv_features[1,]))
    #print(csv_features)
})
training_data <- do.call(rbind, test)
