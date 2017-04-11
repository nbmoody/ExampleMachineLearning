train <- read.csv("pml-training.csv")
test <- read.csv("pnl-training.csv")


# I need to exclude the predictors that are entirely NA values in the test set.

naCount <- vector()
for(i in 1:160) {
    
    
    naSum <- sum(is.na(train[,i]))
    
    naCount <- c(naCount, naSum)
    
}

sum(is.na(train[,149]))
