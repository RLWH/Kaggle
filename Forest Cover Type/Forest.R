library(dyplr)
library(tidyr)
library(ggplot2)

#Data import and exploration
data <- read.csv("train.csv")
summary(data)
str(data)
head(data)
tail(data)

#Data cleaning process
data$Cover_Type <- as.factor(data$Cover_Type)
summary(data)
str(data)

#Visualise Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_to_Hydrology w.r.t Tree Type

# ggplot(data, aes(x=Aspect)) + geom_histogram() + facet_grid(. ~ Cover_Type)
# ggplot(data, aes(x=Slope)) + geom_histogram() + facet_grid(. ~ Cover_Type)

ggplot(data, aes(x=Aspect, ..count.., col=Cover_Type)) + geom_density()
ggplot(data, aes(x=Slope, ..count.., col=Cover_Type)) + geom_density()
ggplot(data, aes(x=Horizontal_Distance_To_Hydrology, ..count.., col=Cover_Type)) + geom_density()
ggplot(data, aes(x=Vertical_Distance_To_Hydrology, ..count.., col=Cover_Type)) + geom_density()
ggplot(data, aes(x=Horizontal_Distance_To_Roadways, ..count.., col=Cover_Type)) + geom_density()
ggplot(data, aes(x=Hillshade_9am, ..count.., col=Cover_Type)) + geom_density()
ggplot(data, aes(x=Hillshade_Noon, ..count.., col=Cover_Type)) + geom_density()
ggplot(data, aes(x=Hillshade_3pm, ..count.., col=Cover_Type)) + geom_density()
ggplot(data, aes(x=Horizontal_Distance_To_Fire_Points, ..count.., col=Cover_Type)) + geom_density()

table(data$Cover_Type, data$Wilderness_Area1)
table(data$Cover_Type, data$Wilderness_Area2)
table(data$Cover_Type, data$Wilderness_Area3)

#As soil type is mutually exclusive, below scripts combine all dummy variables of soil type to a categorical variable
Soil_Type <- colnames(data)[16:55][max.col(data[16:55]==1)]
data$Soil_Type<-as.numeric(gsub("Soil_Type","",Soil_Type))

result <- sapply(data[16:55], function(x) {table(data$Cover_Type, x)[,1]})
result <- 2160-result
result <- as.data.frame(result)

#Tidy up result data frame
result <- result %>% 
  gather("Soil_Type") %>%
  mutate(Cover_Type=rep(1:7,40)) %>%
  spread(Cover_Type,value) %>%
  mutate(sum = rowSums(.[2:8]))

# ggplot(result, aes(x=Cover_Type,y=value)) + geom_bar(stat="identity") + facet_wrap(. ~ Soil_Type, ncol=4)
# ggplot(result, aes(x=Cover_Type, y=value, col=Soil_Type)) + geom_line()

#Feature selection
drops <- c("Id", paste0("Soil_Type", c(1:40)))
feature <- data[ , !(names(data) %in% drops)]

#Randomly split training and testing set for cross-validation

set <- 5
n <- nrow(feature)

training_sets <- list()
testing_sets <- list()
forests <- list()
predicts <- list()

#Shuffle the dataset
shuffle <- feature[sample(n),]

for (i in 1:set) {
  index <- (c((i-1) * round((1/set)*n)+1):(i*round((1/set)*n)))
  training_sets[[i]] <- shuffle[-index,]
  testing_sets[[i]] <- shuffle[index,]
  
  #Run randomForest
  forests[[i]] <- randomForest(Cover_Type ~ ., data=training_sets[[i]], ntree=1000, importance=TRUE)
  
  #Run predictions
  predicts[[i]] <- predict(forests[[i]], testing_sets[[i]])
  
}
#Evaluation
for (i in 1:set) {
  evaluation <- data.frame(Test=testing_sets[[i]]$Cover_Type, Prediction=predicts[[i]])
  evaluation$Correct <- evaluation$Test==evaluation$Prediction
  acc <- sum(evaluation$Correct)/nrow(evaluation)
  print(paste0("Model ",i," : ", acc))
}


#True testing test
test <- read.csv("test.csv")
Soil_Type_Test <- colnames(test)[16:55][max.col(test[16:55]==1)]
test$Soil_Type<-as.numeric(gsub("Soil_Type","",Soil_Type_Test))

predict_final <- predict(forests[[1]], test)

#Product the result
ans <- data.frame(Id=test$Id,Cover_Type=predict_final)
write.csv(ans,file="answer.csv",row.names=FALSE)
