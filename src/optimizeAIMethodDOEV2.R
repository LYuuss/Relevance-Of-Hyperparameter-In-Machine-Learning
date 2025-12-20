library(dplyr)
library(ggplot2)
library(caret)

data_gen <- read.csv("data/data_gen_replicate1V2.csv")

head(data_gen)
summary(data_gen)

cor_matrix <- cor(data_gen[, c("learning_rate", "batch_size", "dropout_rate", "accuracy")])
print(cor_matrix)

model <- lm(
    accuracy ~ learning_rate + batch_size + dropout_rate,
    data = data_gen
    )
    summary(model)
model_anova <- aov(accuracy ~ learning_rate * batch_size * dropout_rate, 
            data = data_gen)
summary(model_anova)


pdf("AIV2.pdf", width=8, height=6) 

par(mfrow = c(2, 2)) 
plot(model) 

dev.off()
