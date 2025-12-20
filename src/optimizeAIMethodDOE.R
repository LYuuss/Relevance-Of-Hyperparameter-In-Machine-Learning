library(dplyr)
library(ggplot2)
library(caret)

data_gen <- read.csv("data/data_gen_randomized.csv")

head(data_gen)
summary(data_gen)

cor_matrix <- cor(data_gen[, c("learning_rate", "batch_size", "dropout_rate", "accuracy")])
print(cor_matrix)

model <- lm(
    accuracy ~ learning_rate + batch_size + dropout_rate + learning_rate*batch_size*dropout_rate,
    data = data_gen
    )
    summary(model)


model_anova <- aov(accuracy ~ learning_rate * batch_size * dropout_rate, 
            data = data_gen)
summary(model_anova)



# Center and scale factors to -1/+1
data_gen <- data_gen %>%
  mutate(
    learning_rate_coded = (learning_rate - (0.01 + 0.001)/2) / ((0.01 - 0.001)/2),
    batch_size_coded = (batch_size - (128 + 32)/2) / ((128 - 32)/2),
    dropout_rate_coded = (dropout_rate - (0.5 + 0.2)/2) / ((0.5 - 0.2)/2)
  )

# Verify coding (should give -1 and +1 for min/max)
summary(data_gen[, c("learning_rate_coded", "batch_size_coded", "dropout_rate_coded")])

# Fit model with coded variables
model <- lm(
  accuracy ~ learning_rate_coded + batch_size_coded + dropout_rate_coded +
    learning_rate_coded * batch_size_coded * dropout_rate_coded,
  data = data_gen
)
summary(model)
