# Design of Experiment
## Problem Description
For several years, AI has been increasingly present in our daily lives, whether for or against our will. From medical field to recommendation algorithm, machine learning has established himself as the most powerful tool, so it's important to understand how it works. We will use a specific model called CNN, meaning Convolutional Neural Network, that basically tries to copy the human way of recognizing images, namely by recurring patterns.
Our goal today will be to find the parameter values that will produce the best model, meaning the one with the best accuracy. To do this, we will tweak this parameters that will be our 3 factors for the experiment:
- **learning rate**: it influences how fast the model will try to converge toward its goal
- **batch size**: it's the number of samples that gets treated before potential correction
- **dropout rate**: it influences how frequent we will use a dropout technique, meaning to randomly disable a fraction of neurons during training iteration

We will perform our tests on the most famous training in machine learning, which is **Mnist**, a large database of handwritten digits. The goal of our model will be to predict to which digit correspond a certain picture. To do so, we will run multiples test by tweaking our factors to the following values:

||High|Low|
|--|:--:|:--:|
|learning rate|0.01|0.001|
|batch size|128|52|
|dropout rate|0.5|0.2|

We chose those levels for multiple reasons:
- it covers a meaningful range of values seen in practice
- it ensures a balance between exploration and computational feasibility
- it matches common hyperparameter tuning ranges used in machine learning

We completly control the level of the factors because we give them to the program before it's executed.
The response will be the percentage of accuracy our model got on some new data. The measurement will be accurate as it's given by the programm, and will not vary much if we rerun it.

## Problem Implementation
As we have 3 2-level factors, we will use the $2^k$ factorial design, that will be run 2 times to get 16 observations. We will not use a blocked design as it's not necessary.

We have randomized our factor values, which gives this following order and output:
|run|learning rate|batch size|dropout rate|accuracy|
|:--:|:--:|:--:|:--:|:--:|
|1 |0.001|32 |0.2|0.9895
|2 |0.001|128|0.2|0.9887
|3 |0.01 |128|0.2|0.9868
|4 |0.01 |32 |0.2|0.9181
|5 |0.001|32 |0.2|0.9898
|6 |0.001|128|0.2|0.9891
|7 |0.01 |32 |0.5|0.9806
|8 |0.001|128|0.5|0.9892
|9 |0.01 |128|0.5|0.9855
|10|0.001|32 |0.5|0.9893
|11|0.01 |128|0.5|0.9862
|12|0.001|128|0.5|0.9884
|13|0.01 |32 |0.5|0.9811
|14|0.01 |128|0.2|0.9864
|15|0.001|32 |0.5|0.9894
|16|0.01 |32 |0.2|0.9829

Each experiment is a genuine run replicate as we run the programm with a new seed and a new order of the datas everytime

### Main Effects
![main effects](images/main_effects.png)

### Interactions
![interactions](images/interactions.png)

### Linear Model
```R
                                        Estimate Std. Error  t value Pr(>|t|)    
(Intercept)                            9.910e-01  8.449e-04 1172.960  < 2e-16 ***
learning_rate                         -8.938e-01  1.189e-01   -7.518 6.81e-05 ***
batch_size                            -1.456e-05  9.056e-06   -1.608  0.14646    
dropout_rate                          -7.099e-04  2.219e-03   -0.320  0.75721    
learning_rate:batch_size               5.363e-03  1.274e-03    4.208  0.00296 ** 
learning_rate:dropout_rate            -5.123e-01  3.122e-01   -1.641  0.13944    
batch_size:dropout_rate                4.823e-06  2.378e-05    0.203  0.84437    
learning_rate:batch_size:dropout_rate  2.122e-03  3.347e-03    0.634  0.54376    

Residual standard error: 0.0004337 on 8 degrees of freedom
Multiple R-squared:  0.9905,	Adjusted R-squared:  0.9822 
F-statistic: 119.5 on 7 and 8 DF,  p-value: 1.856e-07
```

### Anova Analysis
```R
                                      Df     Sum Sq    Mean Sq  F value    Pr(>F)    
learning_rate                          1 1.1078e-04 1.1078e-04 588.8405 8.874e-09 ***
batch_size                             1 1.5801e-05 1.5801e-05  83.9900 1.622e-05 ***
dropout_rate                           1 1.7560e-06 1.7560e-06   9.3322   0.01570 *  
learning_rate:batch_size               1 2.7826e-05 2.7826e-05 147.9103 1.936e-06 ***
learning_rate:dropout_rate             1 8.5600e-07 8.5600e-07   4.5482   0.06552 .  
batch_size:dropout_rate                1 2.2600e-07 2.2600e-07   1.1993   0.30533    
learning_rate:batch_size:dropout_rate  1 7.6000e-08 7.6000e-08   0.4020   0.54376    
Residuals                              8 1.5050e-06 1.8800e-07                       
```

### Residuals
![residual fitted](images/Residual_fitted.png)

![q-q residuals](images/QQ_residuals.png)

![scale location](images/scale_location.png)

### Correlation Matrix
![correlation matrix](images/matrix_correlation.png)

## Data Analysis
### Main Effects
- `learning_rate`:
  - Has a strong negative effect, meaning that a higher learning rate significantly reduces accuracy
  - Has a very significant p-value, meaning that it has the highest impact on accuracy
- `batch_size`:
  - Increasing it from 32 to 128 slightly increases accuracy
  - Has a significant p-value, meaning that it has a significant effect
- `dropout_rate`:
  - Is not very significant, meaning changing it does not strongly affect accuracy
  - Has a weak statistical significiance

### Interaction
- `learning_rate` and `batch_size`: highly significant, meaning that the effect of the first one on accuracy is dependent on the second
- `learning_rate` and `dropout_rate`: weak effect
- others interactions are not significant

### ANOVA and R-Squared
- `F-statistic` confirms that the overall model is statistically significant
- `Multiple R-squared` and `Adjusted R-squared` are close, meaning that the model is strong

### Residual and Assumptions Check
- `Residual vs Fitted Plot`: the data is well distributed around 0, meaning that constant variance is satisfied
- `Q-Q Plot`: it follows a straight line meaning that normality of residuals is satisfied
- `Scale-Location Plot`: it shows a straight line with a dip at the end, meaning no major heteroscedascity issues
- `Correlation Matrix`: there are no strong multicollinearity issues

### Overall
- The model surperformed as we see the results, we do recommand to configure those parameters at their best level but, also be aware of ressources available. Indeed, as the parameters tend to their best level, the more computational power you need.

## Documentation
### Python Code
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np

# Load dataset (example: MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Define factors and levels
factors = {
    'learning_rate': [0.001, 0.01],  # -1, +1
    'batch_size': [32, 128],         # -1, +1
    'dropout_rate': [0.2, 0.5]       # -1, +1
}

# Function to train model with given parameters
def train_model(learning_rate, batch_size, dropout_rate):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=20,
        validation_split=0.2,
        verbose=0
    )
    return max(history.history['val_accuracy'])

# Run all 8 experiments

with open('data_gen_replicate2.csv', 'a', newline='') as file:
    firstLine = "learning_rate, batch_size, dropout_rate,accuracy\n"
    file.write(firstLine)
    for lr in factors['learning_rate']:
        for bs in factors['batch_size']:
            for dr in factors['dropout_rate']:
                print(f"Training with LR={lr}, BS={bs}, DR={dr}")
                response = train_model(lr, bs, dr)
                newline = f"{lr},{bs},{dr},{response:.4f}\n"
                file.write(newline)
```
### R Code
```R
library(dplyr)
library(ggplot2)
library(caret)
library(ggcorrplot)
library(FrF2)
library(forcats)

data <- read.csv("data_gen_randomized.csv")

# Correlation Matrix
cor_matrix <- cor(data[, c("learning_rate", "batch_size", "dropout_rate", "accuracy")])
ggcorrplot(cor_matrix, lab = TRUE, type = "lower", colors = c("red", "white", "blue"))

fit <- lm(accuracy ~ learning_rate*batch_size*dropout_rate, data = data)
summary(fit)

# Main effects
MEPlot(fit, abbrev=5, las=1)

# Interaction effects
IAPlot(fit, abbrev=5, las=1)

# Anova
anova(fit)

# Residual plots
plot(fit)
```
