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