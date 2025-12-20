import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
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

def train_model(learning_rate, batch_size, dropout_rate):
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Flatten(),  # Remove heavy Conv2D layers!
        Dense(64, activation='relu'),  # Smaller layer
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
    # Keep same compile/fit code
    return max(history.history['val_accuracy'])

# Run all 8 experiments

with open('data_gen_replicate2V2.csv', 'a', newline='') as file:
    firstLine = "learning_rate, batch_size, dropout_rate,accuracy\n"
    file.write(firstLine)
    for lr in factors['learning_rate']:
        for bs in factors['batch_size']:
            for dr in factors['dropout_rate']:
                print(f"Training with LR={lr}, BS={bs}, DR={dr}")
                response = train_model(lr, bs, dr)
                newline = f"{lr},{bs},{dr},{response:.4f}\n"
                file.write(newline)