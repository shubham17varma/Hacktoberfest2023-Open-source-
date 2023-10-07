import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(input_dim,), name='input_layer'),
    layers.Dense(128, activation='relu', name='hidden_layer1'),
    layers.Dense(64, activation='relu', name='hidden_layer2'),
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_val, y_val))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
