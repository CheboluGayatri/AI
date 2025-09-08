import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Shortcuts for readability
datasets = keras.datasets
layers = keras.layers
models = keras.models
to_categorical = keras.utils.to_categorical

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values (0–1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape to (28, 28, 1) because images are grayscale
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# One-hot encode the labels
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')   # 10 digits: 0–9
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels,
                    epochs=5,
                    batch_size=64,
                    validation_data=(test_images, test_labels))

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Make a prediction on the first test image
predictions = model.predict(test_images)
predicted_label = np.argmax(predictions[0])
print(f"Prediction for first test image: {predicted_label}")

# Plot first test image with prediction
plt.imshow(test_images[0].reshape(28, 28), cmap="gray")
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()
