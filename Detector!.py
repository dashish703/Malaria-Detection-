# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
# Assuming dataset folder has 'Parasitized' and 'Uninfected' subfolders with respective images
dataset_dir = "path_to_your_dataset"

# Image preprocessing using ImageDataGenerator for data augmentation
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training and validation datasets
train_data = data_gen.flow_from_directory(dataset_dir, target_size=(64, 64), batch_size=32, class_mode='binary', subset='training')
validation_data = data_gen.flow_from_directory(dataset_dir, target_size=(64, 64), batch_size=32, class_mode='binary', subset='validation')

# Define the CNN model
model = Sequential()

# Add convolutional and pooling layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers and adding dense layers
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification (infected/uninfected)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model architecture
model.summary()

# Train the model
history = model.fit(train_data, epochs=10, validation_data=validation_data)

# Save the model for future use
model.save('malaria_detection_model.h5')

# Plot training and validation accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
validation_data.reset()
predictions = model.predict(validation_data, verbose=1)
y_pred = np.where(predictions > 0.5, 1, 0)
y_true = validation_data.classes

# Display classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Uninfected', 'Parasitized']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Example: Load a single image for prediction
from tensorflow.keras.preprocessing import image

def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    if prediction > 0.5:
        return "Parasitized"
    else:
        return "Uninfected"

# Predict on a new image
image_path = "path_to_single_image"
result = predict_image(image_path, model)
print(f"Prediction for the given image: {result}")
