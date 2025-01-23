# Neural-Nets-IBM-PowerAI-NIMBIX-PowerAI-Vision-on-NIMBIX-to-train-Cat-Dog-Classifier-Neural-Net
To build and train a Cat/Dog Classifier Neural Network using IBM PowerAI Vision and NIMBIX, you will need to leverage the GPU capabilities provided by NIMBIX for faster model training. IBM PowerAI Vision is a platform specifically designed for computer vision tasks, including training deep learning models.

In this example, we'll train a Cat/Dog Classifier using TensorFlow and Keras, and we will assume that the IBM PowerAI Vision and NIMBIX are used to provide GPU acceleration and cloud-based services for training.
Step-by-Step Guide:

    Set up your dataset: You need a labeled dataset with images of cats and dogs. You can use the Kaggle Cats vs. Dogs dataset or any similar dataset that contains images of cats and dogs.

    Setup NIMBIX for GPU-accelerated training: NIMBIX provides GPU instances that are optimized for machine learning. You can set up a TensorFlow environment on NIMBIX.

    Use IBM PowerAI Vision: IBM PowerAI Vision is an optimized tool for accelerating the training of deep learning models using IBM's POWER architecture. We'll integrate this with TensorFlow to train the Cat/Dog classifier.

Step 1: Preparing the Dataset

Ensure your dataset is organized as follows:

data/
  train/
    cats/
      cat_001.jpg
      cat_002.jpg
      ...
    dogs/
      dog_001.jpg
      dog_002.jpg
      ...
  validation/
    cats/
      cat_001.jpg
      cat_002.jpg
      ...
    dogs/
      dog_001.jpg
      dog_002.jpg
      ...

This structure helps in easy loading of images using ImageDataGenerator.
Step 2: Set up the Environment on NIMBIX

    Sign in to NIMBIX: First, create an account or sign in to NIMBIX (https://www.nimbix.net/).

    Create a new job: Create a new job that provides a GPU-enabled machine instance. Use the following steps to configure a machine with TensorFlow and Keras preinstalled.

    Upload the Dataset: Upload your dataset (the train and validation folders) to the NIMBIX environment.

    Install Necessary Libraries: If needed, you can install TensorFlow, Keras, and other required libraries in the job environment:

    pip install tensorflow keras matplotlib numpy

Step 3: Build the Cat/Dog Classifier Model

Now we will use Keras (which is integrated with TensorFlow) to build the Cat/Dog classifier model.

Here is the complete Python code to train a neural network on the NIMBIX cloud environment:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define dataset directories
train_dir = 'data/train'  # Replace with your train directory
validation_dir = 'data/validation'  # Replace with your validation directory

# Define image size and batch size
img_size = (150, 150)  # Resize images to 150x150
batch_size = 32

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data should only rescale (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'  # Binary classification (Cat vs Dog)
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Build the Convolutional Neural Network (CNN) model
model = keras.Sequential([
    # First convolutional layer with max-pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    # Second convolutional layer with max-pooling
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Third convolutional layer with max-pooling
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Flatten the output for the dense layer
    layers.Flatten(),
    
    # Fully connected layer with 512 units
    layers.Dense(512, activation='relu'),
    
    # Output layer for binary classification (0 for cat, 1 for dog)
    layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model using the train and validation generators
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the trained model
model.save('cat_dog_classifier_model.h5')

# Plot training and validation accuracy/loss
def plot_history(history):
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

# Plot the results
plot_history(history)

Explanation of the Code:

    Image Data Generator: We use ImageDataGenerator to load and preprocess images for training and validation. The training data is augmented using techniques like rotation, shift, zoom, and horizontal flipping to improve model generalization.

    Model Architecture:
        The model is a simple CNN with 3 convolutional layers followed by max-pooling layers to reduce the spatial dimensions.
        After flattening, we add a dense layer with 512 neurons followed by an output layer with a sigmoid activation function for binary classification (cat vs. dog).

    Model Compilation:
        The model uses binary cross-entropy loss because it's a binary classification task.
        We use the Adam optimizer for faster convergence.

    Training:
        The model is trained for 10 epochs with the training data generated from the train_generator and validated using the validation_generator.

    Model Saving: After training, the model is saved as cat_dog_classifier_model.h5.

    Plotting: The training and validation accuracy and loss are plotted using Matplotlib to visualize the model's performance.

Step 4: Run the Model on NIMBIX

    Upload the Code and Dataset: Upload the Python script and your dataset to NIMBIX.

    Select GPU Instance: Choose a GPU-enabled instance in NIMBIX to accelerate the training. NIMBIX supports various instances, including those with NVIDIA GPUs like Tesla V100, A100, or T4 for deep learning tasks.

    Run the Code: Run the script using NIMBIX's environment. It will automatically utilize GPU acceleration to speed up the model training.

Step 5: Predicting Using the Trained Model

Once the model is trained, you can use it to classify new images of cats and dogs.

from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = keras.models.load_model('cat_dog_classifier_model.h5')

# Load and preprocess a new image for prediction
img_path = 'path_to_new_image.jpg'  # Replace with your test image path
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the class (0 = Cat, 1 = Dog)
prediction = model.predict(img_array)

if prediction[0] > 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")

Conclusion

This example shows how to train a Cat/Dog Classifier neural network using IBM PowerAI Vision and NIMBIX for GPU-accelerated deep learning tasks. The model is built using Keras (with TensorFlow backend) and trained on a GPU instance to significantly speed up the training process. Once the model is trained, it can be deployed to classify new images of cats and dogs.
