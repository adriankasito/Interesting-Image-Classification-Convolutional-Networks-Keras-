# Neural Network Image Classification with TensorFlow and Keras

This repository contains a simple image classification project using TensorFlow and Keras. The neural network is designed to classify images into different categories such as buildings, forests, glaciers, mountains, seas, and streets.

## Prerequisites

Make sure you have the following libraries installed:

- TensorFlow
- Keras
- OpenCV
- Matplotlib
- NumPy
- Scikit-learn
- IPython

You can install the required packages using the following:

```bash
pip install tensorflow keras opencv-python matplotlib numpy scikit-learn
```

## Dataset

The dataset used for this project is located in the `./adrian/seg_train/seg_train/` directory. It contains images categorized into different classes. The classes and their corresponding labels are as follows:

- 0: Buildings
- 1: Forests
- 2: Glaciers
- 3: Mountains
- 4: Seas
- 5: Streets

## Code Structure

- `get_images(directory)`: Function to load and preprocess images from the specified directory.
- `get_classlabel(class_code)`: Function to get the class label based on the class code.
- Neural Network Model: A convolutional neural network model is implemented using TensorFlow and Keras.
- Model Training: The model is compiled and trained using the Adam optimizer and sparse categorical crossentropy loss.
- Model Evaluation: The model's accuracy and loss are visualized using Matplotlib.
- Prediction Visualization: Random images from the dataset are selected for visualization of model predictions.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Interesting-Image-Classification-Convolutional-Networks-Keras-.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Interesting-Image-Classification-Convolutional-Networks-Keras-
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook or Python script to train the model and visualize the results.

   ```bash
   jupyter notebook Intel image Classification....ipynb
   ```

   or

   ```bash
   python Intel image Classification....py
   ```

Feel free to customize the code, experiment with different hyperparameters, or use your own dataset for image classification.

