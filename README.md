# Visual Tag Recognition

This project focuses on visual tag recognition using computer vision techniques. It involves detecting and recognizing tags in images using the AprilTag library and OpenCV. The project is divided into two parts: data preprocessing and building a neural network for tag recognition.

## Part 1: Data Preprocessing

In the first part, the following steps are performed:

- Installing the necessary packages: apriltag and OpenCV.
- Importing the required libraries: cv2, apriltag, matplotlib, imutils, pandas, and os.
- Connecting to a Google Drive account to access the dataset of images.
- Exploring and understanding the data: obtaining the paths of the images.

## Part 2: Building a Neural Network for Tag Recognition

In the second part, the focus is on building a neural network for tag recognition. This involves the following steps:

- Importing the necessary libraries: TensorFlow, Keras, numpy, and sklearn.
- Preparing the dataset: resizing the images, extracting features using the AprilTag library, and splitting the data into training and testing sets.
- Building the neural network model: defining the architecture, compiling the model, and training it on the training data.
- Evaluating the model: assessing its performance on the testing data and visualizing the results.

The project aims to provide an automated solution for visual tag recognition, which can be applied in various domains such as robotics, augmented reality, and object tracking.

## Requirements

To run the project, the following packages need to be installed:

- apriltag
- opencv-python
- TensorFlow
- Keras
- numpy
- sklearn

## Usage

1. Install the required packages mentioned in the requirements section.
2. Clone the project repository.
3. Run the Jupyter notebooks in the given order: Part_1_Pre_processing_Data_.ipynb and Part_2_Tag_Recognition.ipynb.
4. Follow the instructions and comments in the notebooks to preprocess the data and build the neural network model.
5. Analyze the results and make any necessary modifications to improve the performance.

## Contributors

- Tariq CHELLALI

## License

This project is licensed under the [MIT License](LICENSE).