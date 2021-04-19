Project: AnimalClassification
In this project I'm going to classify 8 different categories of animals.
Dataset: The dataset used for this task was collected from Kaggle, named Animal 10 Dataset. It consists of images of ten different types of animals. The dataset does not consist of equal number of images of each type.
The categories that we have chosen here are Dogs, Chicken, Squirrel, Horse, Cat and Butterfly. The dataset also contains so random unrelated images.
Libraries used: TensorFlow API, Keras, NumPy, Pandas, matplotlib.pyplot, cv2, classification_report

Model built consists of 4 Conv2D layers, 2 MaxPooling layers, 5 Dropout layers. We have implemented Batch normalization technique. Loss functions used is categorical cross entropy.
