# -*- coding: utf-8 -*-
"""
Created on December 14th, 2018

@author: Kevin Hou
"""
from sklearn.neural_network import MLPClassifier
from six.moves import urllib
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.io import loadmat
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import datetime
from sklearn.preprocessing import StandardScaler  

#####################################################################################################
###                          This function is to print the confusion matrix                       ###
#####################################################################################################
def plotConfusionMatrix(cm):
    plt.figure(figsize=(9,9))
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size = 15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
    plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
    plt.tight_layout()
    plt.ylabel('Actual label', size = 15)
    plt.xlabel('Predicted label', size = 15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',verticalalignment='center')
    plt.show()
    
#####################################################################################################
###                    Six Steps of Using Scikit-Learn for a Machine Learning Model               ###
#####################################################################################################

print('\n@@@@@@@@@@@@@@ Step 1: Load Data @@@@@@@@@@@@@@@')

#define the data path. We put it under the same folder where the Python code is located
mnist_path = './mnist-original.mat'

#download the data from website if it doesn't exist yet
if not exists(mnist_path):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    response = urllib.request.urlopen(mnist_alternative_url)
    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)
    print("######### Downloading dataset from source #########")
else:
    print("######### Dataset exists already - no re-downloading #########")
      
#load data from file into memory          
mnist_raw = loadmat(mnist_path)
mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original"
}   

# Print to show there are 70,000 images (28 by 28 images for a dimensionality of 784)
print("Image Data Shape" , mnist['data'].shape)

# Print to show there are 70,000 labels (integers from 0-9)
print("Label Data Shape", mnist['target'].shape)

print('\n@@@@@@@@@ Step 2: Split Data to Train/Test & standardize @@@@@@@@@@')
#data - both X (images) and Y (lables) are split inot train and test 
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist['data'], mnist['target'], test_size=1/7.0, random_state=0)

#print out the size and shapes of train/test images and labels
print("Train image shape: ", train_img.shape)
print("Train label shape: ", train_lbl.shape)
print("Test image shape: ", test_img.shape)
print("Test label shape: ", test_lbl.shape)


#print out a few images and lables in the training data set
startIndex = 0
numOfSampleImg = 8
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[startIndex:startIndex+numOfSampleImg], train_lbl[startIndex:startIndex+numOfSampleImg])):
    plt.subplot(1, numOfSampleImg, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.binary)
    plt.title('Training: %i\n' % label, fontsize = 18)
plt.show()    

#Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler.fit(train_img) 
train_img = scaler.transform(train_img) 
test_img = scaler.transform(test_img) 

print('\n@@@@@@@@@@@@@ Step 3: Instantiate a Scikit-Learn Model @@@@@@@@@@@@@')
clf = MLPClassifier(activation='relu', alpha=0.01, #logistic #relu
                    hidden_layer_sizes=(150, 150),
                    learning_rate='adaptive', learning_rate_init=0.001,
                    max_iter=500, random_state=1, shuffle=True)

print('\n@@@@@@@@@@@@@ Step 4: Train model @@@@@@@@@@@@@@@')
print(datetime.datetime.now())
# fitting the model
clf.fit(train_img, train_lbl)

print('\n@@@@@@@@@@@@@ Step 5: Predict with the trained model @@@@@@@@@@@@@@@')
print(datetime.datetime.now())
# predict the response
pred = clf.predict(test_img)

print('\n@@@@@@@@@@@@@ Step 6: Evaluate model performance @@@@@@@@@@@@@@@')

print('\n@@@@@@@@@@@@@ score @@@@@@@@@@@@@@')
print(datetime.datetime.now())
print (accuracy_score(test_lbl, pred))

print('\n@@@@@@@@@@@@@ confusion matrix @@@@@@@@@@@@@@')
print(datetime.datetime.now())
cm = metrics.confusion_matrix(test_lbl, pred)
plotConfusionMatrix(cm)
