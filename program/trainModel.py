'''This program is used to train the machine learning model with the images
in the dataset folder. It is one time run process after that a model file is created
and stored in the model folder'''
from sklearn.neighbors import KNeighborsClassifier  #For ML model
from sklearn.externals import joblib                #For storing ML model
from skimage import feature                         #For calculating HOF features
from imutils import paths                           #For iterating images

import cv2                                          #For images processing purpose
import time                                         #To detect the time taken by the process
import os                                           #For setting paths

import values as val                                #To load default values
import imageProcessing as IP                        #For image processing


'''This Program initiates the training process'''
def trainModel():
    print 'Initiating Training Model....'
    print'Image Height: ' + str(val.imageHeight)
    print'Image Width: ' + str(val.imageWidth)
    t = time.time()

    train_image_list = []
    train_label_list = []

    for imagePath in paths.list_images(val.datasetPath):
        # the label for the image the second last data will be the image folder name
        label=imagePath.split('/')[-2]
        print label
        print 'Processing Image : ' +str(imagePath)

        #Load and convert image to gray scale, then resize it so that every image is of same size
        image=cv2.imread(imagePath)
        gray=IP.preprocess(image)

        #Calculate the HOG feature from the image
        H=feature.hog(gray,orientations=9,pixels_per_cell=(8,8),
                         cells_per_block=(2,2),transform_sqrt=True)

        #Append the features and the labels in a list
        train_image_list.append(H)
        train_label_list.append(label)


    # "train" the nearest neighbors classifier
    print "[INFO] training classifier..."
    model = KNeighborsClassifier(n_neighbors=val.nearrestNeighbors,weights='distance')
    model.fit(train_image_list, train_label_list)

    # Save Model
    print"[INFO] saving ML Model as "+str(val.modelName)
    os.chdir(val.modelPath)
    joblib.dump(model,val.modelName)

    #Calculate the Model informations
    modelSize = os.stat(val.modelName).st_size / (1024 * 1024)
    print 'Model Name: ' + val.modelName
    print 'Model Size: ' + str(modelSize) + ' MB'
    print 'Model Training Completed....'
    print 'Time Taken: ' + str(time.time() - t) + ' sec'

trainModel()