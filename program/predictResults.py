'''This Program is used to predict the Results and Probability'''
from skimage import feature         #For Hog feature extraction
import values as val                #For loading default values
import cv2                          #For image processing purpose
import imageProcessing as IP        #For image processing

'''This fuction predicts the result and the probability
It takes test image and ml model as input and returns result and probability 
as output'''
def predict(image, model):
    #Convert and Resize the image
    gray=IP.preprocess(image)

    #Calculate the HOG features from the image
    H=feature.hog(gray,orientations=9,pixels_per_cell=(8,8),
                         cells_per_block=(2,2),transform_sqrt=True)

    #Predict the results
    result = model.predict(H.reshape(1, -1))[0]

    #Calculate the probability
    probability=max(model.predict_proba(H.reshape(1, -1))[0])
    #Convert probability in percent
    probability=probability*100
    #Round Probability to 2 figures
    probability=round(probability,2)

    #Calculate the distance of the predicted image
    dist,ind=model.kneighbors(H.reshape(1, -1), n_neighbors=None, return_distance=True)

    #round the distance to 2 figures
    dist=round(dist[0][0],2)

    #Return Result and probability to the main function
    return result,probability,dist
