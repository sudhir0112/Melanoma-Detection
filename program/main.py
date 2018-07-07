'''This is the main program of the project it drives all the other codes. This program
Loads the machine learning model then the image and predicts the result and the probability'''
import cv2                         #For image processing
import sys                         #To exit from code when needed
import os                          #For setting paths

import imageProcessing as IP       #For image processing
import values as val               #For Loading default values
import mlModel as ML               #For Loading machine Learning Model
import predictResults as PR        #To predict Results



#Open Image, if image not opened rise error
os.chdir(val.inputPath)
try:
    image=cv2.imread(val.imageName)
except:
    print 'Cannot Open Image!!!'
    sys.exit(0)


#Display Original Image
cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
cv2.imshow("Original Image",image)

'''Load ML Model'''
model=ML.LoadTrainModel()

'''Predict Results'''
result,probability,distance=PR.predict(image,model)
imageDisplay=image.copy()

#Display results on screen
print'Prediction of the Model:'+str(result)
print 'Probability:'+str(probability)
print 'Distance:'+str(distance)


#Display the results on the image
cv2.putText(imageDisplay,result.title(),(50,50),1,2,val.textColor,2)
cv2.putText(imageDisplay,'Probability: '+str(probability)+'%',(50,85),1,2,val.textColor,2)
cv2.putText(imageDisplay,'Distance: '+str(distance),(50,115),1,2,val.textColor,2)

#display processed image and save the image in output folder
os.chdir(val.outputPath)
cv2.imwrite(val.imageName,imageDisplay)
cv2.namedWindow("Result",cv2.WINDOW_AUTOSIZE)
cv2.imshow("Result",imageDisplay)

cv2.waitKey(0)
cv2.destroyAllWindows()

