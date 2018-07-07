from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from skimage import feature
from skimage import exposure
from imutils import paths
import imutils
import values as val
import cv2
import os
import values as val

i=0

os.chdir(val.inputPath)
image=cv2.imread(val.imageName)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gray,(5,5),0)
#gray=cv2.equalizeHist(gray)
os.chdir(val.outputPath)

cv2.imwrite("blur.jpg",gray)
# cv2.imwrite("EqualizeHistogram.jpg",gray)
cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
cv2.imshow("Image",gray)
cv2.waitKey(0)



