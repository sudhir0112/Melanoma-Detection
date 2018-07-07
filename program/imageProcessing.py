import cv2
import values as val

#Preprocessing of image is define here
def preprocess(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #Resize the image to the default size
    gray = cv2.resize(gray, (val.imageWidth, val.imageHeight))
    return gray


'''SEGMENTATION PROCESS AND FEATURE EXTRACTION'''

#Otsu For segmentation,this functions takes gray image as input and returns binay image
def Otsu(gray):
    t,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    thresh=cv2.erode(thresh,(5,5),iterations=3)

    m2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 400
    targetCnt = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            targetCnt = cnt

    if len(targetCnt)!=0:
        x, y, w, h = cv2.boundingRect(targetCnt)
        thresh = thresh[y:y + h, x:x + w]
        gray=gray[y:y + h, x:x + w]
    imageProcessed = cv2.bitwise_and(gray, gray, mask=thresh)
    return imageProcessed