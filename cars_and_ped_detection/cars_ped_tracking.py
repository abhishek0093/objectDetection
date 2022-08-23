'''
we are again using pre-trained datasets/model from the openCv library. 
'''

import cv2

'''
#Our Car Image
img_file = 'cars_Img.jpg'
#path to our training file
path = 'cars_train.xml'

#Our cv image
cv_img = cv2.imread(img_file)
#converting to B/w (for haar features detection)
gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
#create model for car classification
car_detector_model = cv2.CascadeClassifier('cars_train.xml')

#make model to detect cars 
detected_cars = car_detector_model.detectMultiScale(gray_img)       #detectMultiscale -> detect cars of all scale, wheteher smalls or large

#Now our detected cars contains cordinates [x,y,h,w] of the car location. 
#(x,y) location of upper-left-corner and h = ht and w = width of rectangle
for (x,y,h,w) in detected_cars:
    cv2.rectangle(cv_img, (x,y), (x+w, y+h), (0,0,255), 3)

cv2.imshow('This is the image that comupter sees',cv_img)

#Don't autoclose, wait for some key to be pressed
cv2.waitKey(0)
'''


#Our Car Image and pedastrian Image
img_file = 'cars_peds_Img.jpg'

#path to our training file
path_cars = 'cars_train.xml'
path_peds = 'peds_train.xml'


#Our cv image
cv_img = cv2.imread(img_file)
#converting to B/w (for haar features detection)
gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
#create model for car classification
car_detector_model = cv2.CascadeClassifier(path_cars)
ped_detector_model = cv2.CascadeClassifier(path_peds)

#make model to detect cars 
#detectMultiscale -> detect cars of all scale, wheteher smalls or large
detected_cars = car_detector_model.detectMultiScale(gray_img)      
detected_peds = ped_detector_model.detectMultiScale(gray_img)       

#Now our detected cars/peds contains cordinates [x,y,h,w] of the car/peds location. 
#(x,y) location of upper-left-corner and h = ht and w = width of rectangle
for (x,y,h,w) in detected_cars:
    cv2.rectangle(cv_img, (x,y), (x+w, y+h), (0,0,255), 3)

for (x,y,h,w) in detected_peds:
    cv2.rectangle(cv_img, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('peds_and_cars',cv_img)

#Don't autoclose, wait for some key to be pressed
cv2.waitKey(0)



print("-------------ENDED------------")