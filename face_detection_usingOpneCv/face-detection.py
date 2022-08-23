#we will make all images to grayscal (black & White), as image and face features don't depend upon color. Mored epends upon face features.
import cv2

# using trained image dataset from the offical openCV site. - Dataset used openCV github page / Data / harsaacodes / 
#the dataset is already trained. We are using it .

"""
UNDERSTANDING ALGORITHM

HarrCasacde - Harr is just scientist name, cascde- The main chain funnel thing, which passess every square in to our model, and we get to see
Haar Features(rudimentary building bloacks), builds relationship. It tries to represent faces as a combination of basic square blocks(3 types each of b/w color), adn putting them layer by layer to match the face feature. Like Lego session
Here it sees only darkenss , and try to put black and white in a manner such that it appropriately represents face color portion (and this is the reason why we used grayscale, we only detecting darkness and brithness area in amage and trying to fit our square combinations in it.).
We scan putting our haar feature,and matching for every size of haar feature (block), every location of haar feature and this haar feature will return that wheter it is an image or not.(like ki % match, and we set threshold to classify as an image

YOUTUBE VISULAIZATION : https://www.youtube.com/watch?v=hPCTwxF0qf4

"""

# open the trained data with cascadeclassifier algorithm -a sort of detector algorithm. 
#can detect sort of anything with this classifier, but here training on faces.
t_FaceData = cv2.CascadeClassifier('trainedData.xml')

'''
#using face detection for an image.

#open the image 
img = cv2.imread('photo.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#make algoritm to detect the face using that trained data
# stores cordinates of the human face in an array in the form [x, y, h, w] - a@ 2D array in case many human faces in one image
# x,y = upper-left corner (x,y)-cordinate
# h - height of rectangle
# w - width of rectangele
faceCordi = t_FaceData.detectMultiScale(grayImg);

#drawing rectangle around the detected face in red-colour.
# parameters denoting upper-left, lower-right cordi of rectangle and last one denoting bgr colour code.
for (x,y,h,w) in faceCordi:
    cv2.rectangle(img, (x,y), (x+h, y+w), (0,0,255), 3)

#pops up an dialog box with title as with string.
cv2.imshow('This is me', img)
#waits till any key is pressed for that pop-up to closed
cv2.waitKey()           #We can't display without usage of wait key. what it does basically after inside bracket value of ms presses the key automatically. and if not given braket value wait for us to manually press the key.

'''
#using face detection mechanism through the web-cam

#using default application(0) for capturing our video
camVideo = cv2.VideoCapture(0);         #can provide a video file here also, with path as second argument instead of 0
while True:
    #camVideo.read() methods reads(extract data) from our webcam and returns two value, boolean success and video_frame(x,y,h,w) cordinates of face in video 
    isSuccess, videoImg = camVideo.read();

    if(isSuccess == False):
        print("Webcam Loading failed !")
        break;

    #convert captured videoImg into black & white and use trained data to find out cordinates of the image
    grayVideoImg = cv2.cvtColor(videoImg, cv2.COLOR_BGR2GRAY);
    faceCordi = t_FaceData.detectMultiScale(grayVideoImg);
    for (x,y,h,w) in faceCordi:
        cv2.rectangle(videoImg, (x,y), (x+h, y+w), (0,0,255), 3)
    cv2.imshow('video Image', videoImg);
    keyPressed = cv2.waitKey(1)  #auto press the key after each ms to capture next loop frame
    #press q to quit. ASCII value used
    if(keyPressed == 113):
        break;

#release the webcam/video capturing device
camVideo.release();
