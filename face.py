import cv2
import sys


imagePath = sys.argv[1]


# Create the haar cascade


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.25,
    minNeighbors=1,
    minSize=(8,8),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
