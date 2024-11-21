import cv2
from detect import detect_face
image_face = cv2.imread("E:/3.hocki1nam4/xu li anh/face_recognition/data/B21DCCN678.jpg")
result = detect_face(image_face)
print(result)
