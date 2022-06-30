import keras_preprocessing.image
import tensorflow as tf
from keras.models import model_from_json
import cv2
from keras_preprocessing import image
import numpy as np

model = model_from_json(open("model_new.json", "r").read())
model.load_weights("fire_detection_weights.h5")

# path = "dataset/test/fire_images/fire.1.png"
#
#
# img = image.load_img(path, target_size=(224,224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0) / 255
# classes = model.predict(x)
# print(np.argmax(classes[0])==0, max(classes[0]))

cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
ret,frame = cap.read() # return a single frame in variable `frame`

while(True):

    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
        cv2.imwrite('images/deneme.png',frame)
        cv2.destroyAllWindows()
        break

cap.release()


# cap = cv2.VideoCapture(0)
#
# while True:
#
#     ret, frame = cap.read()
#
#     if ret:
#         frame = cv2.resize(frame, (224,224))
#         x = np.array(frame)
#         x = x / 255
#         classes = model.predict(x)
#         print(np.argmax(classes[0]) == 0, max(classes[0]))
#
#
#     if cv2.waitKey(1) & 0xFF == ord("q"): break
#
# cap.release()
# cv2.destroyAllWindows()
