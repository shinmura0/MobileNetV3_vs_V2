import cv2
import time
import os
import numpy as np
from keras.models import model_from_json
from keras import backend as K
from model.mobilenet_v3_small import MobileNetV3_Small

input_size = 32
model_path = "weights/" 
shape = (input_size, input_size, 3)
classes = 10
alpha = 0.2

message = "Input size : 32 x 32"
label = ['airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck']

# model load
print("V3_32 Model loading...")
model = MobileNetV3_Small(shape, classes, alpha).build()
model.load_weights(path + 'v3_32.h5')
    
    
def main():
    camera_width =  352
    camera_height = 288
    fps = ""
    flag_score = False
    elapsedTime = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 40)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    time.sleep(1)

    while cap.isOpened():
        t1 = time.time()

        ret, image = cap.read()
        if not ret:
            break

        image = image[:,32:320]

        # inference
        img = cv2.resize(image, (input_size, input_size))
        img = np.array(img).reshape((1, input_size, input_size,3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(np.max(img))
        test = model.predict(img/255)

        class_ = np.argmax(test)
        score = test[class_]
        name = label[class_]

        # output score
        cv2.putText(image, "{:.2f} Score".format(score),(camera_width - 150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
              
        # message
        cv2.putText(image, message, (camera_width - 200, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, fps, (camera_width - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 255, 0 ,0), 1, cv2.LINE_AA)

        cv2.imshow("Result", image)
            
        # FPS
        elapsedTime = time.time() - t1
        fps = "{:.0f} FPS".format(1/elapsedTime)

        # quit
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()