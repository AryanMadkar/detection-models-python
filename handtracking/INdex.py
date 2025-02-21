import cv2
from main import HandDetector
import time
import mediapipe as mp
cap = cv2.VideoCapture(0)


cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

detector = HandDetector()
ptime = time.time()

while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Error: Unable to read from camera.")
            break  

        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        # if lm_list:
        #     print(lm_list[4])  # Print the position of the 5th landmark

        # FPS Calculation
        ctime = time.time()
        fps = int(1 / (ctime - ptime))
        ptime = ctime

        cv2.putText(img, f'FPS: {fps}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
