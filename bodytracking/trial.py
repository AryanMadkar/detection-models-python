import cv2
import time
import mediapipe as mp
from model import PoseDetector

cap = cv2.VideoCapture(r"F:\dektop241205\camstar\bodytracking\fight.mp4")  # Change path if needed
detector = PoseDetector()
ptime = time.time()

while cap.isOpened():
        success, img = cap.read()
        if not success:
            break  

        img = detector.find_pose(img)
        lm_list = detector.find_position(img,draw=False)

        # Uncomment this to print landmark coordinates
        # if lm_list:
        #     print(lm_list[0])  # Print first landmark position

        # FPS Calculation
        ctime = time.time()
        fps = int(1 / (ctime - ptime)) if (ctime - ptime) > 0 else 0
        ptime = ctime

        cv2.putText(img, f'FPS: {fps}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Pose Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()