import cv2
import mediapipe as mp
import time

class PoseDetector:
    def __init__(self, mode=False, smooth_landmarks=True, detection_confidence=0.5, tracking_confidence=0.5):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=mode,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.landmark_color = self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=2)
        self.connection_color = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1)
        self.results = None  # Store results for processing

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, 
                                        mp.solutions.pose.POSE_CONNECTIONS, 
                                        self.landmark_color, self.connection_color)
        
        return img

    def find_position(self, img, draw=True):
        lm_list = []
        if not self.results or not self.results.pose_landmarks:
            return lm_list

        h, w, _ = img.shape
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((id, cx, cy))
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

        return lm_list


def main():
    cap = cv2.VideoCapture(r"F:\dektop241205\camstar\bodytracking\fight2.mp4")  # Change path if needed
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
