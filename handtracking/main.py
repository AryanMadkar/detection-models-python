import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detection_confidence=0.5, track_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=mode, 
            max_num_hands=maxHands, 
            min_detection_confidence=detection_confidence, 
            min_tracking_confidence=track_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.landmark_color = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.results = None  # Store results for processing

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, mp.solutions.hands.HAND_CONNECTIONS, self.landmark_color, self.landmark_color)
        
        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if not self.results or not self.results.multi_hand_landmarks:
            return lm_list

        try:
            hand = self.results.multi_hand_landmarks[hand_no]
            h, w, _ = img.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), -1)
        except IndexError:
            pass  # Avoid unnecessary print statements for performance

        return lm_list


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
