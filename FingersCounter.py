# For now,This works for Right Hand only
import cv2
import HandTracking_Module as htm
import os
import time

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 640)

path = "Images"
overlayList = []
for impath in os.listdir(path):
    img = cv2.imread(f"{path}/{impath}")
    img = cv2.resize(img, (200, 200))
    overlayList.append(img)

detector = htm.HandDetection(detect_conf=0.80)

tipIds = [4, 8, 12, 16, 20]
ptime = 0
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    fingers = detector.fingersUp()

    totalFingers = fingers.count(1)
    img[0:200, 0:200] = overlayList[totalFingers-1]

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, f"FPS {int(fps)} ", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.putText(img, f" {totalFingers} ", (25, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

    cv2.imshow("Image", img)

    if cv2.waitKey(5) & 0xFF == ord('q'): break


