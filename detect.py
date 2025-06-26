import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

# ---------------- Configuration ----------------
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction to define active region
smoothening = 7  # Smooth movement factor

# ---------------- Initialization ----------------
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)  # Use 0 or 1 depending on your camera
if not cap.isOpened():
    print("❌ Error: Could not access the webcam.")
    exit()

cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

# ---------------- Main Loop ----------------
while True:
    success, img = cap.read()
    if not success or img is None:
        print("⚠️ Skipping frame: Camera not ready.")
        continue

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        # Tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        # Draw active area
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # ---------------- Mouse Movement Mode ----------------
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smooth the movement
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move the mouse
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # ---------------- Click Mode ----------------
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # ---------------- Frame Rate Display ----------------
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-6)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # ---------------- Display ----------------
    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) == ord('q'):
        break

# ---------------- Cleanup ----------------
cap.release()
cv2.destroyAllWindows()
