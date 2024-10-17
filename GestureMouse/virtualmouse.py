import cv2
import numpy as np
import time
import handTrackingModule as htm
import autopy

###################
wCam, hCam = 640, 480
frameR = 64 # Frame Reduction

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

smoothening = 6
###################

wScr, hScr = autopy.screen.size()
print(wScr, hScr)
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)

    # 1. Find the hand landmarks.
    # 2. Get the tip of the middle and the index finger.
    # 3. Check which fingers are up.
    # 4. Only Index finger - moving mode.
    # 5. Convert coordinates to get the correct positioning.
    # 6. Smoothen the values
    # 7. Move mouse
    # 8. When both index and middle fingers are up: Clicking mode
    # 9. Find distance b/w fingers.
    # 10. Click Mouse if distance is short.
    # 11. Frame rate.
    # 12. Display

while True:
    
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # print(x1, y1, x2, y2)

        fingers = detector.fingersUp()
        # print(fingers)

        if fingers[1]==1 and fingers[2]==0:
            cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255, 0, 255), 2)
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr)) 
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            clocX = plocX + (x3-plocX) / smoothening
            clocY = plocY + (y3-plocY) / smoothening

            autopy.mouse.move(wScr-clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        
        if fingers[1]==1 and fingers[2]==1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            if length < 28:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime 
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()