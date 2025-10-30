import cv2 , time

cap=cv2.VideoCapture(0)
frames=0
t0=time.time()

while True:
    ok,frame=cap.read()
    frames= frames + 1
    if time.time() - t0 >= 1.0:
        cv2.setWindowTitle("Preview", f"Preview - FPS ~ {frames}")
        frames, t0 = 0, time.time()

    cv2.imshow("Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()