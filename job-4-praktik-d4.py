import cv2
from cvzone.HandTrackingModule import HandDetector

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi detector tangan
detector = HandDetector(staticMode=False, maxHands=1,
                        modelComplexity=1,
                        detectionCon=0.5, minTrackCon=0.5)

while True:
    # Baca frame
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi tangan
    # flipType=True agar hasilnya mirror (sesuai dengan gerakan tangan kita)
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        # Ambil tangan pertama yang terdeteksi
        hand = hands[0]  # dict berisi "lmList", "bbox", "center", "type"
        
        # Hitung jumlah jari yang terangkat
        fingers = detector.fingersUp(hand)  # list panjang 5 berisi 0 (turun) atau 1 (naik)
        count = sum(fingers)
        
        # Tampilkan jumlah jari
        cv2.putText(img, f"Fingers: {count}  {fingers}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("Hands + Fingers", img)
    
    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()