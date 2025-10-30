import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

def dist(a, b):
    """Menghitung jarak Euclidean antara dua titik (a, b)"""
    return np.linalg.norm(np.array(a) - np.array(b))

def classify_gesture(hand):
    """Mengklasifikasikan gestur tangan berdasarkan posisi landmark"""
    # hand["lmList"] berisi 21 titik (x,y,z) dalam piksel saat flipType=True
    lm = hand["lmList"]
    
    # Ambil koordinat x, y dari ujung jari dan pergelangan tangan
    wrist = np.array(lm[0][:2])
    thumb_tip = np.array(lm[4][:2])
    index_tip = np.array(lm[8][:2])
    middle_tip = np.array(lm[12][:2])
    ring_tip = np.array(lm[16][:2])
    pinky_tip = np.array(lm[20][:2])
    
    # Heuristik jarak relatif (rata-rata jarak ujung jari ke pergelangan)
    r_mean = np.mean([
        dist(index_tip, wrist),
        dist(middle_tip, wrist),
        dist(ring_tip, wrist),
        dist(pinky_tip, wrist),
        dist(thumb_tip, wrist)
    ])
    
    # --- Aturan Klasifikasi Gestur Sederhana ---
    # Aturan ini sangat bergantung pada resolusi kamera dan jarak tangan.
    # Nilai ambang (35, 40, 120, 200, 180, 160) mungkin perlu disesuaikan.

    # "OK": Jarak antara ujung ibu jari dan ujung telunjuk sangat dekat
    if dist(thumb_tip, index_tip) < 35:
        return "OK"
    
    # "THUMBS_UP": Ibu jari di atas (y lebih kecil) dan jauh dari pergelangan
    if (thumb_tip[1] < wrist[1] - 40) and (dist(thumb_tip, wrist) > 0.8 * dist(index_tip, wrist)):
        return "THUMBS_UP"
    
    # "ROCK": Jari-jari menggenggam (jarak rata-rata ke pergelangan pendek)
    if r_mean < 120:
        return "ROCK"
    
    # "PAPER": Jari-jari terbuka lebar (jarak rata-rata ke pergelangan panjang)
    if r_mean > 200:
        return "PAPER"
    
    # "SCISSORS": Jari telunjuk dan tengah lurus, jari manis dan kelingking ditekuk
    if dist(index_tip, wrist) > 180 and dist(middle_tip, wrist) > 180 and \
       dist(ring_tip, wrist) < 160 and dist(pinky_tip, wrist) < 160:
        return "SCISSORS"
    
    return "UNKNOWN"

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
    hands, img = detector.findHands(img, draw=True, flipType=True)
    
    if hands:
        # Klasifikasikan gestur dari tangan pertama yang terdeteksi
        label = classify_gesture(hands[0])
        
        # Tampilkan label gestur
        cv2.putText(img, f"Gesture: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Tampilkan hasil
    cv2.imshow("Hand Gestures (cvzone)", img)
    
    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()