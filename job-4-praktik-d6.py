import cv2
import numpy as np
from collections import deque
from cvzone.PoseModule import PoseDetector

# --- Konfigurasi ---
MODE = "squat"  # tekan 'm' untuk toggle ke "pushup"
KNEE_DOWN, KNEE_UP = 80, 160     # ambang squat (deg)
DOWN_R, UP_R = 0.85, 1.00       # ambang push-up (rasio)
SAMPLE_OK = 4                   # minimal frame konsisten sebelum ganti state

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi detector pose
detector = PoseDetector(staticMode=False, modelComplexity=1,
                        enableSegmentation=False, detectionCon=0.5,
                        trackCon=0.5)

# Variabel state untuk counter
count = 0
state = "up"  # "up" atau "down"
debounce = deque(maxlen=6) # Menyimpan state beberapa frame terakhir


def ratio_pushup(lm):
    """
    Menghitung rasio push-up: jarak(bahu-pergelangan) / jarak(bahu-pinggul)
    """
    # gunakan kiri: 11=shoulderL, 15=wristL, 23=hipL
    # Ambil koordinat y dan z (indeks 1 dan 2) untuk perhitungan 2D
    sh = np.array(lm[11][1:3]) 
    wr = np.array(lm[15][1:3])
    hp = np.array(lm[23][1:3])
    # Tambahkan 1e-8 untuk menghindari pembagian dengan nol
    return np.linalg.norm(sh - wr) / (np.linalg.norm(sh - hp) + 1e-8)


while True:
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi pose
    img = detector.findPose(img, draw=True)
    # Dapatkan daftar landmark [(id,x,y,z,vis), ...]
    lmList, _ = detector.findPosition(img, draw=False)
    
    flag = None  # Status frame ini ("up", "down", atau None)

    if lmList:
        if MODE == "squat":
            # Hitung sudut lutut kiri (hip, knee, ankle)
            angL, img = detector.findAngle(lmList[23][0:2],
                                           lmList[25][0:2],
                                           lmList[27][0:2],
                                           img=img,
                                           color=(0, 0, 255),
                                           scale=10)
            
            # Hitung sudut lutut kanan
            angR, img = detector.findAngle(lmList[24][0:2],
                                           lmList[26][0:2],
                                           lmList[28][0:2],
                                           img=img,
                                           color=(0, 255, 0),
                                           scale=10)
            
            # Rata-rata kedua sudut
            ang = (angL + angR) / 2.0
            
            # Tentukan status "up" atau "down" berdasarkan ambang
            if ang < KNEE_DOWN:
                flag = "down"
            elif ang > KNEE_UP:
                flag = "up"
                
            cv2.putText(img, f"Knee: {ang:5.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        else: # "pushup"
            # Hitung rasio push-up
            r = ratio_pushup(lmList)
            
            # Tentukan status "up" atau "down" berdasarkan ambang
            if r < DOWN_R:
                flag = "down"
            elif r > UP_R:
                flag = "up"
                
            cv2.putText(img, f"Ratio: {r:4.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # --- Logika Counter (Debouncing) ---
        debounce.append(flag)
        
        # Jika state saat ini "up" dan frame "down" sudah konsisten -> ubah state
        if debounce.count("down") >= SAMPLE_OK and state == "up":
            state = "down"
            
        # Jika state saat ini "down" dan frame "up" sudah konsisten -> ubah state dan tambah hitungan
        if debounce.count("up") >= SAMPLE_OK and state == "down":
            state = "up"
            count += 1

    # Tampilkan informasi di layar
    cv2.putText(img, f"Mode: {MODE.upper()}  Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"State: {state}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Tampilkan frame
    cv2.imshow("Pose Counter", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Tekan 'q' untuk keluar
        break
    if key == ord('m'):  # Tekan 'm' untuk ganti mode
        MODE = "pushup" if MODE == "squat" else "squat"

# Bersihkan
cap.release()
cv2.destroyAllWindows()