import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Fungsi untuk mengubah warna menjadi ungu (warna khas Thanos)
def apply_thanos_color(image):
    # Mengubah intensitas warna ungu
    purple_tint = image.copy()
    purple_tint[:, :, 0] = image[:, :, 0] * 1.2  # Meningkatkan komponen biru
    purple_tint[:, :, 1] = image[:, :, 1] * 0.8  # Menurunkan komponen hijau
    purple_tint[:, :, 2] = image[:, :, 2] * 1.1  # Meningkatkan komponen merah
    return purple_tint

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Gagal mengambil frame dari kamera")
        continue

    # Konversi BGR ke RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Deteksi wajah menggunakan MediaPipe
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Mendapatkan koordinat wajah
            h, w, _ = image.shape
            face_points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                face_points.append((x, y))
            
            # Membuat mask untuk area wajah
            mask = np.zeros((h, w), dtype=np.uint8)
            face_points_array = np.array(face_points)
            hull = cv2.convexHull(face_points_array)
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Aplikasikan efek Thanos pada area wajah
            thanos_effect = apply_thanos_color(image)
            
            # Gabungkan efek dengan gambar asli menggunakan mask
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            image = np.where(mask_3d > 0, thanos_effect, image)
            
            # Tambahkan garis tepi wajah
            cv2.polylines(image, [hull], True, (145, 0, 145), 2)

    # Tampilkan hasil
    cv2.imshow('Thanos Face Effect', image)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()