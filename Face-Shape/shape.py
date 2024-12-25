import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Fungsi untuk mengubah warna kulit wajah menjadi ungu dengan transparansi optimal
def apply_purple_skin(image, landmarks, opacity=0.5):
    h, w, _ = image.shape

    # Buat mask untuk area wajah
    face_mask = np.zeros((h, w), dtype=np.uint8)
    face_points = [
        (int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks
    ]
    convex_hull = cv2.convexHull(np.array(face_points, dtype=np.int32))
    cv2.fillConvexPoly(face_mask, convex_hull, 255)

    # Buat overlay ungu
    purple_overlay = np.zeros_like(image, dtype=np.uint8)
    purple_overlay[:, :] = (128, 0, 128)

    # Gabungkan overlay ungu dengan area wajah menggunakan transparansi
    face_area = cv2.bitwise_and(purple_overlay, purple_overlay, mask=face_mask)
    result = cv2.addWeighted(image, 1 - opacity, face_area, opacity, 0)
    return result

# Fungsi untuk menambahkan dagu khas Thanos
def add_thanos_chin(image, landmarks):
    h, w, _ = image.shape

    # Landmark dagu di MediaPipe
    chin_tip = landmarks[152]  # Titik dagu utama
    left_jaw = landmarks[234]  # Rahang kiri
    right_jaw = landmarks[454]  # Rahang kanan

    # Konversi ke koordinat piksel
    chin_tip_coords = (int(chin_tip.x * w), int(chin_tip.y * h))
    left_jaw_coords = (int(left_jaw.x * w), int(left_jaw.y * h))
    right_jaw_coords = (int(right_jaw.x * w), int(right_jaw.y * h))

    # Buat mask transparan untuk dagu
    mask = np.zeros_like(image, dtype=np.uint8)
    triangle_points = np.array([chin_tip_coords, left_jaw_coords, right_jaw_coords], np.int32)
    cv2.fillConvexPoly(mask, triangle_points, (128, 0, 128))

    # Gabungkan mask dengan frame menggunakan transparansi
    modified_image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    return modified_image

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera.")
        break

    # Konversi gambar ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Peroleh koordinat landmark wajah
            landmarks = face_landmarks.landmark

            # Ubah warna kulit menjadi ungu
            frame = apply_purple_skin(frame, landmarks, opacity=0.5)

            # Tambahkan dagu Thanos
            frame = add_thanos_chin(frame, landmarks)

    # Tampilkan hasil
    cv2.imshow("Thanos Face Transformation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
