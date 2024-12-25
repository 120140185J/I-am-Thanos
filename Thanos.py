import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import threading
from scipy.signal import butter, lfilter

# Inisialisasi MediaPipe Face Mesh untuk deteksi wajah
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Konfigurasi audio
CHUNK = 2048  # Ukuran chunk lebih besar untuk respons lebih cepat
FORMAT = pyaudio.paFloat32  # Menggunakan format float32 untuk kualitas lebih baik
CHANNELS = 1
RATE = 44100
NOISE_THRESHOLD = 0.005  # Menurunkan threshold untuk meningkatkan sensitivitas

# Memuat gambar Thanos
thanos_image = cv2.imread('assets/thanoss.png', cv2.IMREAD_UNCHANGED)
if thanos_image is None:
    raise FileNotFoundError("Thanos image not found. Please check the path 'assets/thanoss.png'")
thanos_image = cv2.resize(thanos_image, (640, 480))  # Ukuran gambar Thanos

def modify_face_shape(image, landmarks):
    """
    Fungsi untuk memodifikasi bentuk wajah:
    - Memperlebar bagian rahang lebih banyak
    - Membuat wajah lebih kekar
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Konversi landmark ke array numpy untuk memudahkan manipulasi
    points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])
    
    # Perlebar rahang lebih signifikan (30%)
    jaw_points = points[152:175]  # Titik-titik rahang
    center = np.mean(jaw_points, axis=0)  # Titik tengah rahang
    
    # Memperlebar rahang dengan faktor 1.3 (30% lebih lebar)
    jaw_points = (jaw_points - center) * 1.3 + center
    
    # Gambar mask wajah dengan titik-titik rahang yang sudah dimodifikasi
    cv2.fillConvexPoly(mask, points.astype(int), 255)
    
    return mask

def apply_thanos_face(frame, landmarks):
    """
    Fungsi untuk mengganti wajah pengguna dengan wajah Thanos
    berdasarkan landmark wajah yang terdeteksi.
    """
    h, w = frame.shape[:2]
    
    # Menyesuaikan gambar Thanos dengan posisi wajah pengguna
    points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])
    
    # Mencari bounding box wajah
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    
    # Menyesuaikan ukuran dan posisi gambar Thanos
    thanos_resized = cv2.resize(thanos_image, (x_max - x_min, y_max - y_min))
    
    # Posisi di mana gambar Thanos akan ditempatkan
    roi = frame[y_min:y_max, x_min:x_max]
    
    # Menambahkan gambar Thanos ke wajah pengguna
    alpha = 0.7  # Transparansi gambar Thanos
    frame[y_min:y_max, x_min:x_max] = cv2.addWeighted(roi, 1 - alpha, thanos_resized, alpha, 0)
    
    return frame

def process_audio(data, pitch_factor=0.8):
    """
    Fungsi untuk memproses audio:
    - Menurunkan pitch suara
    - Menambah bass
    - Menghilangkan noise
    """
    # Konversi ke float32 untuk pemrosesan
    audio_data = np.frombuffer(data, dtype=np.float32)
    
    # Noise gate
    if np.max(np.abs(audio_data)) < NOISE_THRESHOLD:
        return np.zeros_like(audio_data)
    
    # Pitch shifting (lebih halus dan dinamis)
    indices = np.arange(0, len(audio_data), pitch_factor)
    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
    
    # Low pass filter untuk menambah bass
    nyquist = RATE / 2
    cutoff = 800 / nyquist
    b, a = butter(4, cutoff, btype='low')
    audio_data = lfilter(b, a, audio_data)
    
    return audio_data.astype(np.float32)

def audio_stream():
    """
    Fungsi untuk menjalankan stream audio secara real-time
    """
    p = pyaudio.PyAudio()
    stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
    while True:
        try:
            data = stream_in.read(CHUNK, exception_on_overflow=False)
            processed = process_audio(data)
            stream_out.write(processed.tobytes())
        except Exception as e:
            print(f"Audio stream error: {e}")
            break
        except:
            break

    stream_in.stop_stream()
    stream_out.stop_stream()
    stream_in.close()
    stream_out.close()
    p.terminate()

def main():
    # Mulai audio processing di thread terpisah
    audio_thread = threading.Thread(target=audio_stream)
    audio_thread.daemon = True
    audio_thread.start()
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        # Proses frame untuk deteksi wajah
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Terapkan gambar Thanos ke wajah pengguna
                frame = apply_thanos_face(frame, face_landmarks)
        
        cv2.imshow('Thanos Effect', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
