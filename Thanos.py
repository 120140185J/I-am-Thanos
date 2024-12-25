import cv2
import numpy as np
import mediapipe as mp
import pyaudio
from scipy.signal import butter, lfilter
import threading

# Konfigurasi audio
CHUNK = 2048  # Ukuran buffer: 2,048 sampel
FORMAT = pyaudio.paInt16  # Format audio: 16-bit PCM
CHANNELS = 1  # Mono
RATE = 44100  # Sampling rate: 44,100 Hz
NOISE_THRESHOLD = 300  # Ambang batas noise lebih rendah untuk responsivitas lebih baik
TARGET_PEAK = 25000    # Disesuaikan untuk audio yang lebih jelas

# Fungsi low-pass filter
def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, data)

# Fungsi untuk pitch shifting dengan interpolasi
def pitch_shift(data, pitch_factor):
    indices = np.arange(0, len(data), pitch_factor)
    indices = indices[indices < len(data)]
    return np.interp(indices, np.arange(len(data)), data).astype(data.dtype)

# Normalisasi amplitudo
def normalize_audio(data, target_peak=TARGET_PEAK):
    max_amplitude = np.max(np.abs(data))
    if max_amplitude == 0:
        return data
    scaling_factor = min(1.0, target_peak / max_amplitude)
    return np.clip(data * scaling_factor, -32768, 32767).astype(np.int16)

# Noise Gate untuk menghilangkan noise
def noise_gate(data, threshold=NOISE_THRESHOLD):
    if np.max(np.abs(data)) < threshold:
        return np.zeros_like(data)
    return data

# Efek suara Thanos
def thanos_effect(audio_chunk, pitch_factor=0.85, resonance_factor=1.2, cutoff_freq=1000):
    gated_audio = noise_gate(audio_chunk)
    pitched_audio = pitch_shift(gated_audio, pitch_factor)
    filtered_audio = low_pass_filter(pitched_audio, cutoff=cutoff_freq, fs=RATE)
    resonant_audio = filtered_audio * resonance_factor
    return normalize_audio(resonant_audio)

# Fungsi utama untuk filter suara real-time
def realtime_voice_filter():
    audio_interface = pyaudio.PyAudio()
    stream_input = audio_interface.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    stream_output = audio_interface.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

    print("Filter suara Thanos aktif. Silahkan bicara ke mikrofon...")
    try:
        while True:
            input_data = stream_input.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(input_data, dtype=np.int16)
            modified_audio = thanos_effect(audio_chunk)
            stream_output.write(modified_audio.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        stream_input.stop_stream()
        stream_input.close()
        stream_output.stop_stream()
        stream_output.close()
        audio_interface.terminate()

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7, refine_landmarks=True)

# Fungsi untuk mengubah warna wajah menjadi ungu (warna khas Thanos)
def apply_thanos_color(image, mask):
    purple_tint = image.copy()
    purple_tint[:, :, 0] = np.clip(purple_tint[:, :, 0] * 1.5, 0, 255)  # Biru
    purple_tint[:, :, 1] = np.clip(purple_tint[:, :, 1] * 0.6, 0, 255)  # Hijau
    purple_tint[:, :, 2] = np.clip(purple_tint[:, :, 2] * 1.8, 0, 255)  # Merah
    return cv2.addWeighted(image, 0.3, purple_tint, 0.7, 0, dtype=cv2.CV_8U)

# Fungsi untuk mendeteksi wajah dan menerapkan filter Thanos
def process_frame(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            face_points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]

            # Buat mask untuk area wajah
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.array(face_points, dtype=np.int32), 255)
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Terapkan warna ungu pada wajah
            image = np.where(mask_3d > 0, apply_thanos_color(image, mask_3d), image)

            # Tambahkan garis tepi wajah
            hull = cv2.convexHull(np.array(face_points, dtype=np.int32))
            cv2.polylines(image, [hull], True, (145, 0, 145), 2)

    return image

# Fungsi utama untuk filter wajah dan suara secara real-time
def main():
    threading.Thread(target=realtime_voice_filter, daemon=True).start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    print("Filter wajah dan suara Thanos aktif.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari kamera.")
            break

        # Proses frame untuk filter wajah
        frame = process_frame(frame)

        # Tampilkan hasil
        cv2.imshow("Thanos Face Effect", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()