import pyaudio
import numpy as np
from scipy.signal import butter, lfilter

# Konfigurasi audio
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
NOISE_THRESHOLD = 1000  # Ambang batas noise untuk Noise Gate
TARGET_PEAK = 20000     # Target untuk normalisasi amplitudo

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
    max_amplitude = np.max(np.abs(data))
    if max_amplitude < threshold:
        return np.zeros_like(data)
    return data

# Efek suara Thanos
def thanos_effect(audio_chunk, pitch_factor=0.8, resonance_factor=0.9, cutoff_freq=800):
    # Terapkan Noise Gate
    gated_audio = noise_gate(audio_chunk)

    # Pitch Shifting
    pitched_audio = pitch_shift(gated_audio, pitch_factor)

    # Filter Low-Pass
    filtered_audio = low_pass_filter(pitched_audio, cutoff=cutoff_freq, fs=RATE)

    # Resonansi dan normalisasi
    resonant_audio = filtered_audio * resonance_factor
    return normalize_audio(resonant_audio)

# Fungsi utama untuk filter suara real-time
def realtime_voice_filter():
    audio_interface = pyaudio.PyAudio()

    # Stream Input dan Output
    stream_input = audio_interface.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    stream_output = audio_interface.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK
    )

    print("Filter suara Thanos aktif. Silahkan bicara ke mikrofon...")
    try:
        while True:
            input_data = stream_input.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(input_data, dtype=np.int16)

            # Terapkan efek Thanos
            modified_audio = thanos_effect(audio_chunk)

            # Tulis ke Output
            stream_output.write(modified_audio.tobytes())

    except KeyboardInterrupt:
        pass  # Hentikan program dengan menekan Ctrl+C 
    finally:
        stream_input.stop_stream()
        stream_input.close()
        stream_output.stop_stream()
        stream_output.close()
        audio_interface.terminate()

if __name__ == "__main__":
    realtime_voice_filter()
