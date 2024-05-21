import os
import numpy as np
import librosa
from scipy.signal import welch

#%% Raw audio segmentation with 1 second

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/Non_drone_merged.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
segment_length = int(segment_duration * sr)  # Calculate the length of each segment in samples
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30, -25, -20, -15, -10, -5, 0, 5, 10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/Raw audio/No Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        noisy_segment = noisy_segment / np.max(np.abs(noisy_segment))
        filename = os.path.join(output_folder, f"seg_NSNR_{snr}_index_{i}.npy")
        np.save(filename, noisy_segment.astype(np.float16))

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/combined drones.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/Raw audio/Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        noisy_segment = noisy_segment / np.max(np.abs(noisy_segment))
        filename = os.path.join(output_folder, f"seg_SNR_{snr}_index_{i}.npy")
        np.save(filename, noisy_segment.astype(np.float16))
        
#%% Preprocessing with 1D-MFCC (20) extraction

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/combined drones.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values =[-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/MFCC 20/Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        mfcc_features = librosa.feature.mfcc(y=noisy_segment, sr=sr, n_fft=1024, hop_length=512, n_mfcc=20)
        pooled_mfcc = np.mean(mfcc_features, axis=1)
        pooled_mfcc = pooled_mfcc/np.max(np.abs(pooled_mfcc))
        pooled_mfcc = pooled_mfcc.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_DMFCC_{snr}_index_{i}.npy")
        np.save(filename, pooled_mfcc)
        
audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/Non_drone_merged.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/MFCC 20/No Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        mfcc_features = librosa.feature.mfcc(y=noisy_segment, sr=sr, n_fft=1024, hop_length=512, n_mfcc=20)
        pooled_mfcc = np.mean(mfcc_features, axis=1)
        pooled_mfcc = pooled_mfcc/np.max(np.abs(pooled_mfcc))
        pooled_mfcc = pooled_mfcc.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_NMFCC_{snr}_index_{i}.npy")
        np.save(filename, pooled_mfcc)


#%% Preprocessing with 1D-Mel (128) extraction

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/combined drones.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/Mel 128/Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        mel_spectrogram = librosa.feature.melspectrogram(y=noisy_segment, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        pooled_mel_spectrogram = np.mean(mel_spectrogram_db, axis=1)
        pooled_mel_spectrogram /= np.max(np.abs(pooled_mel_spectrogram))
        
        pooled_mel_spectrogram = pooled_mel_spectrogram.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_DMEL_{snr}_index_{i}.npy")
        np.save(filename, pooled_mel_spectrogram)
        

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/Non_drone_merged.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/Mel 128/No Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        mel_spectrogram = librosa.feature.melspectrogram(y=noisy_segment, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        pooled_mel_spectrogram = np.mean(mel_spectrogram_db, axis=1)
        pooled_mel_spectrogram /= np.max(np.abs(pooled_mel_spectrogram))
        
        pooled_mel_spectrogram = pooled_mel_spectrogram.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_NMEL_{snr}_index_{i}.npy")
        np.save(filename, pooled_mel_spectrogram)

#%% Preprocessing with PSD (513) extraction

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/combined drones.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/PSD/Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        _, psd = welch(noisy_segment, fs=sr, nperseg=1024)
        psd = psd/ np.max(np.abs(psd))
        pooled_psd = psd.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_DPSD_{snr}_index_{i}.npy")
        np.save(filename, pooled_psd)
        

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/Non_drone_merged.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/PSD/No Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        _, psd = welch(noisy_segment, fs=sr, nperseg=1024)
        psd = psd/ np.max(np.abs(psd))
        pooled_psd = psd.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_NPSD_{snr}_index_{i}.npy")
        np.save(filename, pooled_psd)

#%% Preprocessing with Zero Crossing Rate (ZCR) extraction

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/combined drones.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/ZCR/Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        zcr = librosa.feature.zero_crossing_rate(y=noisy_segment, frame_length=1024, hop_length=512)
        zcr = zcr[0]/np.max(np.abs(zcr[0]))
        filename = os.path.join(output_folder, f"seg_DZCR_{snr}_index_{i}.npy")
        np.save(filename, zcr)
        

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/Non_drone_merged.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/ZCR/No Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        zcr = librosa.feature.zero_crossing_rate(y=noisy_segment, frame_length=1024, hop_length=512)
        zcr = zcr[0]/np.max(np.abs(zcr[0]))
        filename = os.path.join(output_folder, f"seg_NZCR_{snr}_index_{i}.npy")
        np.save(filename, zcr)

#%% Preprocessing with Spectral Centroid (SC) extraction

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/combined drones.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/SC/Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        centroid = librosa.feature.spectral_centroid(y=noisy_segment, sr=sr, n_fft=1024, hop_length=512)
        centroid = centroid[0]/ np.max (np.abs(centroid[0]))
        filename = os.path.join(output_folder, f"seg_DSC_{snr}_index_{i}.npy")
        np.save(filename, centroid)
        

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/Non_drone_merged.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/SC/No Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        centroid = librosa.feature.spectral_centroid(y=noisy_segment, sr=sr, n_fft=1024, hop_length=512)
        centroid = centroid[0]/ np.max (np.abs(centroid)[0])
        filename = os.path.join(output_folder, f"seg_NSC_{snr}_index_{i}.npy")
        np.save(filename, centroid)

#%%  Preprocessing with MFCC (2D) extraction

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/combined drones.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/2D Features/MFCC 20/Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        mfcc_features = librosa.feature.mfcc(y=noisy_segment, sr=sr, n_fft=1024, hop_length=512, n_mfcc=20)
        mfcc_norm = np.linalg.norm(mfcc_features)
        normalized_mfcc_features = mfcc_features / mfcc_norm 
        mfcc = normalized_mfcc_features.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_DMFCC_2D_{snr}_index_{i}.npy")
        np.save(filename, mfcc)


audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/Non_drone_merged.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/2D Features/MFCC 20/No Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        mfcc_features = librosa.feature.mfcc(y=noisy_segment, sr=sr, n_fft=1024, hop_length=512, n_mfcc=20)
        mfcc_norm = np.linalg.norm(mfcc_features)
        normalized_mfcc_features = mfcc_features / mfcc_norm 
        mfcc = normalized_mfcc_features.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_NMFCC_2D_{snr}_index_{i}.npy")
        np.save(filename, mfcc)
        
#%%Preprocessing with mel (2D) extraction

audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/combined drones.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/2D Features/Mel 128/Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        mel_spectrogram = librosa.feature.melspectrogram(y=noisy_segment, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_norm = np.linalg.norm(mel_spectrogram_db)
        normalized_mel_features = mel_spectrogram_db / mel_norm 
        normalized_mel_features = normalized_mel_features.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_DMEL_{snr}_index_{i}.npy")
        np.save(filename, normalized_mel_features)


audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/Non_drone_merged.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/2D Features/Mel 128/No Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        mel_spectrogram = librosa.feature.melspectrogram(y=noisy_segment, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_norm = np.linalg.norm(mel_spectrogram_db)
        normalized_mel_features = mel_spectrogram_db / mel_norm 
        normalized_mel_features = normalized_mel_features.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_NMEL_{snr}_index_{i}.npy")
        np.save(filename, normalized_mel_features)
        
#%% MFCC extraction for 1d-M-CNN with 128


audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/combined drones.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
segment_length = int(segment_duration * sr)
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values =[-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/1D_M_CNN/Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        mfcc_features = librosa.feature.mfcc(y=noisy_segment, sr=sr, n_fft=int(segment_duration*sr), hop_length=int(segment_duration*sr), n_mfcc=128)
        pooled_mfcc =  mfcc_features[:, :1]
        pooled_mfcc = pooled_mfcc/np.max(np.abs(pooled_mfcc))
        pooled_mfcc = pooled_mfcc.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_DMFCC_{snr}_index_{i}.npy")
        np.save(filename, pooled_mfcc)
        
audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/Non_drone_merged.wav"
audio_data, sr = librosa.load(audio_file, sr=None)
total_duration = librosa.get_duration(y=audio_data, sr=sr)
segment_duration = 0.8
num_segments = int(np.ceil(total_duration / segment_duration))
snr_values = [-30,-25,-20,-15,-10,-5,0,5,10]  # Example SNR values in dB
output_folder = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/1D_M_CNN/No Drone"
os.makedirs(output_folder, exist_ok=True)
for snr in snr_values:
    for i in range(num_segments):
        start_sample = int(i * segment_length)
        end_sample = int(min((i + 1) * segment_length, len(audio_data)))
        segment = audio_data[start_sample:end_sample]
        if len(segment) < segment_length:  # If segment is shorter, pad with zeros
            segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
        elif len(segment) > segment_length:  # If segment is longer, trim it
            segment = segment[:segment_length]
        power = np.mean(segment ** 2)
        noise_power = power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), segment.shape)
        noisy_segment = segment + noise
        mfcc_features = librosa.feature.mfcc(y=noisy_segment, sr=sr, n_fft=int(segment_duration*sr), hop_length=int(segment_duration*sr), n_mfcc=128)
        pooled_mfcc =  mfcc_features[:, :1]
        pooled_mfcc = pooled_mfcc/np.max(np.abs(pooled_mfcc))
        pooled_mfcc = pooled_mfcc.astype(np.float16)
        filename = os.path.join(output_folder, f"seg_NMFCC_{snr}_index_{i}.npy")
        np.save(filename, pooled_mfcc)


