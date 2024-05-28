"""
Drone Data augmentation with noise addition creating a single audio file

@author: karan_nathwani
"""

import numpy as np
import librosa
import soundfile as sf

def add_noise(segment, snr):
    segment_power = np.mean(segment ** 2)
    noise_power = segment_power / (10 ** (snr / 10))
    noise = np.random.normal(scale=np.sqrt(noise_power), size=len(segment))
    return segment + noise

# Load the audio file
audio_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/Non_drone_merged.wav"
y, sr = librosa.load(audio_file, sr=None)

# Define segment length in samples (100ms)
segment_length_samples = int(0.8 * sr)

# Calculate the total number of segments
num_segments = len(y) // segment_length_samples

# Initialize an empty list to store concatenated audio
concatenated_audio = []

# Define SNR levels
snr_levels = [-30,-25,-20,-15,-10,-5,0,5,10]  # dB

# Loop through each SNR level
for snr in snr_levels:
    # Initialize an empty array to store noisy segments
    noisy_segments = np.zeros((num_segments, segment_length_samples))

    # Loop through each segment
    for i in range(num_segments):
        # Extract the segment
        segment = y[i * segment_length_samples: (i + 1) * segment_length_samples]

        # Add noise to the segment
        noisy_segments[i] = add_noise(segment, snr)

    # Concatenate the noisy segments across different SNR levels
    concatenated_audio.append(noisy_segments)

# Concatenate audio across different SNR levels
final_concatenated_audio = np.concatenate(concatenated_audio, axis=0)

# Max normalize the final concatenated audio
final_concatenated_audio /= np.max(np.abs(final_concatenated_audio))

# Write the concatenated audio as a single audio file with a specific sampling rate
output_file = "/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Binary Data/Non_drone_merged_all_SNRs.wav"
sf.write(output_file, final_concatenated_audio.flatten(), 44100)
