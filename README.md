This Repository contains the trained models for detecting UAVs in airport conditions. The code for training the model and the trained model are placed in the “Models” Folder.
A sample of UAV audio is saved as “Drone_tst.wav”. The UAV audio must first be pre-processed for different models like 1D-F-CNN for PSD, MFCC, Mels ZCR, etc, using the
“Data_pre_processing_codes.py” and the files extracted must be used to detect UAVs. The models can be loaded and tested using the script “evaluate differnt SNRs.py”.
The process for running the pre-trained model is as follows:
Extract features (pre-processing ) from the given audio sample >> Load the models and use the samples extracted in pre-processing for prediction with different models saved in the folder. 

In the case of WST-CNN saved as.zip, the pre-processing only segments the audio samples in 0.8 seconds of raw audio.
