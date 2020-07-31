# Download CommonVoice2 dataset
curl https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-2/en.tar.gz -o cv2_en.tar.gz

# Extract CommonVoice2 dataset
tar xvzf cv2_en.tar.gz --one-top-level=CommonVoice2_dataset

# Run resampling 48khz to 16khz
python resample_common_voice2_48khz_to_16khz.py