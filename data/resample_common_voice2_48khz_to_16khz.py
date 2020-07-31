import os
import numpy as np
import torchaudio

src_path = "CommonVoice2_dataset/clips/"
tgt_path = "CommonVoice2_dataset/clips_16khz/"
resample = torchaudio.transforms.Resample(48000, 16000)

def load_audio(old_path, new_path):
    sound, _ = torchaudio.load(old_path, normalization=False)
    sound = resample(sound)
    torchaudio.save(new_path, sound, 16000)

error_files = []
dirs = os.listdir(src_path)

for i in range(742728, len(dirs)):
    name = dirs[i]
    old_path = os.path.join(src_path, name)
    new_path = os.path.join(tgt_path, name)
    try:
        print(i+1, "/", len(dirs), new_path, "error:", len(error_files))
        load_audio(old_path, new_path)
        i += 1
    except Exception as e:
        print(e)
        error_files.append(old_path)

print("Error:", len(error_files))
print(error_files)