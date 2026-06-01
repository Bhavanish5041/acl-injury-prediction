import shutil
import glob
import os

artifact_dir = '/home/bhavanish/.gemini/antigravity/brain/b0dad85b-c3ee-4b10-8d9f-2606e9889746'
os.makedirs(artifact_dir, exist_ok=True)

for file_path in glob.glob('output/*.png'):
    shutil.copy(file_path, artifact_dir)
    print(f"Copied {file_path} to {artifact_dir}")
