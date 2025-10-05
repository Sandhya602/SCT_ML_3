import os
import numpy as np
from tqdm import tqdm
from features import extract_features


def load_images(data_dir):
X, y = [], []
for file in tqdm(os.listdir(data_dir)):
if file.endswith('.jpg'):
label = 1 if 'dog' in file else 0
path = os.path.join(data_dir, file)
try:
feat = extract_features(path)
X.append(feat)
y.append(label)
except Exception as e:
print(f"[WARN] Skipping {file}: {e}")
return np.array(X), np.array(y)
