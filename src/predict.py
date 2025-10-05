import argparse
import joblib
import os
import cv2
from features import extract_features




def predict_image(model_path, image_path):
model = joblib.load(model_path)
feat = extract_features(image_path).reshape(1, -1)
pred = model.predict(feat)[0]
label = 'Dog' if pred == 1 else 'Cat'
print(f"{image_path} -> {label}")




def predict_folder(model_path, folder_path):
for file in os.listdir(folder_path):
if file.endswith('.jpg'):
predict_image(model_path, os.path.join(folder_path, file))




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='path to trained model')
parser.add_argument('--image', help='path to single image')
parser.add_argument('--folder', help='path to folder of images')
args = parser.parse_args()


if args.image:
predict_image(args.model, args.image)
elif args.folder:
predict_folder(args.model, args.folder)
else:
parser.error('Please provide either --image or --folder')
