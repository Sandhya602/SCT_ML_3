import argparse
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from load_data import load_images




def train_model(data_dir, output_path):
print(f"[INFO] Loading dataset from {data_dir}...")
X, y = load_images(data_dir)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Pipeline([
('scaler', StandardScaler()),
('svm', SVC(kernel='rbf', C=10, gamma='scale'))
])


print("[INFO] Training SVM model...")
model.fit(X_train, y_train)


preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"[INFO] Accuracy: {acc:.4f}")
print(classification_report(y_test, preds, target_names=['Cat', 'Dog']))


joblib.dump(model, output_path)
print(f"[INFO] Model saved to {output_path}")




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='path to training data directory')
parser.add_argument('--output', default='svm_model.joblib', help='output model filename')
args = parser.parse_args()


train_model(args.data, args.output)
