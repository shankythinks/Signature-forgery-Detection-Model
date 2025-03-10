import os
import cv2 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import local_binary_pattern, hog
from scipy.fft import fft2
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

print("Importing libraries completed.")

# Paths to datasets
real_signatures_path = "real"  # Path to real signatures folder
forged_signatures_path = "forged"  # Path to forged signatures folder

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Noise reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Binary conversion using Otsu's thresholding
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours to crop signature
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]

    # Align signature horizontally by rotation
    coords = np.column_stack(np.where(binary_img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return img

print("Preprocessing function ready.")

def extract_features(img):
    features = []

    # Ratio
    h, w = img.shape
    features.append(h / w)

    # Standard deviation
    features.append(np.std(img))

    # Centroid
    binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    moments = cv2.moments(binary_img)
    if moments["m00"] != 0:
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
    else:
        centroid_x, centroid_y = 0, 0
    features.extend([centroid_x / w, centroid_y / h])

    # Eccentricity and Solidity
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0
        x, y, w, h = cv2.boundingRect(cnt)
        eccentricity = np.sqrt(1 - (min(w, h) / max(w, h))**2) if h != 0 and w != 0 else 0
        features.extend([eccentricity, solidity])

    # Contours
    features.append(len(contours))

    # Stroke Width and Length
    stroke_width = np.mean([cv2.arcLength(c, True) / (cv2.contourArea(c) + 1e-5) for c in contours])
    stroke_length = np.mean([cv2.arcLength(c, True) for c in contours])
    features.extend([stroke_width, stroke_length])

    # Skewness and Kurtosis
    features.append(pd.Series(img.flatten()).skew())
    features.append(pd.Series(img.flatten()).kurtosis())

    # Histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    features.extend(hist.flatten() / np.sum(hist))

    # LBP (Local Binary Patterns)
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    features.extend(lbp_hist / np.sum(lbp_hist))

    # Gradient Features (HOG)
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
    features.extend(hog_features)

    # Fourier Transform Features
    f_transform = np.abs(fft2(img))
    features.append(np.mean(f_transform))
    features.append(np.std(f_transform))

    return features

print("Feature extraction function ready.")

def prepare_and_save_csv(dataset_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        output_csv = os.path.join(output_folder, os.path.splitext(file)[0] + ".csv")
        
        # Skip processing if CSV already exists
        if os.path.exists(output_csv):
            print(f"CSV for {file} already exists. Skipping.")
            continue

        if file.endswith(".png") or file.endswith(".jpg"):
            img = preprocess_image(file_path)
            features = extract_features(img)
            df = pd.DataFrame([features])
            df.to_csv(output_csv, index=False)
            print(f"Processed and saved CSV for {file}.")

print("CSV preparation function ready.")

# Create folders for extracted data
parent_folder = "extracted_data"
real_csv_folder = os.path.join(parent_folder, "real_csv")
forged_csv_folder = os.path.join(parent_folder, "forged_csv")
test_csv_folder = os.path.join(parent_folder, "test_csv")
os.makedirs(test_csv_folder, exist_ok=True)

prepare_and_save_csv(real_signatures_path, real_csv_folder)
prepare_and_save_csv(forged_signatures_path, forged_csv_folder)

print(f"Feature CSV files for real signatures saved in: {real_csv_folder}")
print(f"Feature CSV files for forged signatures saved in: {forged_csv_folder}")

# Load dataset
print("Loading and combining dataset...")
real_csv_files = [os.path.join(real_csv_folder, f) for f in os.listdir(real_csv_folder) if f.endswith(".csv")]
forged_csv_files = [os.path.join(forged_csv_folder, f) for f in os.listdir(forged_csv_folder) if f.endswith(".csv")]

real_data = pd.concat([pd.read_csv(f) for f in real_csv_files], ignore_index=True)
forged_data = pd.concat([pd.read_csv(f) for f in forged_csv_files], ignore_index=True)

data = pd.concat([real_data.assign(label=1), forged_data.assign(label=0)], ignore_index=True)
data.to_csv("combined_signature_features.csv", index=False)
print("Combined dataset saved as 'combined_signature_features.csv'.")

# Split data for model training
print("Splitting dataset...")
X = data.drop("label", axis=1)
y = data["label"]

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the model
import joblib
joblib.dump(model, "signature_forgery_detector_rf.pkl")
print("Model saved as 'signature_forgery_detector_rf.pkl'.")

# Prediction function
def predict_signature(image_path):
    img = preprocess_image(image_path)
    features = np.array(extract_features(img)).reshape(1, -1)
    features = scaler.transform(features)

    # Save test image features to test_csv folder
    test_csv_path = os.path.join(test_csv_folder, os.path.splitext(os.path.basename(image_path))[0] + ".csv")
    pd.DataFrame(features).to_csv(test_csv_path, index=False)
    print(f"Test image features saved at: {test_csv_path}")

    prediction = model.predict(features)
    return "Genuine" if prediction[0] == 1 else "Forged"

# Test the model on a new sample
test_image_path = "data/test_image.png"  # Replace with a test signature path
print(f"The signature is: {predict_signature(test_image_path)}")

