import os
from sklearn.decomposition import PCA
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Preprocess the Images
def load_images_from_folder(folder, label, image_size=(64, 64)):
    images = []
    labels = []
    file_list = os.listdir(folder)
    for filename in tqdm(file_list, desc=f"Loading {label} images"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = img.flatten()  # Flatten the image
            images.append(img)
            labels.append(label)
    return images, labels

cat_images, cat_labels = load_images_from_folder('C:/Users/Anjali Patel/PycharmProjects/PRODIGY_ML_03/Dataset/cats', label=0)
dog_images, dog_labels = load_images_from_folder('C:/Users/Anjali Patel/PycharmProjects/PRODIGY_ML_03/Dataset/dogs', label=1)

# Combine the data
images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

pca = PCA(n_components=100) 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

svm = SVC(kernel='linear')
svm.fit(X_train_pca, y_train)

# Evaluate the Model
y_pred = svm.predict(X_test_pca)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))