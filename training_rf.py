import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# 1. Load dataset
df = pd.read_csv("D:/skripsi/dataset_rfakhir.csv", sep=';')

# 2. Label encoding untuk kolom kategori
label_encoders = {}
for col in ['Jenis_Obyek', 'Sensor_Relevan', 'Keputusan']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. Pisahkan fitur dan label
X = df.drop(columns=['Keputusan'])
y = df['Keputusan']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Training model
model = RandomForestClassifier(n_estimators=100, random_state=42)
start_time_train = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time_train
print(f"=== Training Time: {train_time:.2f} seconds ===")

# 6. Perkiraan Inference Time Selama Training
n_samples_inference = min(100, len(X_test))
inference_samples = X_test.sample(n_samples_inference, random_state=42)
inference_start_time = time.time()
model.predict(inference_samples)
inference_time_train = (time.time() - inference_start_time) / n_samples_inference * 1000
print(f"\n=== Perkiraan Inference Time Selama Training: {inference_time_train:.2f} ms per sample ===")

# 7. Evaluasi Training
y_pred_train = model.predict(X_train)
print("\n=== Evaluation Metrics (Training Set) ===")
print(f"Accuracy : {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Precision : {precision_score(y_train, y_pred_train, average='weighted', zero_division=0):.4f}")
print(f"Recall : {recall_score(y_train, y_pred_train, average='weighted', zero_division=0):.4f}")
print("\nClassification Report (Training):")
print(classification_report(y_train, y_pred_train, target_names=label_encoders['Keputusan'].classes_))

# Confusion Matrix Training
cm_train = confusion_matrix(y_train, y_pred_train)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoders['Keputusan'].classes_,
            yticklabels=label_encoders['Keputusan'].classes_)
plt.xlabel('Predicted (Training)')
plt.ylabel('Actual (Training)')
plt.title('Confusion Matrix (Training Set)')
plt.tight_layout()
plt.show()

# 8. Evaluasi Testing
y_pred_test = model.predict(X_test)
print("\n=== Evaluation Metrics (Testing Set) ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_test, average='weighted', zero_division=0):.4f}")
print(f"Recall : {recall_score(y_test, y_pred_test, average='weighted', zero_division=0):.4f}")
print("\nClassification Report (Testing):")
print(classification_report(y_test, y_pred_test, target_names=label_encoders['Keputusan'].classes_))

# Confusion Matrix Testing
cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoders['Keputusan'].classes_,
            yticklabels=label_encoders['Keputusan'].classes_)
plt.xlabel('Predicted (Testing)')
plt.ylabel('Actual (Testing)')
plt.title('Confusion Matrix (Testing Set)')
plt.tight_layout()
plt.show()

# 9. Feature Importance
importances = model.feature_importances_
feat_names = X.columns
fi_df = pd.DataFrame({'Fitur': feat_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\n=== Feature Importance ===")
print(fi_df)

plt.figure(figsize=(8, 5))
sns.barplot(data=fi_df, x='Importance', y='Fitur', palette='viridis')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# 10. Average Inference Time (10 percobaan)
inference_times_testing = []
for _ in range(10):
    sample = X_test.sample(1)
    start_inf = time.time()
    _ = model.predict(sample)
    inference_times_testing.append(time.time() - start_inf)

avg_inference_time_testing_ms = np.mean(inference_times_testing) * 1000
print(f"\nAverage Inference Time per Sample (Testing): {avg_inference_time_testing_ms:.2f} ms")

# 11. Save model dan encoder
joblib.dump(model, "D:/skripsi/modelRFreal.joblib")
joblib.dump(label_encoders, "D:/skripsi/labelRFreal.joblib")
print("\nModel dan label encoder berhasil disimpan.")


