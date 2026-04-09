from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import numpy as np
import joblib

CSV_PATH = Path("data/classifier/kp_data.csv")

data = np.loadtxt(CSV_PATH, delimiter=",")

X = data[:, :-1]
y = data[:, -1]
print(X.shape,y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Accuracy:", model.score(X_test, y_test))

joblib.dump(model, "models/classifier/posture_model.pkl")