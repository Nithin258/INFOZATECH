"""Task 1 - Iris Flower Classification (InfozaTech Data Science Internship)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1. Load
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print(df.head(), "\n")
print(df.describe().round(2), "\n")

# 2. Visualise
sns.pairplot(df.drop(columns="target"), hue="species", corner=True)
plt.savefig("outputs/iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Split (stratified so every species is equally represented)
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train three classifiers, compare with 5-fold CV on the training set
models = {
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
}
cv = {}
for name, m in models.items():
    cv[name] = cross_val_score(m, X_train, y_train, cv=5).mean()
    print(f"{name:15s} 5-fold CV accuracy: {cv[name]:.3f}")
best = max(cv, key=cv.get)
print(f"\nSelected model: {best}")

# 5. Evaluate on the held-out test set
model = models[best].fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}\n")
print(classification_report(y_test, pred, target_names=iris.target_names))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=iris.target_names, cmap="Blues")
plt.title(f"Iris - {best}")
plt.savefig("outputs/iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
