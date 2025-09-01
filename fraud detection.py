# fraud_detection.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt




# Load dataset
df = pd.read_csv('C:/Users/USER/Downloads/Data Science ass1/Question 2 Datasets.csv')

# Feature Engineering
df['Time_Category'] = pd.cut(df['Time_Hour'], bins=[0,6,12,18,24], labels=['Night','Morning','Afternoon','Evening'])
df['Amount_Normalized'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df['High_Amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)

# Prepare features for clustering
X_cluster = df[['Amount', 'Time_Hour', 'Location', 'Merchant']]
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Amount', 'Time_Hour']),
    ('cat', OneHotEncoder(), ['Location', 'Merchant'])
])
X_cluster_processed = preprocessor.fit_transform(X_cluster)

# K-Means Clustering
wcss = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster_processed)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 10), wcss)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.tight_layout()
plt.show()


# Filter labeled data
df_labeled = df[df['Is_Fraud (Labeled Subset)'] != -1]
X_class = df_labeled[['Amount', 'Time_Hour', 'Location', 'Merchant']]
y_class = df_labeled['Is_Fraud (Labeled Subset)']

# Preprocess classification features
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Amount', 'Time_Hour']),
    ('cat', OneHotEncoder(), ['Location', 'Merchant'])
])
X_class_processed = preprocessor.fit_transform(X_class)


# Resample to balance classes
smote = SMOTE(random_state=42, k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X_class_processed, y_class)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train Random Forest
clf_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# Evaluate
f1_rf = f1_score(y_test, y_pred_rf)
print(f"Random Forest F1 Score: {f1_rf:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))




# Prepare labeled data for classification
df_labeled = df[df['Is_Fraud (Labeled Subset)'] != -1]
X_class = df_labeled[['Amount', 'Time_Hour', 'Location', 'Merchant']]
y_class = df_labeled['Is_Fraud (Labeled Subset)']
X_class_processed = preprocessor.transform(X_class)


#plot random forest
y_proba = clf_rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.tight_layout()
plt.show()


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_class_processed, y_class, test_size=0.2, random_state=42)

# Naïve Bayes Classification
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(cm)

# Cross-validation
cv_scores = cross_val_score(clf, X_class_processed, y_class, cv=10, scoring='f1')
print(f"Cross-Validated F1 Score: {cv_scores.mean():.2f}")



