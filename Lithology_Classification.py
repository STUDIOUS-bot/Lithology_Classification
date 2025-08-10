import pandas as pd
import lasio as ls
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load your data (assume already in a DataFrame)
df = pd.read_csv("C:\Users\kabha\OneDrive\Desktop\ONGC\Project1\litho_data")

# Drop or fill missing values
df = df.dropna(subset=['GR', 'RHOB', 'NPHI', 'DT', 'PE', 'Lithology'])

# Encode labels
le = LabelEncoder()
df['Lithology_encoded'] = le.fit_transform(df['Lithology'])

# Features and Target
X = df[['GR', 'RHOB', 'NPHI', 'DT', 'PE']]
y = df['Lithology_encoded']

# Normalize inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)



clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))



cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation=45)
plt.title("Lithology Confusion Matrix")
plt.tight_layout()
plt.show()

feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh', title='Feature Importance')
plt.tight_layout()
plt.show()