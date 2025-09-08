import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt

# load the dataset
data = pd.read_csv("spam.csv")
x = data.drop("spam", axis=1)
y = data["spam"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Train the logistic Regression model to classify emails as spam or not spam
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(x_test)
# evaluate the model using accuracy,confusion matrix,precision,recall and f1 score 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", {accuracy})
print("Precision:", {precision})
print("Recall:", {recall})
print("F1 Score:", {f1})
# Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()