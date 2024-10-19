import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier


# Load the combined CSV file
combined_csv_path = './csv-prompt-splm-5/combined_predictions.csv'  # Replace with the actual path to your combined CSV file
df = pd.read_csv(combined_csv_path)

# Extract the true values (column 0) and the predicted values (columns 1-30)
true_values = df.iloc[:, 0].to_numpy()
predictions = df.iloc[:, 1:].to_numpy()

print(true_values)

# Split the data into 60% training, 20% validation, and 20% test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(predictions, true_values, test_size=0.2, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=10)

# Train a Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf=rf_clf.fit(X_train, y_train)



# Evaluate the model on the test set
y_test_pred = rf_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
f1=f1_score(y_test, y_test_pred, average=None, labels=[1, 0])
precision = precision_score(y_test, y_test_pred, average=None, labels=[1,0])
recall = recall_score(y_test, y_test_pred, average=None, labels=[1,0])
conf=confusion_matrix(y_test, y_test_pred, labels=[1, 0])
print(f'Random Test set accuracy: {test_accuracy:.4f}')
print(f'f1 score: ', f1)
print(f'precision score: ', precision)
print(f'recall score: ', recall)
print(f'confusion matrix: ', conf )
print("")


#ROC-AUC
y_score = rf_clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Step 6: Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# Step 7: Save the figure to a file
plt.savefig('rf_roc.png', dpi=300)


# save the trained model
with open('random_forest-prompt.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)
