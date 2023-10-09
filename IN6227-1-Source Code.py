import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split





# Reading the data
# u.data file
df_names = ['age', 'workclass', 'fnlwgt', 'education','education-num','marital-status','occupation','relationship',
           'race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
df = pd.read_csv('/Users/kaihuaguo/Documents/IN6227 Assignment 1/IN6227-assignment-1/census+income/adult.data', names=df_names)




# Display the first 6 rows of the DataFrame
print(df.head(6))




total_missing_values = df.isna().sum().sum()
print(f"Total number of missing values in the DataFrame: {total_missing_values}")






# Coding the target variable income into dummy variable

df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# Print the first few rows to verify the coding
print(df.head())









# Define the list of categorical columns to be label-encoded
categorical_columns = ['education', 'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column and replace them
for column in categorical_columns:
    df[column + '_encoded'] = label_encoder.fit_transform(df[column])
# Drop the original categorical columns
df.drop(columns=categorical_columns, inplace=True)

# Print the first few rows to verify the encoding and replacement
print(df.head())





# Split the data into a training set (80%) and a testing set (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Training set shape:", train_df.shape)
print("Testing set shape:", test_df.shape)




# Separate the features (X) and the target variable (y)
X_train = train_df.drop(columns=['income'])
y_train = train_df['income']

X_test = test_df.drop(columns=['income'])
y_test = test_df['income']
print(y_train.head())




# Create and train the Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust n_estimators as needed
random_forest_model.fit(X_train, y_train)




# predictions on the training set to assess accuracy
test_predictions = random_forest_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Testing Accuracy:", test_accuracy)




# Make predictions on the training set
test_predictions = random_forest_model.predict(X_test)

# Calculate precision
precision = precision_score(y_test, test_predictions)

# Print the precision score
print("Testing Precision:", precision)






# Get the predicted probabilities for the positive class (class 1)
y_pred_prob = random_forest_model.predict_proba(X_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate the Area Under the ROC Curve (AUC)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()




# Extract feature importances from the trained Random Forest model
feature_importances = random_forest_model.feature_importances_




# Create a DataFrame to display feature names and their importance scores
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})




# Sort the DataFrame by importance scores in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the top N important features (adjust N as needed)
top_n = 10  # You can change this to show the top N features
print(f"Top {top_n} Important Features:")
print(importance_df.head(top_n))