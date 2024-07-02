import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools


def handle_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)
        else:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.figure(figsize=(12, 8))
    sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Classification Report')
    plt.show()

def main(train_path, validation_path, test_path, output_path):
    # Load datasets
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    # Replace '?' with NaN
    train_df.replace('?', np.nan, inplace=True)
    validation_df.replace('?', np.nan, inplace=True)
    test_df.replace('?', np.nan, inplace=True)

    # Handle missing values
    handle_missing_values(train_df)
    handle_missing_values(validation_df)
    handle_missing_values(test_df)

    # Separate features and target variable
    X_train = train_df.drop(columns=['y'])
    y_train = train_df['y']
    X_validation = validation_df.drop(columns=['y'])
    y_validation = validation_df['y']
    X_test = test_df.drop(columns=['y'])
    y_test = test_df['y']
    # Drop 'index' column if it exists
    if 'index' in X_validation.columns:
        X_validation = X_validation.drop(columns=['index'])
    if 'index' in X_test.columns:
        X_test = X_test.drop(columns=['index'])

    # Combine datasets for encoding
    combined_df = pd.concat([X_train, X_validation, X_test], axis=0)

    # Apply one-hot encoding to the combined dataset
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    combined_encoded = encoder.fit_transform(combined_df)

    # Split the encoded data back into training, validation, and test sets
    n_train = X_train.shape[0]
    n_validation = X_validation.shape[0]
    X_train_encoded = combined_encoded[:n_train]
    X_validation_encoded = combined_encoded[n_train:n_train + n_validation]
    X_test_encoded = combined_encoded[n_train + n_validation:]

    # Initialize and train the classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_encoded, y_train)

    # Predict on the validation set and evaluate
    y_validation_pred = clf.predict(X_validation_encoded)
    accuracy = accuracy_score(y_validation, y_validation_pred)
    report = classification_report(y_validation, y_validation_pred, output_dict=True)
    print(f'Validation Accuracy: {accuracy:.2f}')
    print('Classification Report for validation data:\n', classification_report(y_validation, y_validation_pred))

    # Plot the classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Classification Report for validation data')
    plt.show()

    # Plot the confusion matrix
    cm = confusion_matrix(y_validation, y_validation_pred)
    plot_confusion_matrix(cm, classes=['No', 'Yes'])
    plt.title("confusion matrix for validation data")
    plt.show()

    # Predict on the test set
    y_test_pred = clf.predict(X_test_encoded)

    # Predict on the validation set and evaluate

    accuracy = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, output_dict=True)
    print(f'test Accuracy: {accuracy:.2f}')
    print('Classification Report for test data:\n', classification_report(y_test, y_test_pred))

    # Plot the classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Classification Report for test data')
    plt.show()

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred  )
    plot_confusion_matrix(cm, classes=['No', 'Yes'])
    plt.title("confusion matrix for test data ")
    plt.show()

    # Save the predictions to a CSV file
    output_df = pd.DataFrame({'index': test_df.index, 'y_pred': y_test_pred})
    output_df.to_csv(output_path, index=False)
    print(f'Test predictions saved to {output_path}')

    # Plot the predictions
    sns.countplot(y_test_pred)
    plt.title('Test Set Predictions')
    plt.xlabel('Predicted Label')
    plt.ylabel('Count')
    plt.show()


# Run the main function with appropriate file paths
main('train.csv', 'validation.csv', 'test.csv', 'output_predictions.csv')