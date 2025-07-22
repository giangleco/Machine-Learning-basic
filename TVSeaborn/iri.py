import logging
from seaborn import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
TEST_SIZE =0.2 # test_size=20%
RANDOM_STATE=42 # repeat don't change

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger=logging.getLogger(__name__)

#1, Load data
def load_data():
    try:
        iris=load_dataset("iris")
        # LabelEncoder
        le=LabelEncoder()
        y=le.fit_transform(iris["species"])
        X= iris.drop("species", axis=1)
        X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=TEST_SIZE, random_state=42)
        logger.info("Data loaded and split successfully...")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error loading dat: {e}")

#2, Train model
def train_model(X_train, y_train):
    try:
        model=LogisticRegression()
        model.fit(X_train, y_train)
        logger.info("Model train successfully")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

#3, Predict on testing data
def predict(model, X):
    return model.predict(X)

#4, Evaluate the model performance on training set
def evaluate_model(y_true, y_pred, y_proba):# Y_proba để tính xác xuất dự đoán mô hình
    accuracy=accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')  # macro = trung bình giữa các lớp
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    try:
        # ROC AUC cho đa lớp với One-vs-Rest
        roc = roc_auc_score(y_true, y_proba, multi_class='ovr')
    except ValueError as e:
        roc = None
        print(f"Cannot calculate ROC AUC: {e}")

    return accuracy, precision, recall, f1, roc
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()
def main():
    try:
        logger.info("Starting the classification... ")
        X_train, X_test, y_train, y_test=load_data()
        model =train_model(X_train,y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)

        accuracy_train,precision_train, recall_train, f1_train, roc_train = evaluate_model(y_train, y_train_pred, y_train_proba)
        accuracy_test,precision_test, recall_test, f1_test, roc_test = evaluate_model(y_test, y_test_pred, y_test_proba)

        print(f"--- Training ---")
        print(f"Accuracy     : {accuracy_train:.3f}")
        print(f"Precision    : {precision_train:.3f}")
        print(f"Recall       : {recall_train:.3f}")
        print(f"F1-Score     : {f1_train:.3f}")
        print(f"ROC AUC      : {roc_train:.3f}")

        print(f"\n--- Testing ---")
        print(f"Accuracy     : {accuracy_test:.3f}")
        print(f"Precision    : {precision_test:.3f}")
        print(f"Recall       : {recall_test:.3f}")
        print(f"F1-Score     : {f1_test:.3f}")
        print(f"ROC AUC      : {roc_test:.3f}")
        plot_confusion_matrix(y_test, y_test_pred, "Confusion Matrix - Test Set")

    except Exception as e:
        logger.error(f"Unexpected error in main(): {e}")

if __name__ == "__main__":
    main()