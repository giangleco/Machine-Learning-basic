from seaborn import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
TEST_SIZE =0.2 # test_size=20%
RANDOM_STATE=42 # repeat don't change

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#1, Load data
def load_data():
    try:
        tips=load_dataset("tips")
        #Onehotcoding
        df_encoded = pd.get_dummies(tips, columns=['sex', 'smoker', 'day', 'time'], drop_first=True)
        X = df_encoded.drop(columns=['total_bill']) # drop total
        y = df_encoded['total_bill']
        print(y)
        # split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        logger.info("Data loaded and split successfully...")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None, None
#2, Train model
def train_model(X_train,y_train):
    # Selection model
    try:
        model = LinearRegression()
        # Train model
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
def evaluate_model(y_true, y_pred):
    mse= mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

#5,Visualize
def plot_predictions(y_true, y_pred, title):
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # y=x
    plt.show()

#6, function Main
def main():
    try:
        logger.info("Starting the regression... ")
        X_train, X_test, y_train, y_test=load_data()
        model =train_model(X_train, y_train)
        y_train_pred = predict(model, X_train)
        y_test_pred = predict(model, X_test)

        mse_train, r2_train = evaluate_model(y_train, y_train_pred)
        mse_test, r2_test = evaluate_model(y_test, y_test_pred)

        print(f"MSE Train (-): {mse_train:.3f}, R2 Train (+): {r2_train:.3f}")
        print(f"MSE Test  (-): {mse_test:.3f}, R2 Test  (+): {r2_test:.3f}")

        plot_predictions(y_test, y_test_pred, "Test Set Predictions")

    except Exception as e:
        logger.error(f"Unexpected error in main(): {e}")
# Running
if __name__ == "__main__":
    main()

