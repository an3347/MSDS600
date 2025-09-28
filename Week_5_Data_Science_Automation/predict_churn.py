import pandas as pd
from pycaret.classification import load_model, predict_model

def predict_churn_probability(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Loads the trained churn prediction model and predicts churn probabilities
    for the input DataFrame.

    Args:
        dataframe: Pandas DataFrame with the same structure as the training data
                   (excluding the target variable).

    Returns:
        Pandas DataFrame with 'customerID' and 'Probability of Churn' columns.
    """
    # Load the trained model
    loaded_model = load_model('best_churn_model', verbose=False)

    # # Predict churn probabilities
    predictions = loaded_model.predict_proba(dataframe)

    # Convert the predictions array to a DataFrame
    predictions_df = pd.DataFrame(predictions, columns=['Probability_0', 'Probability_1'])

    # Return the churn probability (probability of class 1)
    return predictions_df[['Probability_1']]


if __name__ == '__main__':
    try:
        # Load the new data (which already contains 'charge_per_tenure' based on inspection)
        new_data = pd.read_csv('data/new_churn_data.csv')
        churn_probabilities = predict_churn_probability(new_data)
        print("Churn Probabilities for new data:")
        print(churn_probabilities)
    except FileNotFoundError:
        print("Error: new_churn_data.csv not found. Please make sure the file is in the correct path.")