import pandas as pd
from pycaret.classification import load_model, predict_model
from scipy.stats import percentileofscore
import sys

class ChurnPredictor:
    """
    A class to predict customer churn probability and percentile.
    """
    def __init__(self, model_path='best_churn_model', train_prob_dist_path='train_probability_distribution.csv'):
        """
        Initializes the ChurnPredictor by loading the model and training data distribution.

        Args:
            model_path: Path to the saved PyCaret model.
            train_prob_dist_path: Path to the CSV file containing the training
                                 probability distribution ('Churn Probability' and 'Probability Percentile').
        """
        self.loaded_model = load_model(model_path, verbose=False)
        try:
            self.train_probability_distribution = pd.read_csv(train_prob_dist_path)
        except FileNotFoundError:
            print(f"Error: Training probability distribution file not found at {train_prob_dist_path}")
            print("Please ensure 'train_probability_distribution.csv' is in the same directory or provide the correct path.")
            sys.exit(1) # Exit if the training distribution file is not found


    def preprocess_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the necessary preprocessing steps to the input DataFrame
        to match the training data structure.

        Args:
            dataframe: Input DataFrame (in the format of new_unmodified_churn_data.csv).

        Returns:
            Preprocessed DataFrame ready for prediction.
        """
        # Make a copy to avoid modifying the original DataFrame
        df_processed = dataframe.copy()

        # Drop rows with missing TotalCharges (matching the training data preprocessing)
        df_processed.dropna(subset=['TotalCharges'], inplace=True)

        # Convert 'PhoneService' column
        df_processed['PhoneService'] = df_processed['PhoneService'].apply(lambda x: 1 if x == 'Yes' else 0)

        # Convert 'Contract' column using mapping
        contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        df_processed['Contract'] = df_processed['Contract'].map(contract_mapping)

        # Convert 'PaymentMethod' column using mapping
        payment_mapping = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
        df_processed['PaymentMethod'] = df_processed['PaymentMethod'].map(payment_mapping)

        # Calculate 'charge_per_tenure' column
        # Handle potential division by zero if tenure is 0
        df_processed['charge_per_tenure'] = df_processed.apply(
            lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else 0, axis=1
        )


        # Reorder columns to match the training data used by the model,
        # excluding the target 'Churn' column if present in the input.
        expected_columns = ['customerID', 'tenure', 'PhoneService', 'Contract', 'PaymentMethod',
                            'MonthlyCharges', 'TotalCharges', 'charge_per_tenure']

        # Drop columns not in the expected list (like 'Churn' if it exists in new data)
        extra_cols = [col for col in df_processed.columns if col not in expected_columns]
        df_processed.drop(columns=extra_cols, inplace=True)

        # Ensure all expected columns are present, if not, add them with a default value (e.g., 0)
        for col in expected_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0 # Or handle this case based on domain knowledge

        # Reorder the columns
        df_processed = df_processed[expected_columns]

        return df_processed

    def predict_churn_probability_and_percentile(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts churn probabilities and calculates their percentiles within
        the training data distribution for the input DataFrame.

        Args:
            dataframe: Pandas DataFrame (in the format of new_unmodified_churn_data.csv).

        Returns:
            Pandas DataFrame with 'customerID', 'Probability of Churn', and
            'Probability Percentile' columns.
        """
        # Preprocess the input data
        processed_df = self.preprocess_data(dataframe.copy()) # Pass a copy to preprocess_data

        # Predict churn probabilities using the loaded model
        predictions  = self.loaded_model.predict_proba(processed_df)

        predictions = pd.DataFrame(predictions, columns=['Probability_0', 'Probability_1'])

        # Extract churn probability
        # Assuming the probability of churn is for the 'Yes' class and column is 'Probability_1'
        if 'Probability_1' not in predictions.columns:
             print("Error: 'Probability_1' column not found in predictions.")
             print("Available columns:", predictions.columns)
             return pd.DataFrame(columns=['customerID', 'Probability of Churn', 'Probability Percentile'])

        churn_probabilities_df = predictions[['Probability_1']].copy()
        churn_probabilities_df.rename(columns={'Probability_1': 'Probability of Churn'}, inplace=True)

        # Calculate the percentile of each predicted probability within the training data distribution
        # Ensure the training distribution is loaded and has the correct column name.
        if 'Churn Probability' not in self.train_probability_distribution.columns:
            print("Error: 'Churn Probability' column not found in training distribution data.")
            return churn_probabilities_df.assign(**{'Probability Percentile': None}) # Add percentile column with None

        churn_probabilities_df['Probability Percentile'] = churn_probabilities_df['Probability of Churn'].apply(
            lambda x: percentileofscore(self.train_probability_distribution['Churn Probability'], x)
        )

        return churn_probabilities_df

if __name__ == '__main__':
    # Get file path from user, load data, predict, and print results
    data_file_path = input("Enter the path to the data file (data/new_unmodified_churn_data.csv): ")

    try:
        # data_file_path = 'data/new_churn_data_unmodified.csv'
    
        # Read the data from the provided file path
        input_df = pd.read_csv(data_file_path)

        # Assuming 'train_probability_distribution.csv' is saved in the current directory:
        predictor = ChurnPredictor(train_prob_dist_path='train_probability_distribution.csv')

        # Get predictions and percentiles
        churn_predictions_with_percentile = predictor.predict_churn_probability_and_percentile(input_df)

        print("\nChurn Predictions with Percentiles:")
        print(churn_predictions_with_percentile)

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")