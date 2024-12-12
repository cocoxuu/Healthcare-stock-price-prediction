import os
import pandas as pd
from model_training import prepare_training_data, train_lstm
from data_viz import plot_training_history, plot_predictions

def main():
    """
    Main function to orchestrate data preparation, model training, and visualization.
    """
    # Step 1: Load the dataset
    print("Step 1: Load Dataset")
    print("Current Working Directory:", os.getcwd())
    file_path = 'output/corps_ori_data.csv'
    df = pd.read_csv(file_path)
    
    # Step 2: Prepare training data
    print("\nStep 2: Prepare Training Data")
    features, target, target_scaled, target_scaler = prepare_training_data(df)
    
    # Step 3: Train the model
    print("\nStep 3: Train LSTM Model")
    results = train_lstm(
        features=features,
        target_scaled=target_scaled,
        target_scaler=target_scaler,
        epochs=50,
        timesteps=10
    )
    
    # Step 4: Evaluate model performance
    print("\nStep 4: Evaluate Model Performance")
    print("-------------------------")
    print(f"MSE: {results['metrics']['mse']:.2f}")
    print(f"RMSE: {results['metrics']['rmse']:.2f}")
    print(f"R² Score: {results['metrics']['r2']:.4f}")
    
    # Step 5: Visualize results
    print("\nStep 5: Visualize Results")
    plot_training_history(results['history'])
    plot_predictions(
        results['true_values'],
        results['predictions'],
        title='Healthcare Sector Price Predictions'
    )

if __name__ == "__main__":
    main()
