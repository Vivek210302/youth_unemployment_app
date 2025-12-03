import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Set paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'youth_unemployment_global.csv')
model_path = os.path.join(current_dir, 'unemployment_model.joblib')
encoder_path = os.path.join(current_dir, 'country_encoder.joblib')

def train_and_save_model():
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Basic Preprocessing
    # Drop rows with missing target or essential predictors if any
    df = df.dropna(subset=['YouthUnemployment', 'Country', 'Year'])

    print(f"Data shape after cleaning: {df.shape}")

    # Feature Engineering
    # Encode Country
    le = LabelEncoder()
    df['Country_Encoded'] = le.fit_transform(df['Country'])

    # Features and Target
    X = df[['Country_Encoded', 'Year']]
    y = df['YouthUnemployment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluation
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance - MSE: {mse:.2f}, R2: {r2:.2f}")

    # Visualization with Seaborn (Actual vs Predicted for a sample)
    print("Generating evaluation plot...")
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    
    # Create a DataFrame for plotting
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    # Sample 100 points for clearer visualization
    results_sample = results_df.sample(n=min(100, len(results_df)), random_state=42)
    
    sns.scatterplot(x='Actual', y='Predicted', data=results_sample)
    plt.plot([results_sample.min().min(), results_sample.max().max()], 
             [results_sample.min().min(), results_sample.max().max()], 
             'r--', lw=2)
    plt.title('Actual vs Predicted Youth Unemployment (Sample)')
    plt.xlabel('Actual Unemployment Rate')
    plt.ylabel('Predicted Unemployment Rate')
    
    plot_path = os.path.join(current_dir, 'model_evaluation_plot.png')
    plt.savefig(plot_path)
    print(f"Evaluation plot saved to {plot_path}")

    # Save Model and Encoder
    print("Saving model and encoder...")
    joblib.dump(rf, model_path)
    joblib.dump(le, encoder_path)
    print("Done!")

if __name__ == "__main__":
    train_and_save_model()
