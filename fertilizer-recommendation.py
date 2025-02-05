import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class FertilizerRecommendationSystem:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.le_soil = LabelEncoder()
        self.le_crop = LabelEncoder()
        self.le_fertilizer = LabelEncoder()
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the fertilizer dataset."""
        # Load the data
        df = pd.read_csv(data_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Encode categorical variables
        df['Soil Type'] = self.le_soil.fit_transform(df['Soil Type'])
        df['Crop Type'] = self.le_crop.fit_transform(df['Crop Type'])
        df['Fertilizer Name'] = self.le_fertilizer.fit_transform(df['Fertilizer Name'])
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training."""
        # Select features and target
        features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
                   'Nitrogen', 'Potassium', 'Phosphorous']
        X = df[features]
        y = df['Fertilizer Name']
        
        return X, y
    
    def train_model(self, X, y):
        """Train the Random Forest model."""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, X_test, y_test
    
    def predict_fertilizer(self, temperature, humidity, moisture, soil_type, crop_type, 
                          nitrogen, potassium, phosphorous):
        """Predict fertilizer for given input parameters."""
        # Encode soil and crop type
        soil_encoded = self.le_soil.transform([soil_type])[0]
        crop_encoded = self.le_crop.transform([crop_type])[0]
        
        # Create input array
        input_data = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded, 
                               nitrogen, potassium, phosphorous]])
        
        # Make prediction
        prediction = self.model.predict(input_data)
        
        # Decode prediction
        recommended_fertilizer = self.le_fertilizer.inverse_transform(prediction)[0]
        
        return recommended_fertilizer
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        features = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
                   'Nitrogen', 'Potassium', 'Phosphorous']
        importance_scores = dict(zip(features, self.model.feature_importances_))
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))

# Example usage
def main():
    # Initialize the system
    fertilizer_system = FertilizerRecommendationSystem()
    
    # Load and preprocess data
    df = fertilizer_system.load_and_preprocess_data('Fertilizer Prediction.csv')
    
    # Prepare features
    X, y = fertilizer_system.prepare_features(df)
    
    # Train model and get accuracy
    accuracy, X_test, y_test = fertilizer_system.train_model(X, y)
    print(f"Model Accuracy: {accuracy:.2%}")
    
    # Example prediction
    sample_prediction = fertilizer_system.predict_fertilizer(
        temperature=26,
        humidity=52,
        moisture=38,
        soil_type='Sandy',
        crop_type='Maize',
        nitrogen=37,
        potassium=0,
        phosphorous=0
    )
    print(f"\nRecommended Fertilizer: {sample_prediction}")
    
    # Get feature importance
    importance = fertilizer_system.get_feature_importance()
    print("\nFeature Importance:")
    for feature, score in importance.items():
        print(f"{feature}: {score:.4f}")

if __name__ == "__main__":
    main()
