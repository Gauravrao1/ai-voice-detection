"""
Simple training script for voice detection model
Run this to train your own model
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def generate_dummy_data(n_samples=1000):
    """
    Generate dummy training data
    Replace this with real audio feature extraction
    """
    # 36 features (matching our feature extractor)
    X = np.random.randn(n_samples, 36)
    
    # Labels: 0 = HUMAN, 1 = AI_GENERATED
    y = np.random.randint(0, 2, n_samples)
    
    # Add some patterns to make it learnable
    # AI voices: lower variance in features
    ai_mask = y == 1
    X[ai_mask] = X[ai_mask] * 0.5  # Lower variance
    
    return X, y

def train_model():
    """
    Train the voice detection model
    """
    print("🚀 Starting model training...")
    
    # Generate or load data
    X, y = generate_dummy_data(n_samples=2000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Training complete!")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['HUMAN', 'AI_GENERATED']))
    
    # Save model
    os.makedirs('ml_model/saved_models', exist_ok=True)
    model_path = 'ml_model/saved_models/voice_detector.pkl'
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")

if __name__ == "__main__":
    train_model()