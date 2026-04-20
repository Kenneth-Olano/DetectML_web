import os
import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

TRAIN_FILE = "../../entropy_training_features_dataset.csv"

FEATURE_COLUMNS = [
    'Perplexity', 'Burstiness',
    'Word_Entropy', 'Char_Entropy', 'Repetition_Rate',
    'Semantic_Neutral', 'Semantic_Passionate', 'Semantic_HighTemp',
    'Lexical_Neutral', 'Lexical_Passionate', 'Lexical_HighTemp',
    'Perplexity_Delta_Neutral', 'Perplexity_Delta_Passionate', 'Perplexity_Delta_HighTemp',
    'Burstiness_Delta_Neutral', 'Burstiness_Delta_Passionate', 'Burstiness_Delta_HighTemp',
    'Word_Entropy_Delta_Neutral', 'Word_Entropy_Delta_Passionate', 'Word_Entropy_Delta_HighTemp',
    'Char_Entropy_Delta_Neutral', 'Char_Entropy_Delta_Passionate', 'Char_Entropy_Delta_HighTemp',
    'Repetition_Delta_Neutral', 'Repetition_Delta_Passionate', 'Repetition_Delta_HighTemp'
]

def load_split(filepath):
    df = pd.read_csv(filepath)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [WARN] missing columns: {missing}")
    X = df[FEATURE_COLUMNS].values
    y = df['Is_AI'].values
    return X, y

def main():
    print("Loading dataset...")
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return

    X_train, y_train = load_split(TRAIN_FILE)

    print("Training final model on full training set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=99,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    print("Model trained.")

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/rf_scaler.pkl')
    joblib.dump(model, 'models/random_forest_classifier.pkl')
    print("Scaler and Model saved to models/.")

if __name__ == "__main__":
    main()
