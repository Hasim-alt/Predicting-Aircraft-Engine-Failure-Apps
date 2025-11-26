import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def train():
    # Load data
    input_file = 'data/Processed_train_FD001.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        print("Please run 'data_preprocessing.py' before running this script.")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    X = df.drop('label', axis=1).values
    y = df['label'].values

    print(f"Features shape: {X.shape}")
    print(f"Label distribution (total): {np.bincount(y)}")

    # Train 80% | Test 20% split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to training data
    print(f"Before SMOTE: {np.bincount(y_train)}")
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE:  {np.bincount(y_train_res)}")

    # Scaling
    print("Scaling data...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler so the Flask app can use it later
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to 'scaler.pkl'")

    # Build ANN Architecture
    # Input dim is dynamic based on how many features survived the correlation filter
    input_dim = X_train_scaled.shape[1]

    model = Sequential()
    
    # Input Layer (Dynamic) -> Hidden Layer 1
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    
    # Hidden Layer 2
    model.add(Dense(256, activation='relu'))
    
    # Hidden Layer 3
    model.add(Dense(128, activation='relu'))
    
    # Output Layer (Sigmoid for binary classification)
    model.add(Dense(1, activation='sigmoid'))

    # Binary crossentropy loss function and Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train with EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    print("Starting training process...")
    history = model.fit(
        X_train_scaled, 
        y_train_res, 
        epochs=50, 
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stop]
    )

    model.save('engine_model.h5')
    print("\nSUCCESS: Model saved to 'engine_model.h5'")

if __name__ == "__main__":
    train()