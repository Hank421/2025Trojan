import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib

# ---------------------------
# Argument Parsing
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a feed-forward NN for Trojan detection."
    )
    parser.add_argument(
        '--train-files', nargs='+', required=True,
        help="List of CSV files for training."
    )
    parser.add_argument(
        '--test-files', nargs='+', required=True,
        help="List of CSV files for testing."
    )
    parser.add_argument(
        '--epochs', type=int, default=20,
        help="Number of training epochs."
    )
    parser.add_argument(
        '--batch-size', type=int, default=512,
        help="Training batch size."
    )
    return parser.parse_args()


# ---------------------------
# Data Loading & Preprocessing
# ---------------------------

def load_design_files(file_list):
    dfs = []
    for path in file_list:
        df = pd.read_csv(path)
        df['design_id'] = os.path.basename(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def preprocess(df, gate_encoder=None):
    # Encode gate_type
    if gate_encoder is None:
        gate_encoder = LabelEncoder()
        gate_encoder.fit(df['gate_type'])
    df['gate_type'] = gate_encoder.transform(df['gate_type'])
    
    # Map labels
    df['label'] = df['label'].map({'Not_Trojan': 0, 'Trojan': 1})
    
    # Drop non-feature cols
    drop_cols = ['gate_name','output','inputs','gate_numbers','design_id']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Coerce numeric & drop NaNs
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df, gate_encoder


# ---------------------------
# Model Definition
# ---------------------------

def build_classifier(input_dim):
    inp = tf.keras.Input(shape=(input_dim,), name='features')
    x = tf.keras.layers.Dense(128, activation='relu')(inp)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='trojan_prob')(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name='TrojanClassifier')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ---------------------------
# Train & Evaluate
# ---------------------------

def train_and_evaluate(train_files, test_files, epochs, batch_size):
    # Load & preprocess train
    df_train = load_design_files(train_files)
    df_train, gate_encoder = preprocess(df_train)
    
    # Load & preprocess test
    df_test = load_design_files(test_files)
    df_test, _ = preprocess(df_test, gate_encoder)
    
    # Split features/labels
    X_train = df_train.drop('label', axis=1).values
    y_train = df_train['label'].values
    X_test  = df_test.drop('label', axis=1).values
    y_test  = df_test['label'].values
    
    # Build model
    model = build_classifier(input_dim=X_train.shape[1])
    model.summary()
    
    # Fit
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    
    # Save artifacts
    model.save('classifier_model.h5')
    print("Model saved to 'classifier_model.h5'")
    joblib.dump(gate_encoder, 'gate_type_encoder.pkl')
    print("LabelEncoder saved to 'gate_type_encoder.pkl'")
    
    # Evaluate on test set
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    
    print("=== Test Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Not_Trojan','Trojan']))
    print("Trojan F1-score:", f1_score(y_test, y_pred, pos_label=1))
    
    # Save test predictions
    out = pd.DataFrame(df_test)
    out['predicted'] = y_pred
    out['predicted'] = out['predicted'].map({0:'Not_Trojan',1:'Trojan'})
    out.to_csv('test_predictions_output.csv', index=False)
    print("Test predictions saved to 'test_predictions_output.csv'")


def main():
    args = parse_args()
    train_and_evaluate(
        train_files   = args.train_files,
        test_files    = args.test_files,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
    )


if __name__ == '__main__':
    main()
