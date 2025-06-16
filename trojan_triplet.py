import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model
from sklearn.metrics import classification_report, f1_score
import joblib
import argparse

# ---------------------------
# Function: Load, Train, Test
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train and test a model on Trojan detection data.")
    parser.add_argument('--algorithm', type=str, default='triplet_loss_nn',
                        help="Choose the algorithm to use for training (e.g., triplet_loss_nn)")
    return parser.parse_args()

# ---------------------------
# Triplet Model Architecture
# ---------------------------

def create_triplet_model(input_dim):
    # Define input layers for anchor, positive, and negative
    anchor_input = layers.Input(shape=(input_dim,), name="anchor_input")
    positive_input = layers.Input(shape=(input_dim,), name="positive_input")
    negative_input = layers.Input(shape=(input_dim,), name="negative_input")
    
    # Define a shared network for generating embeddings
    def embedding_network(input_layer):
        x = layers.Dense(128, activation='relu')(input_layer)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)  # Embedding output layer
        return x

    # Generate embeddings for the three inputs
    anchor_embedding = embedding_network(anchor_input)
    positive_embedding = embedding_network(positive_input)
    negative_embedding = embedding_network(negative_input)
    
    # Concatenate embeddings to get a final output tensor of shape (batch_size, 3 * embedding_dim)
    embeddings = layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=1)
    
    # Define the model
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=embeddings)
    
    return model

# ---------------------------
# Triplet Loss Function
# ---------------------------

def triplet_loss(y_true, y_pred, alpha=0.2):
    # Ensure y_pred has the shape (batch_size, 3 * embedding_dim)
    batch_size = tf.shape(y_pred)[0]
    embedding_dim = tf.shape(y_pred)[1] // 3  # Since the output is concatenated

    # Slice the embeddings for anchor, positive, and negative
    anchor_embedding = y_pred[:, 0:embedding_dim]
    positive_embedding = y_pred[:, embedding_dim:2*embedding_dim]
    negative_embedding = y_pred[:, 2*embedding_dim:3*embedding_dim]

    # Compute the pairwise distances (L2 distance)
    positive_distance = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), axis=1)
    negative_distance = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding), axis=1)
    
    # Triplet loss: Ensure anchor-positive distance is smaller than anchor-negative by margin alpha
    loss = tf.maximum(positive_distance - negative_distance + alpha, 0)
    return loss

# ---------------------------
# Load Design Files and Create Triplets
# ---------------------------

def load_design_files(filenames):
    dfs = []
    for file in filenames:
        df = pd.read_csv(file)
        df['design_id'] = os.path.basename(file)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Modified function to create triplets for both Trojan and Not_Trojan
def create_triplets(X, y):
    anchors, positives, negatives = [], [], []
    
    # Iterate through all samples
    for i in range(len(X)):
        if y[i] == 1:  # Trojan (positive class)
            # Find another Trojan as positive and a Not_Trojan as negative
            pos_idx = np.random.choice(np.where(y == 1)[0])
            neg_idx = np.random.choice(np.where(y == 0)[0])
            anchors.append(X.iloc[i].values)
            positives.append(X.iloc[pos_idx].values)
            negatives.append(X.iloc[neg_idx].values)
        else:  # Not_Trojan (negative class)
            # Find a Not_Trojan as positive and a Trojan as negative
            pos_idx = np.random.choice(np.where(y == 0)[0])
            neg_idx = np.random.choice(np.where(y == 1)[0])
            anchors.append(X.iloc[i].values)
            positives.append(X.iloc[pos_idx].values)
            negatives.append(X.iloc[neg_idx].values)

    return np.array(anchors), np.array(positives), np.array(negatives)

# ---------------------------
# Train the Model
# ---------------------------

def train_and_test(train_files, test_files, algorithm='triplet_loss_nn'):
    # Load the training and testing data
    df_train = load_design_files(train_files)
    df_test = load_design_files(test_files)

    # Encode the 'gate_type' (if needed for the triplet model)
    le_gate_type = LabelEncoder()
    le_gate_type.fit(df_train['gate_type'])
    df_train['gate_type'] = le_gate_type.transform(df_train['gate_type'])
    df_test['gate_type'] = le_gate_type.transform(df_test['gate_type'])

    # Map labels to binary format
    df_train['label'] = df_train['label'].map({'Not_Trojan': 0, 'Trojan': 1})
    df_test['label'] = df_test['label'].map({'Not_Trojan': 0, 'Trojan': 1})

    # Drop non-numeric or ID-based columns
    drop_cols = ['gate_name', 'output', 'inputs', 'gate_numbers', 'design_id']
    df_train.drop(columns=drop_cols, inplace=True, errors='ignore')
    df_test.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Ensure numeric columns and drop NaNs
    df_train = df_train.apply(pd.to_numeric, errors='coerce').dropna()
    df_test = df_test.apply(pd.to_numeric, errors='coerce').dropna()

    # Split features and target
    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    X_test = df_test.drop(columns=['label'])
    y_test = df_test['label']

    # Generate triplets (anchor, positive, negative)
    X_train_anchor, X_train_positive, X_train_negative = create_triplets(X_train, y_train)
    X_test_anchor, X_test_positive, X_test_negative = create_triplets(X_test, y_test)

    # Create a triplet model
    input_dim = X_train.shape[1]
    model = create_triplet_model(input_dim)

    # Compile the model with triplet loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Try different learning rates
    model.compile(optimizer=optimizer, loss=triplet_loss)   

    # Summary of the model architecture
    model.summary()

    # Train the model
    model.fit([X_train_anchor, X_train_positive, X_train_negative], 
              np.zeros(len(X_train_anchor)),  # Dummy y_true for triplet loss (not used)
              epochs=10, batch_size=2048)

    # Save the model
    model.save('triplet_loss_model.h5')
    print("Triplet Loss model saved as 'triplet_loss_model.h5'")

    # Evaluate the model on the test data
    # After training, use embeddings for classification
    embeddings_train = model.predict([X_train_anchor, X_train_positive, X_train_negative])
    embeddings_test = model.predict([X_test_anchor, X_test_positive, X_test_negative])

    # Check the shapes of the embeddings
    print(f"Train embeddings shape: {embeddings_train.shape}")
    print(f"Test embeddings shape: {embeddings_test.shape}")

    # Calculate Euclidean distance to the mean embedding for each class
    mean_train_embedding = np.mean(embeddings_train, axis=0)

    # Compute distances for all test samples
    train_distances = np.linalg.norm(embeddings_train - mean_train_embedding, axis=1)
    test_distances = np.linalg.norm(embeddings_test - mean_train_embedding, axis=1)

    # Thresholding the distances to classify Trojan/Not_Trojan
    threshold = np.percentile(train_distances, 50)
    y_pred = (test_distances > threshold).astype(int)

    # Ensure that y_pred has the same number of samples as y_test
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Now, print the classification report
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("Trojan F1-score:", f1_score(y_test, y_pred, pos_label=1))


    # Optionally save predictions
    df_test['predicted'] = y_pred
    df_test['predicted'] = df_test['predicted'].map({0: 'Not_Trojan', 1: 'Trojan'})
    df_test.to_csv("test_predictions_output.csv", index=False)

    return model

# ---------------------------
# Example Usage
# ---------------------------

if __name__ == "__main__":
    # Choose your files here
    train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24]
    test_id = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29]
    train_files = [f"training_data_w_label/design{i}_label.csv" for i in train_id]
    test_files = [f"training_data_w_label/design{i}_label.csv" for i in test_id]
    args = parse_args()

    # Train and evaluate the model
    train_and_test(train_files, test_files, args.algorithm)
