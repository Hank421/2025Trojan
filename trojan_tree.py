import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import os
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import f1_score
import joblib
import argparse

# ---------------------------
# Function: Load, Train, Test
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and test a model on Trojan detection data.")
    parser.add_argument('--algorithm', type=str, default='lightgbm',
                        help="Choose the algorithm to use for training")
    return parser.parse_args()

def train_and_test(train_files, test_files, algorithm='lightgbm'):
    def load_design_files(filenames):
        dfs = []
        for file in filenames:
            df = pd.read_csv(file)
            df['design_id'] = os.path.basename(file)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    # Load data
    df_train = load_design_files(train_files)
    df_test = load_design_files(test_files)

    # Encode 'gate_type'
    le_gate_type = LabelEncoder()
    le_gate_type.fit(df_train['gate_type'])

    df_train['gate_type'] = le_gate_type.transform(df_train['gate_type'])
    df_test['gate_type'] = le_gate_type.transform(df_test['gate_type'])

    # Map labels
    df_train['label'] = df_train['label'].map({'Not_Trojan': 0, 'Trojan': 1})
    df_test['label'] = df_test['label'].map({'Not_Trojan': 0, 'Trojan': 1})

    # Drop non-numeric or ID-based columns
    drop_cols = ['gate_name', 'output', 'inputs', 'gate_numbers', 'design_id']
    df_train.drop(columns=drop_cols, inplace=True, errors='ignore')
    df_test.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Ensure numeric and drop NaNs
    df_train = df_train.apply(pd.to_numeric, errors='coerce').dropna()
    df_test = df_test.apply(pd.to_numeric, errors='coerce').dropna()

    # Split features and target
    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    X_test = df_test.drop(columns=['label'])
    y_test = df_test['label']

    # Train model
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    if algorithm == 'catboost': # 0.69 
        model = CatBoostClassifier(
            scale_pos_weight=scale_pos_weight,  # If your dataset is imbalanced
            random_state=42,
            verbose=0  # Suppress output during training
        )
    elif algorithm == 'lightgbm': # 0.81 
        model = lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,  # If your dataset is imbalanced
            random_state=42
        )
    elif algorithm == 'random_forest': # 0.48
        model = RandomForestClassifier(
            class_weight={0: 1, 1: scale_pos_weight},  # If your dataset is imbalanced
            random_state=42
        )
    elif algorithm == 'gradient_boosting': # 0.61
        model = GradientBoostingClassifier(
            random_state=42
        )
    elif algorithm == 'ada_boost': # 0.36 
        model = AdaBoostClassifier(
            random_state=42
        )
    elif algorithm == 'xgboost': # 0.80
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
    elif algorithm == 'ensemble':
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        model_lightgbm = lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        model.fit(X_train, y_train)
        model_lightgbm.fit(X_train, y_train)
        y_pred = (model.predict_proba(X_train)[:, 1] + model_lightgbm.predict_proba(X_train)[:, 1]) / 2
        y_pred = (y_pred > 0.5).astype(int)  # Thresholding at 0.5 for Trojan detection
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from 'catboost', 'lightgbm', 'random_forest', 'gradient_boosting', or 'ada_boost'.")
    
    if algorithm != 'xgboost_catboost':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)

    # Save the model to a file
    joblib.dump(model, f'xgb_model.pkl')
    print(f"\nModel saved as {algorithm}_model.pkl\n")
    # Save the fitted LabelEncoder
    joblib.dump(le_gate_type, 'le_gate_type.pkl')
    print("\nLabelEncoder saved as le_gate_type.pkl\n")

    print("=== Training Classification Report ===")
    print(classification_report(y_train, y_pred))


    # Predict and evaluate
    if algorithm != 'xgboost_catboost':
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = (model.predict_proba(X_test)[:, 1] + model_lightgbm.predict_proba(X_test)[:, 1]) / 2
    y_pred = (y_probs > 0.5).astype(int)  # Thresholding at 0.5 for Trojan detection

    # Now check F1
    print("Trojan F1-score:", f1_score(y_test, y_pred, pos_label=1))
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # Optionally save predictions
    df_test['predicted'] = y_pred
    df_test['predicted'] = df_test['predicted'].map({0: 'Not_Trojan', 1: 'Trojan'})
    df_test.to_csv("test_predictions_output.csv", index=False)
    
        # plot_importance(model, importance_type='weight', max_num_features=50)
        # plt.grid(False)
        # plt.savefig("feature_importance.png", bbox_inches='tight')
        
        # # List all feature names and their importances
        # importances = model.feature_importances_
        # feature_names = model.feature_names_in_

        # # Display feature names and their corresponding importances
        # print("\n=== Feature Importances ===")
        # for feature, importance in zip(feature_names, importances):
        #     print(f"Feature: {feature}, Importance: {importance}")

    return model  # if you want to reuse the model

# ---------------------------
# Example Usage
# ---------------------------

if __name__ == "__main__":
    # Choose your files here
    train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 24, 28, 29]
    test_id = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 25, 26, 27]
    train_files = [f"training_data_w_label/design{i}_label.csv" for i in train_id]
    test_files = [f"training_data_w_label/design{i}_label.csv" for i in test_id]
    args = parse_args()


    train_and_test(train_files, test_files, args.algorithm)
