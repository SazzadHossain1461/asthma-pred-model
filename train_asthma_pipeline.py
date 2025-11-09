"""
train_asthma_pipeline.py
Train classical classifiers on a CSV dataset and save the best pipeline.

Usage:
    python train_asthma_pipeline.py --data /path/to/asthma_disease_data.csv --target HasAsthma --outdir ./artifacts
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

def build_preprocessor(X):
    """Build preprocessor for numerical and categorical features."""
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    print(f"🔢 Numerical features: {numeric_cols}")
    print(f"📝 Categorical features: {categorical_cols}")
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')
    
    return preprocessor

def main(args):
    """Main training function."""
    print("🫁 TRAINING ASTHMA PREDICTION MODEL")
    print("=" * 50)
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        print("📁 Creating sample data...")
        create_sample_data(args.data)
    
    # Load data
    print(f"📁 Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    
    if args.target not in df.columns:
        available_cols = df.columns.tolist()
        print(f"❌ Target column '{args.target}' not found in dataset.")
        print(f"📋 Available columns: {available_cols}")
        
        # Try to find a suitable target column
        possible_targets = [col for col in available_cols if 'diagnosis' in col.lower() or 'asthma' in col.lower() or 'target' in col.lower()]
        if possible_targets:
            args.target = possible_targets[0]
            print(f"🎯 Using detected target column: {args.target}")
        else:
            args.target = available_cols[-1]  # Use last column as fallback
            print(f"⚠️  Using fallback target column: {args.target}")
    
    print(f"🎯 Using target: {args.target}")
    target_col = args.target
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # Simple encode for categorical labels
    if y.dtype == object or str(y.dtype).startswith('category'):
        y = y.astype('category').cat.codes

    print("📊 Class balance:")
    print(y.value_counts())
    print(f"📈 Dataset shape: {df.shape}")

    # Build preprocessor
    preprocessor = build_preprocessor(X)

    # Train-test split
    stratify_arg = y if len(np.unique(y)) > 1 and y.nunique() < len(y) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    print(f"📚 Training set: {X_train.shape[0]} samples")
    print(f"🧪 Test set: {X_test.shape[0]} samples")

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver='liblinear', random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    results = {}
    pipelines = {}

    print("\n🔄 TRAINING MODELS")
    print("=" * 30)

    for name, clf in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
        
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps['clf'], "predict_proba") else None
            
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_proba) if (y_proba is not None and len(np.unique(y_test)) == 2) else None

            results[name] = {'accuracy': acc, 'roc_auc': roc, 'report': report}
            pipelines[name] = pipe

            print(f"✅ {name}: Accuracy = {acc:.4f}, ROC AUC = {roc if roc else 'N/A'}")
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            continue

    # Choose best model by roc_auc if available else accuracy
    best = None
    best_score = -1
    for name, r in results.items():
        score = r['roc_auc'] if r['roc_auc'] is not None else r['accuracy']
        if score > best_score:
            best_score = score
            best = name

    print(f"\n🏆 BEST MODEL: {best} (score={best_score:.4f})")

    # Save results
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if best in pipelines:
        model_path = out_dir / "best_asthma_model.pkl"
        joblib.dump(pipelines[best], model_path)
        
        # Save results summary
        results_path = out_dir / "training_results.csv"
        results_df = pd.DataFrame(results).T
        results_df.to_csv(results_path)
        
        # Save README
        readme_path = out_dir / "README_model.txt"
        with open(readme_path, "w") as f:
            f.write(f"Best model: {best}\n")
            f.write(f"Best score: {best_score:.4f}\n")
            f.write(f"Target: {target_col}\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Results path: {results_path}\n")
            f.write(f"Training date: {pd.Timestamp.now()}\n")
        
        print(f"💾 Saved model to: {model_path}")
        print(f"💾 Saved results to: {results_path}")
        print(f"💾 Saved README to: {readme_path}")
        
        # Print detailed results
        print(f"\n📈 DETAILED RESULTS")
        print("=" * 30)
        for name, r in results.items():
            print(f"{name}:")
            print(f"  Accuracy: {r['accuracy']:.4f}")
            print(f"  ROC AUC: {r['roc_auc'] if r['roc_auc'] else 'N/A'}")
    else:
        print("❌ No successful models to save")

def create_sample_data(data_path):
    """Create sample asthma dataset if it doesn't exist."""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Create directory if needed
    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("📝 Creating sample asthma dataset...")
    np.random.seed(42)
    n_samples = 500

    data = {
        'Age': np.random.randint(18, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'BMI': np.random.uniform(18, 35, n_samples),
        'Smoking_Status': np.random.choice(['Never', 'Former', 'Current'], n_samples),
        'FEV1': np.random.uniform(50, 120, n_samples),
        'Eosinophil_Count': np.random.uniform(0, 1, n_samples),
        'Allergy_History': np.random.choice([0, 1], n_samples),
        'Family_History': np.random.choice([0, 1], n_samples),
        'Wheezing': np.random.choice([0, 1], n_samples),
        'Cough': np.random.choice([0, 1], n_samples)
    }

    # Create target variable with some relationship to features
    data['Diagnosis'] = (
        (data['FEV1'] < 70).astype(int) |
        (data['Eosinophil_Count'] > 0.5).astype(int) |
        (data['Wheezing'] == 1).astype(int) |
        (data['Family_History'] == 1).astype(int)
    )

    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)
    print(f"✅ Sample dataset created: {data_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Target distribution:\n{df['Diagnosis'].value_counts()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train asthma prediction models')
    parser.add_argument("--data", required=True, help="Path to asthma data CSV file")
    parser.add_argument("--target", required=True, help="Name of the target column to predict")
    parser.add_argument("--outdir", default="./artifacts", help="Output directory for models and results")
    
    args = parser.parse_args()
    main(args)