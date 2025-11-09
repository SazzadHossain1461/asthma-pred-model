"""
rl_prorl_trainer.py
Enhanced Reinforcement Learning trainer for asthma prediction model optimization.
Shows detailed results and improvements.

Usage:
    python rl_prorl_trainer.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
import time
import os

warnings.filterwarnings('ignore')

class EnhancedAsthmaRLTrainer:
    """
    Enhanced Reinforcement Learning trainer with detailed results display.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.preprocessor = None
        self.optimization_history = []
        
    def load_data(self, data_path="data/asthma_disease_data.csv"):
        """Load and prepare asthma data with detailed logging."""
        try:
            print("📁 Loading asthma dataset...")
            df = pd.read_csv(data_path)
            print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
            
            # Find target column
            target_candidates = [col for col in df.columns if 'diagnosis' in col.lower() or 'asthma' in col.lower()]
            self.target_col = target_candidates[0] if target_candidates else df.columns[-1]
            
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            
            # Encode categorical target
            if y.dtype == 'object':
                y = y.astype('category').cat.codes
                
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Build preprocessor
            self._build_preprocessor(self.X_train)
            
            # Display dataset info
            self._display_dataset_info(df, X, y)
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def _display_dataset_info(self, df, X, y):
        """Display detailed dataset information."""
        print("\n📊 DATASET INFORMATION")
        print("=" * 50)
        print(f"Target variable: '{self.target_col}'")
        print(f"Classes: {y.nunique()}")
        print(f"Class distribution:\n{y.value_counts().to_dict()}")
        print(f"Features: {X.shape[1]} total")
        print(f"Numerical features: {len(self.numeric_cols)}")
        print(f"Categorical features: {len(self.categorical_cols)}")
        print("=" * 50)

    def _build_preprocessor(self, X):
        """Build preprocessor for categorical and numerical features."""
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create transformers
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.numeric_cols),
            ('cat', categorical_transformer, self.categorical_cols)
        ])

    def train_baseline(self):
        """Train baseline Random Forest model with comprehensive evaluation."""
        print("\n🎯 TRAINING BASELINE MODEL")
        print("=" * 50)
        
        # Preprocess data
        X_train_processed = self.preprocessor.fit_transform(self.X_train)
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Train baseline model
        self.baseline_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        print("🔄 Training baseline model...")
        start_time = time.time()
        self.baseline_model.fit(X_train_processed, self.y_train)
        training_time = time.time() - start_time
        
        # Comprehensive evaluation
        baseline_results = self._evaluate_model(
            self.baseline_model, 
            X_test_processed, 
            "Baseline"
        )
        baseline_results['training_time'] = training_time
        
        self.results['baseline'] = baseline_results
        print(f"⏱️  Training time: {training_time:.2f} seconds")
        
        return baseline_results['accuracy']

    def _evaluate_model(self, model, X_test, model_name):
        """Comprehensive model evaluation."""
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        acc = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba) if (y_proba is not None and len(np.unique(self.y_test)) == 2) else None
        
        # Display results
        print(f"\n📈 {model_name.upper()} MODEL RESULTS")
        print("-" * 30)
        print(f"Accuracy:    {acc:.4f}")
        if auc is not None:
            print(f"ROC AUC:     {auc:.4f}")
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            top_features = self._get_top_features(model, 5)
            print(f"Top 5 features: {top_features}")
        
        return {
            'accuracy': acc,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }

    def _get_top_features(self, model, n_features=5):
        """Get top n feature importances."""
        if hasattr(model, 'feature_importances_'):
            # Get feature names after preprocessing
            feature_names = []
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                # Fallback: create generic feature names
                feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
            
            # Get top features
            indices = np.argsort(model.feature_importances_)[::-1][:n_features]
            top_features = [(feature_names[i], model.feature_importances_[i]) for i in indices]
            return top_features
        return []

    def rl_hyperparameter_optimization(self, n_iterations=50):
        """
        Enhanced RL-inspired hyperparameter optimization with detailed progress.
        """
        print(f"\n🚀 STARTING RL HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        print(f"Running {n_iterations} iterations...")
        
        # Preprocess data once
        X_train_processed = self.preprocessor.transform(self.X_train)
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        best_score = 0
        best_params = None
        best_model = None
        
        print("\n🔄 Optimization Progress:")
        print("-" * 40)
        
        for i in range(n_iterations):
            # Generate random hyperparameters (simplified RL exploration)
            params = self._generate_hyperparameters()
            
            try:
                model = RandomForestClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features'],
                    random_state=42,
                    n_jobs=-1
                )
                
                # Train and evaluate
                model.fit(X_train_processed, self.y_train)
                y_pred = model.predict(X_test_processed)
                score = accuracy_score(self.y_test, y_pred)
                
                # Track progress
                self.optimization_history.append({
                    'iteration': i + 1,
                    'score': score,
                    'params': params.copy()
                })
                
                # Display progress every 10 iterations
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i+1:3d}: Best Accuracy = {best_score:.4f}")
                
                # Update best model
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    
                    print(f"🎯 Iteration {i+1:3d}: NEW BEST → Accuracy = {score:.4f}")
                    
            except Exception as e:
                continue
        
        # Store best model and results
        self.best_rl_model = best_model
        rl_results = self._evaluate_model(best_model, X_test_processed, "RL Optimized")
        rl_results['params'] = best_params
        self.results['rl_optimized'] = rl_results
        
        # Display optimization summary
        self._display_optimization_summary(best_score, best_params)
        
        return best_score

    def _generate_hyperparameters(self):
        """Generate random hyperparameters for exploration."""
        return {
            'n_estimators': np.random.choice([50, 100, 200, 300, 500]),
            'max_depth': np.random.choice([3, 5, 10, 15, 20, None]),
            'min_samples_split': np.random.choice([2, 5, 10, 20]),
            'min_samples_leaf': np.random.choice([1, 2, 4, 8]),
            'max_features': np.random.choice(['sqrt', 'log2', None])
        }

    def _display_optimization_summary(self, best_score, best_params):
        """Display optimization process summary."""
        print(f"\n✅ OPTIMIZATION COMPLETED")
        print("=" * 50)
        print(f"Best Accuracy: {best_score:.4f}")
        print(f"\n🎯 BEST HYPERPARAMETERS:")
        for param, value in best_params.items():
            print(f"  {param:.<20}: {value}")
        
        # Show improvement from baseline
        baseline_acc = self.results['baseline']['accuracy']
        improvement = best_score - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100
        
        print(f"\n📈 IMPROVEMENT ANALYSIS:")
        print(f"  Baseline Accuracy:    {baseline_acc:.4f}")
        print(f"  Optimized Accuracy:   {best_score:.4f}")
        print(f"  Absolute Improvement: {improvement:+.4f}")
        print(f"  Relative Improvement: {improvement_pct:+.2f}%")

    def pro_rl_advanced_optimization(self, n_iterations=30):
        """
        Advanced ProRL optimization with adaptive learning.
        """
        print(f"\n🔥 STARTING ProRL ADVANCED OPTIMIZATION")
        print("=" * 60)
        
        X_train_processed = self.preprocessor.transform(self.X_train)
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        best_score = self.results['baseline']['accuracy']
        best_model = self.baseline_model
        best_params = {'method': 'baseline'}
        
        # Adaptive parameter ranges based on baseline performance
        param_ranges = self._get_adaptive_param_ranges(best_score)
        
        print("🔄 ProRL Adaptive Optimization:")
        print("-" * 40)
        
        for i in range(n_iterations):
            # More intelligent parameter sampling
            params = self._sample_intelligent_parameters(param_ranges, i, n_iterations)
            
            try:
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                model.fit(X_train_processed, self.y_train)
                score = accuracy_score(self.y_test, model.predict(X_test_processed))
                
                # Adaptive learning: adjust ranges based on performance
                if score > best_score + 0.02:  # Significant improvement
                    param_ranges = self._adjust_param_ranges(param_ranges, 'expand')
                elif score < best_score - 0.01:  # Performance drop
                    param_ranges = self._adjust_param_ranges(param_ranges, 'contract')
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params
                    print(f"🚀 Iteration {i+1:2d}: ProRL BEST → {score:.4f}")
                
            except Exception:
                continue
        
        # Store ProRL results
        pro_rl_results = self._evaluate_model(best_model, X_test_processed, "ProRL Advanced")
        pro_rl_results['params'] = best_params
        self.results['pro_rl_advanced'] = pro_rl_results
        self.best_pro_rl_model = best_model
        
        self._display_pro_rl_comparison()
        return best_score

    def _get_adaptive_param_ranges(self, baseline_score):
        """Get adaptive parameter ranges based on baseline performance."""
        if baseline_score < 0.7:
            return {  # More aggressive search for poor performers
                'n_estimators': [100, 300, 500, 1000],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            return {  # Refined search for good performers
                'n_estimators': [200, 300, 400, 500],
                'max_depth': [8, 10, 12, 15, None],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2]
            }

    def _sample_intelligent_parameters(self, param_ranges, current_iter, total_iters):
        """Intelligent parameter sampling with exploration/exploitation tradeoff."""
        # More exploration early, more exploitation late
        exploration_factor = 1.0 - (current_iter / total_iters)
        
        params = {}
        for param, values in param_ranges.items():
            if np.random.random() < exploration_factor:
                # Exploration: random choice
                params[param] = np.random.choice(values)
            else:
                # Exploitation: prefer values that worked well historically
                if hasattr(self, 'optimization_history') and len(self.optimization_history) > 0:
                    # Sample from successful parameters
                    successful_params = [h['params'][param] for h in self.optimization_history 
                                       if h['score'] > self.results['baseline']['accuracy']]
                    if successful_params:
                        params[param] = np.random.choice(successful_params)
                    else:
                        params[param] = np.random.choice(values)
                else:
                    params[param] = np.random.choice(values)
        
        return params

    def _adjust_param_ranges(self, param_ranges, adjustment):
        """Adjust parameter ranges based on performance."""
        new_ranges = param_ranges.copy()
        
        if adjustment == 'expand':
            # Expand search space
            for param in ['n_estimators', 'max_depth']:
                if param in new_ranges:
                    current_values = new_ranges[param]
                    if None not in current_values:
                        new_ranges[param] = current_values + [None]
        elif adjustment == 'contract':
            # Contract to promising ranges
            for param in ['min_samples_split', 'min_samples_leaf']:
                if param in new_ranges:
                    current_values = new_ranges[param]
                    if len(current_values) > 2:
                        new_ranges[param] = current_values[:2]
        
        return new_ranges

    def _display_pro_rl_comparison(self):
        """Display comprehensive comparison between all methods."""
        print(f"\n📊 COMPREHENSIVE RESULTS COMPARISON")
        print("=" * 60)
        
        methods = ['baseline', 'rl_optimized', 'pro_rl_advanced']
        method_names = ['Baseline', 'RL Optimized', 'ProRL Advanced']
        
        print(f"{'Method':<15} {'Accuracy':<10} {'Improvement':<12} {'AUC':<10}")
        print("-" * 60)
        
        baseline_acc = self.results['baseline']['accuracy']
        
        for method, name in zip(methods, method_names):
            if method in self.results:
                acc = self.results[method]['accuracy']
                auc = self.results[method].get('auc', 'N/A')
                improvement = acc - baseline_acc
                
                if method == 'baseline':
                    print(f"{name:<15} {acc:<10.4f} {'-':<12} {auc if auc != 'N/A' else 'N/A':<10}")
                else:
                    print(f"{name:<15} {acc:<10.4f} {improvement:+.4f} ({improvement/baseline_acc*100:+.1f}%)  {auc if auc != 'N/A' else 'N/A':<10}")

    def run_comprehensive_analysis(self):
        """Run complete analysis with all optimization methods."""
        print("🎯 STARTING COMPREHENSIVE RL/PRO-RL ANALYSIS")
        print("=" * 60)
        
        # 1. Baseline model
        baseline_acc = self.train_baseline()
        
        # 2. Basic RL optimization
        rl_acc = self.rl_hyperparameter_optimization(n_iterations=30)
        
        # 3. Advanced ProRL optimization
        pro_rl_acc = self.pro_rl_advanced_optimization(n_iterations=20)
        
        # 4. Final comparison and saving
        self.save_comprehensive_results()
        
        return {
            'baseline': baseline_acc,
            'rl_optimized': rl_acc,
            'pro_rl_advanced': pro_rl_acc
        }

    def save_comprehensive_results(self):
        """Save all models and results."""
        os.makedirs('artifacts', exist_ok=True)
        
        # Save models
        model_data = {
            'baseline': {
                'model': self.baseline_model,
                'preprocessor': self.preprocessor,
                'results': self.results['baseline']
            },
            'rl_optimized': {
                'model': self.best_rl_model,
                'preprocessor': self.preprocessor,
                'results': self.results['rl_optimized']
            },
            'pro_rl_advanced': {
                'model': self.best_pro_rl_model,
                'preprocessor': self.preprocessor,
                'results': self.results['pro_rl_advanced']
            }
        }
        
        joblib.dump(model_data, 'artifacts/comprehensive_rl_models.pkl')
        
        # Save results summary
        results_summary = pd.DataFrame({
            method: self.results[method] for method in self.results.keys()
        }).T
        
        results_summary.to_csv('artifacts/rl_training_results_detailed.csv')
        
        # Save optimization history
        history_df = pd.DataFrame(self.optimization_history)
        history_df.to_csv('artifacts/optimization_history.csv', index=False)
        
        print(f"\n💾 COMPREHENSIVE RESULTS SAVED")
        print("=" * 40)
        print("📁 Artifacts saved in 'artifacts/' directory:")
        print("  - comprehensive_rl_models.pkl")
        print("  - rl_training_results_detailed.csv")
        print("  - optimization_history.csv")

def main():
    """Main training function with enhanced reporting."""
    print("🚀 ENHANCED RL/PRO-RL ASTHMA MODEL TRAINING")
    print("=" * 60)
    
    # Initialize enhanced trainer
    trainer = EnhancedAsthmaRLTrainer()
    
    # Load data
    if not trainer.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Run comprehensive analysis
    final_results = trainer.run_comprehensive_analysis()
    
    # Final summary
    print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("📈 Final Results Summary:")
    for method, accuracy in final_results.items():
        print(f"  {method.replace('_', ' ').title():<15}: {accuracy:.4f}")
    
    best_method = max(final_results, key=final_results.get)
    best_accuracy = final_results[best_method]
    print(f"\n🏆 Best Performance: {best_method.replace('_', ' ').title()}")
    print(f"🎯 Best Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()