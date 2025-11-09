import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import random
import time

# --- Complete SmartBiomaterialDesignSystem class ---
class SmartBiomaterialDesignSystem:
    """
    AI-driven biomaterial design system for predicting cellular responses.
    """
    
    def __init__(self, stimulus_type="photo"):
        self.stimulus_type = stimulus_type
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'crosslink_density', 'stiffness', 'porosity', 'molecular_weight',
            'hydrophobicity', 'charge_density', 'degradation_rate', 'surface_roughness'
        ]
        self.training_history = []
    
    def generate_sample(self):
        """Generate a random biomaterial sample."""
        return {
            'crosslink_density': np.random.uniform(0.1, 0.9),
            'stiffness': np.random.uniform(5, 100),
            'porosity': np.random.uniform(0.1, 0.8),
            'molecular_weight': np.random.uniform(10000, 200000),
            'hydrophobicity': np.random.uniform(0.1, 0.9),
            'charge_density': np.random.uniform(0.1, 0.9),
            'degradation_rate': np.random.uniform(0.1, 0.5),
            'surface_roughness': np.random.uniform(10, 200)
        }
    
    def predict_cell_response(self, material_params):
        """Predict cellular response to biomaterial."""
        if not self.is_trained:
            # Return educated mock predictions based on material properties
            porosity = material_params['porosity']
            stiffness = material_params['stiffness']
            charge = material_params['charge_density']
            
            proliferation = 0.6 + 0.3 * (porosity - 0.5)  # Higher porosity -> better proliferation
            differentiation = 0.5 + 0.2 * (stiffness - 50) / 50  # Medium stiffness optimal
            adhesion = 0.7 + 0.3 * (charge - 0.5)  # Moderate charge good for adhesion
            overall = (proliferation + differentiation + adhesion) / 3
            
            return {
                "proliferation_score": max(0.1, min(0.99, proliferation)),
                "differentiation_score": max(0.1, min(0.99, differentiation)),
                "adhesion_score": max(0.1, min(0.99, adhesion)),
                "overall_biocompatibility": max(0.1, min(0.99, overall))
            }
        
        # Convert material parameters to feature array
        features = np.array([material_params[feat] for feat in self.feature_names]).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Predict using trained model
        if hasattr(self.model, 'predict'):
            overall_biocompatibility = self.model.predict(features_scaled)[0]
            overall_biocompatibility = max(0.1, min(0.99, overall_biocompatibility))
        else:
            overall_biocompatibility = np.random.uniform(0.5, 0.9)
        
        # Calculate related scores based on material properties (more realistic)
        porosity = material_params['porosity']
        stiffness = material_params['stiffness']
        charge = material_params['charge_density']
        hydrophobicity = material_params['hydrophobicity']
        
        proliferation = 0.6 + 0.3 * (porosity - 0.5) - 0.1 * (hydrophobicity - 0.5)
        differentiation = 0.5 + 0.2 * (stiffness - 50) / 50 - 0.1 * abs(charge - 0.4)
        adhesion = 0.7 + 0.3 * (charge - 0.5) - 0.2 * (hydrophobicity - 0.3)
        
        return {
            "proliferation_score": max(0.1, min(0.99, proliferation)),
            "differentiation_score": max(0.1, min(0.99, differentiation)),
            "adhesion_score": max(0.1, min(0.99, adhesion)),
            "overall_biocompatibility": overall_biocompatibility
        }
    
    def train_on_database(self, materials_list, responses):
        """Train on historical biomaterial data."""
        if len(materials_list) == 0:
            print("No training data provided. Using mock predictions.")
            return
        
        # Convert to feature matrix
        X = np.array([[m[feat] for feat in self.feature_names] for m in materials_list])
        y = np.array(responses)
        
        print(f"🔄 Training biomaterial model with {len(materials_list)} samples")
        print(f"📊 Response range: {y.min():.3f} to {y.max():.3f}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Always use regressor for continuous responses
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training score
        train_score = self.model.score(X_scaled, y)
        print(f"✅ Biomaterial model trained. R² score: {train_score:.4f}")
        
        # Store training history
        self.training_history.append({
            'samples': len(materials_list),
            'r2_score': train_score,
            'response_range': (y.min(), y.max())
        })
    
    def optimize_composition(self, target_response, n_iterations=100):
        """Optimize biomaterial composition for target cellular response."""
        print(f"🎯 Optimizing for target biocompatibility: {target_response:.3f}")
        
        best_material = None
        best_score = -np.inf
        best_response = None
        
        for i in range(n_iterations):
            candidate = self.generate_sample()
            response = self.predict_cell_response(candidate)
            
            # Score based on proximity to target and overall quality
            proximity_score = -abs(response['overall_biocompatibility'] - target_response)
            quality_bonus = response['overall_biocompatibility'] * 0.1  # Prefer higher overall scores
            score = proximity_score + quality_bonus
            
            if score > best_score:
                best_score = score
                best_material = candidate
                best_response = response
        
        print(f"✅ Best material found with biocompatibility: {best_response['overall_biocompatibility']:.3f}")
        return best_material, best_response

    def display_optimization_results(self, material, response):
        """Display detailed results of material optimization."""
        print("\n" + "="*50)
        print("🧬 OPTIMIZED BIOMATERIAL COMPOSITION")
        print("="*50)
        for key, value in material.items():
            print(f"{key:.<20}: {value:>8.3f}")
        
        print("\n📊 CELLULAR RESPONSE PREDICTIONS")
        print("-" * 30)
        for key, value in response.items():
            print(f"{key:.<25}: {value:>8.3f}")
        print("="*50)

# --- Enhanced AsthmaPredictionRLModel class ---
class AsthmaPredictionRLModel:
    """
    Reinforcement Learning–assisted Asthma Disease Prediction System.
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.preprocessor = None
        self.is_trained = False
        self.training_history = []

    def load_data(self, csv_path: str):
        """Load asthma dataset."""
        print(f"📁 Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        possible_targets = [c for c in df.columns if 'asthma' in c.lower() or 'label' in c.lower() or 'diagnosis' in c.lower()]
        self.target_col = possible_targets[0] if possible_targets else df.columns[-1]
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # Encode categorical target
        if y.dtype == object:
            y = y.astype('category').cat.codes
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build preprocessor for categorical and numerical features
        self._build_preprocessor(self.X_train)
        
        print(f"✅ Loaded data with {X.shape[1]} features and target '{self.target_col}'")
        print(f"📊 Class distribution: {dict(y.value_counts())}")

    def _build_preprocessor(self, X):
        """Build preprocessor for categorical and numerical features."""
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"🔢 Numerical columns: {self.numeric_cols}")
        print(f"📝 Categorical columns: {self.categorical_cols}")
        
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

    def train(self):
        """Train baseline model with preprocessing."""
        print("🔄 Training baseline asthma model...")
        # Preprocess the data
        X_train_processed = self.preprocessor.fit_transform(self.X_train)
        
        # Train model
        self.model.fit(X_train_processed, self.y_train)
        self.is_trained = True
        print("✅ Baseline model trained.")

    def evaluate(self, detailed=False):
        """Evaluate accuracy with preprocessing."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        
        X_test_processed = self.preprocessor.transform(self.X_test)
        preds = self.model.predict(X_test_processed)
        probas = self.model.predict_proba(X_test_processed)[:, 1]
        
        acc = accuracy_score(self.y_test, preds)
        
        if detailed:
            print("\n" + "="*40)
            print("📈 MODEL EVALUATION RESULTS")
            print("="*40)
            print(f"Accuracy: {acc:.4f}")
            
            if len(np.unique(self.y_test)) == 2:
                auc = roc_auc_score(self.y_test, probas)
                print(f"ROC AUC: {auc:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(self.y_test, preds))
            print("="*40)
        else:
            print(f"📊 Model Accuracy: {acc:.4f}")
        
        return acc

    def reinforce_with_biomaterial(self, biomaterial_system, n_iterations=5):
        """
        Enhanced reinforcement with biomaterial data.
        """
        print("\n🚀 STARTING BIOMATERIAL REINFORCEMENT LEARNING")
        print("="*60)
        
        # Store original performance
        original_acc = self.evaluate()
        self.training_history.append({'iteration': 0, 'accuracy': original_acc, 'type': 'baseline'})
        
        print(f"🎯 Baseline Accuracy: {original_acc:.4f}")
        print(f"🔄 Running {n_iterations} reinforcement iterations...")
        
        best_accuracy = original_acc
        
        for i in range(n_iterations):
            try:
                print(f"\n--- Reinforcement Iteration {i+1}/{n_iterations} ---")
                
                # Generate optimized biomaterial
                target_biocompatibility = 0.7 + (i * 0.05)  # Gradually increase target
                best_material, best_response = biomaterial_system.optimize_composition(
                    target_response=target_biocompatibility
                )
                
                # Display biomaterial results
                biomaterial_system.display_optimization_results(best_material, best_response)
                
                # Create synthetic patient features based on biomaterial response
                synthetic_features = self._create_synthetic_features(best_response)
                
                # Create synthetic label (asthma diagnosis probability based on biomaterial response)
                synthetic_label = self._generate_synthetic_label(best_response)
                
                print(f"🧪 Synthetic sample - Label: {synthetic_label}")
                
                # Enhanced retraining with synthetic data
                current_acc = self._retrain_with_synthetic(synthetic_features, synthetic_label)
                
                # Track progress
                self.training_history.append({
                    'iteration': i+1,
                    'accuracy': current_acc,
                    'type': 'reinforced',
                    'biocompatibility': best_response['overall_biocompatibility'],
                    'synthetic_label': synthetic_label
                })
                
                if current_acc > best_accuracy:
                    best_accuracy = current_acc
                    print(f"🎉 NEW BEST ACCURACY: {current_acc:.4f}")
                
                # Small delay to see progress
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Iteration {i+1} failed: {e}")
                continue

        print("\n" + "="*60)
        print("✅ REINFORCEMENT LEARNING COMPLETED")
        print("="*60)
        
        # Final evaluation
        final_acc = self.evaluate(detailed=True)
        
        # Display training history
        self._display_training_history()
        
        improvement = final_acc - original_acc
        print(f"\n📈 OVERALL IMPROVEMENT: {improvement:+.4f}")
        
        return final_acc

    def _create_synthetic_features(self, biomaterial_response):
        """Create synthetic patient features based on biomaterial response."""
        # Use biomaterial response scores as synthetic physiological markers
        base_features = np.array([
            biomaterial_response["proliferation_score"] * 100,  # Simulate cell count
            biomaterial_response["differentiation_score"] * 50,  # Simulate biomarker level
            biomaterial_response["adhesion_score"] * 80,        # Simulate inflammation marker
            biomaterial_response["overall_biocompatibility"] * 120,  # Simulate lung function
        ])
        
        # Add some noise and variation
        noise = np.random.normal(0, 5, base_features.shape)
        synthetic_features = base_features + noise
        
        return synthetic_features

    def _generate_synthetic_label(self, biomaterial_response):
        """Generate synthetic asthma label based on biomaterial response."""
        # Lower biocompatibility might correlate with higher asthma risk in this simulation
        asthma_probability = 1.0 - biomaterial_response["overall_biocompatibility"]
        
        # Add some randomness but bias based on biocompatibility
        if asthma_probability > 0.6:
            return 1  # Asthma diagnosis
        elif asthma_probability < 0.3:
            return 0  # No asthma
        else:
            return np.random.choice([0, 1], p=[1-asthma_probability, asthma_probability])

    def _retrain_with_synthetic(self, synthetic_features, synthetic_label):
        """Retrain model with synthetic data."""
        # Preprocess original training data
        X_train_processed = self.preprocessor.transform(self.X_train)
        
        # Ensure synthetic features match dimensions
        if synthetic_features.shape[0] < X_train_processed.shape[1]:
            # Pad synthetic features
            padding = np.zeros(X_train_processed.shape[1] - synthetic_features.shape[0])
            synthetic_features = np.concatenate([synthetic_features, padding])
        elif synthetic_features.shape[0] > X_train_processed.shape[1]:
            # Truncate synthetic features
            synthetic_features = synthetic_features[:X_train_processed.shape[1]]
        
        synthetic_features = synthetic_features.reshape(1, -1)
        
        # Combine original and synthetic data
        X_combined = np.vstack([X_train_processed, synthetic_features])
        y_combined = np.append(self.y_train, synthetic_label)
        
        # Retrain model
        self.model.fit(X_combined, y_combined)
        
        # Evaluate current performance
        X_test_processed = self.preprocessor.transform(self.X_test)
        current_preds = self.model.predict(X_test_processed)
        current_acc = accuracy_score(self.y_test, current_preds)
        
        print(f"📊 Current Accuracy: {current_acc:.4f}")
        return current_acc

    def _display_training_history(self):
        """Display reinforcement learning progress."""
        print("\n📋 REINFORCEMENT TRAINING HISTORY")
        print("-" * 50)
        for history in self.training_history:
            if history['type'] == 'baseline':
                print(f"Initial  : Accuracy = {history['accuracy']:.4f}")
            else:
                print(f"Iteration {history['iteration']:2d}: Accuracy = {history['accuracy']:.4f}, "
                      f"BioComp = {history.get('biocompatibility', 0):.3f}")


# --- Enhanced Pipeline Integration ---
if __name__ == "__main__":
    print("🚀 STARTING ENHANCED BIOMATERIAL-ASTHMA PIPELINE")
    print("="*60)
    
    # Initialize biomaterial system
    biomaterial_ai = SmartBiomaterialDesignSystem(stimulus_type="photo")
    
    # Create comprehensive training data for biomaterial AI
    print("\n🧪 INITIALIZING BIOMATERIAL AI TRAINING")
    training_materials = [
        # Low biocompatibility materials
        {'crosslink_density': 0.8, 'stiffness': 90, 'porosity': 0.2, 
         'molecular_weight': 180000, 'hydrophobicity': 0.8, 'charge_density': 0.1,
         'degradation_rate': 0.4, 'surface_roughness': 180},
        {'crosslink_density': 0.9, 'stiffness': 95, 'porosity': 0.1,
         'molecular_weight': 190000, 'hydrophobicity': 0.9, 'charge_density': 0.9,
         'degradation_rate': 0.45, 'surface_roughness': 190},
         
        # Medium biocompatibility materials  
        {'crosslink_density': 0.5, 'stiffness': 50, 'porosity': 0.5,
         'molecular_weight': 100000, 'hydrophobicity': 0.5, 'charge_density': 0.5,
         'degradation_rate': 0.25, 'surface_roughness': 100},
        {'crosslink_density': 0.6, 'stiffness': 60, 'porosity': 0.4,
         'molecular_weight': 120000, 'hydrophobicity': 0.4, 'charge_density': 0.4,
         'degradation_rate': 0.3, 'surface_roughness': 120},
         
        # High biocompatibility materials
        {'crosslink_density': 0.3, 'stiffness': 20, 'porosity': 0.7, 
         'molecular_weight': 50000, 'hydrophobicity': 0.3, 'charge_density': 0.3,
         'degradation_rate': 0.15, 'surface_roughness': 50},
        {'crosslink_density': 0.2, 'stiffness': 15, 'porosity': 0.8,
         'molecular_weight': 30000, 'hydrophobicity': 0.2, 'charge_density': 0.6,
         'degradation_rate': 0.1, 'surface_roughness': 30},
    ]
    training_responses = [0.3, 0.2, 0.6, 0.5, 0.8, 0.9]  # Biocompatibility scores
    
    biomaterial_ai.train_on_database(training_materials, training_responses)

    # Initialize and train asthma model
    print("\n🫁 INITIALIZING ASTHMA PREDICTION MODEL")
    asthma_model = AsthmaPredictionRLModel()
    
    try:
        asthma_model.load_data("data/asthma_disease_data.csv")
        asthma_model.train()
        
        print("\n=== BASELINE PERFORMANCE ===")
        baseline_acc = asthma_model.evaluate(detailed=True)

        print("\n=== BIOMATERIAL REINFORCEMENT LEARNING ===")
        # Enhanced reinforcement with biomaterial data
        final_acc = asthma_model.reinforce_with_biomaterial(biomaterial_ai, n_iterations=5)

        print(f"\n🎯 FINAL RESULTS SUMMARY")
        print("="*40)
        print(f"Baseline Accuracy: {baseline_acc:.4f}")
        print(f"Final Accuracy:    {final_acc:.4f}")
        print(f"Improvement:       {final_acc - baseline_acc:+.4f}")

        # Save model
        joblib.dump({
            'model': asthma_model.model,
            'preprocessor': asthma_model.preprocessor,
            'training_history': asthma_model.training_history
        }, "asthma_biomaterial_model.pkl")
        
        print("\n💾 Saved integrated model as 'asthma_biomaterial_model.pkl'")
    
    except FileNotFoundError:
        print("❌ Asthma data file not found. Please ensure 'data/asthma_disease_data.csv' exists.")
    except Exception as e:
        print(f"❌ Error in enhanced biomaterial pipeline: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ ENHANCED BIOMATERIAL PIPELINE COMPLETED!")