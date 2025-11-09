# Asthma Prediction AI System 🫀

A comprehensive machine learning pipeline for predicting asthma disease using clinical data, biomaterial design, and reinforcement learning optimization.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Requirements](#data-requirements)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This project provides an end-to-end AI system for asthma disease prediction that integrates multiple machine learning approaches:

- **Classical ML Models**: Compare and select the best performing classifier
- **Biomaterial Design**: AI-driven cellular response prediction
- **Reinforcement Learning**: Automated hyperparameter optimization
- **Single-Cell RNA Integration**: Process and analyze single-cell sequencing data

The system is designed for researchers and data scientists working in respiratory disease prediction and biomedical AI applications.

## 🚀 Features

### Core Components

1. **Asthma Prediction Pipeline**
   - Automated data preprocessing and feature engineering
   - Multiple classifier comparison (Random Forest, Logistic Regression, Gradient Boosting)
   - Best model selection and persistence

2. **Smart Biomaterial Design**
   - Optimize biomaterial compositions for specific cellular responses
   - Predict proliferation, differentiation, and adhesion scores
   - Reinforcement learning integration

3. **RL Hyperparameter Optimization**
   - Automated model tuning using reinforcement learning
   - Adaptive parameter search strategies
   - Performance tracking and comparison

4. **Single-Cell RNA Processing**
   - Quality control and preprocessing of single-cell data
   - Feature aggregation from cell-level to sample-level
   - Integration with clinical prediction models

## 🛠️ Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum
- 1GB free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/asthma-pred-model.git
   cd asthma-pred-model
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv asthma_env
   
   # On Windows:
   asthma_env\Scripts\activate
   # On macOS/Linux:
   source asthma_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   If requirements.txt is not available, install core packages:
   ```bash
   pip install numpy pandas scikit-learn joblib matplotlib seaborn
   ```

## 🎯 Usage

### Run Complete Pipeline

Execute the entire pipeline with one command:

```bash
python run_all.py
```

This will automatically:
- Create sample data if needed
- Train asthma prediction models
- Process single-cell RNA data
- Run biomaterial design system
- Execute RL optimization
- Generate comprehensive results

### Run Individual Components

**1. Train Asthma Prediction Model**
```bash
python train_asthma_pipeline.py --data ./data/asthma_disease_data.csv --target Diagnosis --outdir ./artifacts
```

**2. Process Single-Cell RNA Data**
```bash
python scrna_preprocess_and_embed.py --adata ./data/scrna_sample/sample_scrna.h5ad --sample-key patient_id --out ./scrna_artifacts
```

**3. Run Biomaterial Design**
```bash
python smart_biomaterial_design.py
```

**4. Execute RL Optimization**
```bash
python rl_prorl_trainer.py
```

## 📁 Project Structure

```
asthma-pred-model/
├── data/                           # Data directory
│   ├── asthma_disease_data.csv     # Sample clinical data
│   └── scrna_sample/               # Single-cell RNA data
├── artifacts/                      # Generated models and results
├── scrna_artifacts/               # Single-cell processing outputs
├── train_asthma_pipeline.py       # Main training pipeline
├── smart_biomaterial_design.py    # Biomaterial AI system
├── rl_prorl_trainer.py           # RL hyperparameter optimization
├── scrna_preprocess_and_embed.py # Single-cell RNA processor
├── run_all.py                    # Complete pipeline runner
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 📊 Data Requirements

### Clinical Data Format
Your clinical data should be a CSV file with the following structure:

| Age | Gender | BMI | Smoking_Status | FEV1 | Eosinophil_Count | ... | Diagnosis |
|-----|--------|-----|----------------|------|------------------|-----|-----------|
| 45  | Male   | 26.5| Former         | 85   | 0.3              | ... | 0         |
| 32  | Female | 24.1| Never          | 95   | 0.2              | ... | 1         |

**Required:**
- CSV format with header row
- Mixed data types supported (numerical + categorical)
- Target column should be binary (0/1) for classification

### Single-Cell RNA Data
Supported formats:
- `.h5ad` (AnnData format)
- CSV counts matrix + metadata CSV

## 📈 Results

After running the pipeline, check these output directories:

### `artifacts/` - Main Results
- `best_asthma_model.pkl` - Best performing model
- `training_results.csv` - Performance metrics for all models
- `README_model.txt` - Training summary and parameters

### `scrna_artifacts/` - Single-Cell Results
- `sample_scvi_aggregated.csv` - Patient-level features from single-cell data
- `preprocessing_summary.json` - Processing statistics

### Expected Performance
On sample data, typical results are:
- **Baseline Accuracy**: 0.75-0.85
- **RL Optimized**: +2-5% improvement
- **Training Time**: 2-5 minutes (depending on hardware)

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   # Install missing packages
   pip install numpy pandas scikit-learn joblib
   ```

2. **Data File Not Found**
   ```bash
   # The pipeline will create sample data automatically
   # Or ensure your data files are in the ./data/ directory
   ```

3. **Memory Issues**
   - Reduce dataset size in configuration
   - Use smaller models (reduce n_estimators)
   - Process data in chunks

4. **Windows Unicode Errors**
   - Use the provided Windows-compatible scripts
   - Run in PowerShell instead of Command Prompt

### Getting Help

1. Check the `artifacts/` directory for detailed error logs
2. Ensure all dependencies are installed
3. Verify your data format matches expectations
4. Check the troubleshooting section in individual script documentation

## 🧪 Advanced Usage

### Custom Data Integration

To use your own data:

1. **Replace clinical data**: Put your CSV file in `data/` directory
2. **Update target variable**: Modify the `--target` parameter
3. **Add single-cell data**: Place `.h5ad` files in `data/scrna_sample/`

### Model Customization

Edit these files to customize models:
- `train_asthma_pipeline.py`: Add new classifiers or preprocessing
- `rl_prorl_trainer.py`: Modify RL optimization parameters
- `smart_biomaterial_design.py`: Adjust biomaterial simulation

### Extension Examples

```python
# Add new model to pipeline (in train_asthma_pipeline.py)
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "XGBoost": xgb.XGBClassifier(),  # Add XGBoost
    "SVM": SVC(probability=True)     # Add Support Vector Machine
}
```


### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
   ```bash
   pip install black flake8 pytest
   ```
4. Make your changes and test
5. Submit a pull request

### Code Style
- Use Black code formatting
- Include docstrings for new functions
- Add tests for new functionality
- Update documentation accordingly

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Clinical researchers for asthma data insights
- scVI team for single-cell processing tools
- Scikit-learn for machine learning infrastructure
- Stable-Baselines3 for reinforcement learning components

## 📞 Support

- **Linkedin**: https://www.linkedin.com/in/sazzadhossain1461/ 
- **Email**: sazzadhossain74274@gmail.com

---


---

<div align="center">

**If you find this project useful, please consider giving it a ⭐️ on GitHub!**

</div>
