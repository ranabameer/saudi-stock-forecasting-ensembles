# saudi-stock-forecasting-ensembles
Enhanced Saudi stock price forecasting using 34 technical indicators and deep learning ensembles (LSTM, GRU, BiRNN). Includes preprocessing, rolling-window multivariate modeling, and evaluation across multiple companies with replication on U.S. stocks for validation.

Saudi Stock Price Forecasting with Technical Indicators & Deep Learning Ensembles

This repository provides a complete research-grade pipeline for Saudi stock price forecasting using 34 technical indicators, feature correlation analysis, and deep learning ensemble architectures (LSTM, GRU, BiRNN, Stacking, Averaging).
The project includes full preprocessing, modeling, evaluation, and replication on U.S. market stocks.

### 1. Feature Correlation Analysis

Understanding relationships among technical indicators and price movements is a key step in building reliable forecasting models.
This repository begins with detailed feature correlation visualizations and analysis.

Notebook:
📌 PaersonCorrelation.ipynb
This notebook includes:

Pearson correlation heatmaps between indicators and target prices

Multicollinearity inspection

Highlighting features with strong predictive signals

Identifying redundant or highly correlated inputs

Visual exploration of indicator–price relationships

These insights help optimize the feature set used across all forecasting models.

### 2. Feature Engineering

Notebook:
📌 PrepareFeatursOfData.ipynb

Includes:

Calculation of 34 technical indicators

Rolling-window transformations

Normalization and scaling

Handling missing values

Multi-step (5-day ahead) target generation

Exporting ready-to-train datasets for all companies

### 3. Standalone Deep Learning Models
🇸🇦 Saudi Market Models

Notebook:
📌 Standalone_Saudi_BiRNN_GRU_LSTM_multivriate_multistep_model.ipynb

Implements:

BiRNN

GRU

LSTM

Multivariate, multi-step forecasting

Model training loops, hyperparameters, evaluation

🇺🇸 U.S. Market Replication

Notebook:
📌 US_Standalone_BiRNN_GRU_LSTM_multivriate_multistep_model.ipynb

Verifies model stability on U.S. stocks for generalization.

###  4. Ensemble Models
 Averaging Ensembles

Combine multiple standalone models to improve robustness and reduce variance.

Stacking Ensembles

Notebook:
📌 EnsembleUsStacking
📌 Ensemble Stacking in Neural Networks.ipynb

Includes:

Stacking GRU-LSTM

Stacking BiRNN-LSTM

Meta-learner design

Optimizing ensemble weights

Cross-company ensemble evaluation

### 5. Experiment Results & Visualization

Notebook:
📌 TestingResults.ipynb
📌 Plot.ipynb

Includes:

5-day ahead prediction plots

Loss curves

Evaluation metrics (MAE, RMSE, MRE, R²)

Saudi vs. U.S. performance comparison

Visualization of ensemble improvements

📂 Project Structure
├── PaersonCorrelation.ipynb
├── PrepareFeatursOfData.ipynb
├── Standalone_Saudi_BiRNN_GRU_LSTM_multivriate_multistep_model.ipynb
├── US_Standalone_BiRNN_GRU_LSTM_multivriate_multistep_model.ipynb
├── Ensemble Stacking in Neural Networks.ipynb
├── EnsembleUsStacking
├── Plot.ipynb
├── TestingResults.ipynb
└── README.md

📦 Key Features

34+ engineered technical indicators

Multivariate 5-day-ahead forecasting

RNN-based deep learning models

Ensemble learning (Averaging + Stacking)

Cross-market validation (Saudi + U.S.)

Reproducible and modular design

Usage

Clone the repository:

git clone https://github.com/yourusername/your-repo-name
cd your-repo-name


Run notebooks in order:

PaersonCorrelation.ipynb

PrepareFeatursOfData.ipynb

Standalone model notebooks

Ensemble notebooks

TestingResults.ipynb and Plot.ipynb
