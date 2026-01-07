# Saudi Stock Forecasting with Deep Learning Ensembles

## Title
**Ensemble Deep Learning Models for StockPrice Forecasting: Evidence from the Saudi Markets**

---

## Description

This repository presents a **research-grade and reproducible pipeline** for stock price forecasting in the Saudi stock market using **deep learning ensemble models**. The framework integrates **34 technical indicators**, **feature correlation analysis**, and **multivariate multi-step forecasting** using recurrent neural networks.

The proposed methodology is evaluated on major Saudi companies listed on the **Tadawul All Share Index (TASI)** and is further **replicated on U.S. stocks** to assess generalization and robustness. The repository includes all stages of the experimental workflow, from data preprocessing to model training, ensemble construction, and performance evaluation.

---

## Dataset Information

### Saudi Stock Market (KSA)

- **Market**: Tadawul All Share Index (TASI)
- **Source**: Publicly available Kaggle dataset  
  https://www.kaggle.com/datasets/salwaalzahrani/saudi-stock-exchange-tadawul
- **Scope**:
  - Ten Saudi companies representing major market sectors
  - Multivariate daily time series
- **Features**:
  - Raw market variables (e.g., Open, High, Low, Volume)
  - Engineered technical indicators (34 indicators)
- **Target Variable**:
  - Closing price

### U.S. Stock Market (Replication Study)

- **Market**: S&P 500
- **Source**: Kaggle dataset  
  https://www.kaggle.com/datasets/paultimothymooney/stock-market-data
- **Selected Companies**:
  - Apple (AAPL)
  - Bank of America (BAC)
  - Alphabet Inc. (GOOG)
  - Johnson & Johnson (JNJ)
- **Time Period**:
  - December 31, 2001 – March 5, 2020 (aligned with Saudi data)
- **Features**:
  - Date, Open, High, Low, Close, Adjusted Close, Volume

---

## Code Information (Repository Structure)

The repository consists of modular Jupyter notebooks organized according to the experimental pipeline:

### 1. Feature Correlation Analysis
**Notebook**: `PaersonCorrelation.ipynb`

- Pearson correlation heatmaps
- Multicollinearity analysis
- Identification of highly correlated and redundant indicators
- Exploration of indicator–price relationships

---

### 2. Feature Engineering
**Notebook**: `PrepareFeatursOfData.ipynb`

- Computation of **34 technical indicators**
- Handling of missing values introduced by rolling calculations
- Preparation of feature-enriched datasets

---

### 3. Standalone Deep Learning Models

#### Saudi Market Models
**Notebook**:  
`Standalone_Saudi_BiRNN_GRU_LSTM_multivriate_multistep_model.ipynb`

Includes:
- Rolling-window transformation
- Normalization and scaling
- Multivariate multi-step (5-day ahead) forecasting
- Model training, hyperparameter configuration, and evaluation

Implemented models:
- BiRNN
- GRU
- LSTM

#### U.S. Market Replication
**Notebook**:  
`US_Standalone_BiRNN_GRU_LSTM_multivriate_multistep_model.ipynb`

- Applies the same methodology used for Saudi data
- Evaluates model generalization on U.S. stocks

---

### 4. Ensemble Models

#### Averaging Ensembles
- Combines predictions from standalone models
- Reduces variance and improves robustness

#### Stacking Ensembles
**Notebooks**:
- `Ensemble Stacking in Neural Networks.ipynb`
- `EnsembleUsStacking`

Includes:
- GRU–LSTM stacking
- BiRNN–LSTM stacking
- Meta-learner design
- Cross-company ensemble evaluation

---

### 5. Experiment Results and Visualization
**Notebooks**:
- `TestingResults.ipynb`
- `Plot.ipynb`

Includes:
- 5-day ahead prediction plots
- Evaluation metrics:
  - MAE
  - RMSE
  - MRE
  - R²
- Saudi vs. U.S. performance comparison
- Visualization of ensemble improvements

---

## Methodology

1. Load and clean raw stock market data
2. Compute 34 technical indicators
3. Perform feature correlation and multicollinearity analysis
4. Construct rolling-window multivariate datasets
5. Normalize and scale input features
6. Train standalone deep learning models (BiRNN, GRU, LSTM)
7. Build ensemble models using averaging and stacking
8. Evaluate forecasting performance using standard regression metrics
9. Replicate experiments on U.S. stocks to validate generalization

---

## Usage Instructions

### Clone the Repository
```bash
git clone https://github.com/ranabameer/saudi-stock-forecasting-ensembles
cd saudi-stock-forecasting-ensembles
```
### Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
### Install Dependencies
All required dependencies are listed in requirements.txt:
```bash
pip install -r requirements.txt
```
### Recommended Execution Order

`PaersonCorrelation.ipynb`

`PrepareFeatursOfData.ipynb`

Standalone model notebooks (Saudi and U.S.)

Ensemble notebooks

`TestingResults.ipynb`

`Plot.ipynb`

Ensure dataset paths inside notebooks are updated to match your local directory structure.

## Requirements

The project dependencies are defined in `requirements.txt`:
```bash
numpy>=1.21
pandas>=1.4
scipy>=1.8
matplotlib>=3.5
seaborn>=0.12
scikit-learn>=1.1
permetrics>=1.5
tensorflow>=2.10
keras>=2.10
pandas-ta>=0.3.14b0
python-dateutil>=2.8
```
## Materials & Methods (Reproducibility)

To ensure reproducibility:

- Use the same dataset versions
  
- Maintain identical rolling-window configurations
  
- Apply normalization using training data only
  
- Fix random seeds for all experiments
  
- Preserve hyperparameter settings and evaluation metrics
  
- Document hardware and software configurations

## Computing Infrastructure

Experiments were conducted using Google Colaboratory (Colab), a cloud-based Jupyter notebook environment. The computational setup is summarized as follows:

- Platform: Google Colaboratory

- Operating System: Linux (Ubuntu-based, managed by Colab)

- CPU: Intel Xeon (cloud virtualized, exact model varies by session)

- RAM: Approximately 12–16 GB (standard Colab runtime)

- GPU: NVIDIA Tesla T4 / P100 (when GPU runtime is enabled)

- Python Version: Python 3.9

- Deep Learning Framework: TensorFlow / Keras

- Execution Environment: Jupyter Notebook (cloud-based)

## Citations

Alzahrani, S. (2020). Saudi stock exchange (tadawul). Available at: https://www.kaggle.com/489
datasets/salwaalzahrani/saudi-stock-exchange-tadawul.490
Ampomah, E. K., Qin, Z., and Nyame, G. (2020). Evaluation of tree-based ensemb

Mooney, P. (2022). Stock market data (nasdaq, nyse, s and p500). https://www.535
kaggle.com/datasets/paultimothymooney/stock-market-data?resource=536
download-directory.537


## License

This repository is provided for **research and academic purposes only**.

The code and materials are not licensed for commercial use or redistribution at this stage.  
A formal open-source license may be added in the future upon publication or project completion.



