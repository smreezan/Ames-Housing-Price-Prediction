#  Predictive Housing Price Model
**Ames Residential Real Estate Valuation Project**

##  Project Overview
This repository contains a high-precision machine learning pipeline designed to predict residential home prices in Ames, Iowa. The project was developed to meet strict industry accuracy standards, specifically targeting a Root Mean Squared Error (RMSE) below **$17,699.59**.

##  Final Performance Results
The final model was evaluated on a 20% hold-out test set. The **Safety Ensemble** successfully exceeded all performance benchmarks.

| Model | RMSE (Final Scale) | MAPE (%) | Status |

| **Safety Ensemble (Champion)** | **$16,321.57** | **6.52%** |
| Linear Ridge | $16,159.99 | 6.87% |  TARGET ACHIEVED |
| Gradient Boosting (Huber) | $18,263.50 | 7.12% | Baseline |
| Random Forest | $19,878.42 | 8.02% | Baseline |

##  Methodology & Pipeline
The core logic is contained within the primary notebook: `Predictive_Housing_Price_Model.ipynb`.

### 1. Advanced Feature Engineering
* **Quality_Power:** An exponential interaction between `OverallQual` and `GrLivArea` to capture the premium value of luxury estates.
* **Neighborhood Richness:** A mapping of location-based median prices to capture geographical equity.
* **Neigh_PPSF:** Price-per-square-foot metrics to evaluate location efficiency.

### 2. Surgical Data Cleaning
* **Outlier Removal:** Implemented a multi-stage filter removing 293 anomalous records (top/bottom 3% of Price-per-SF and 2.0 Sigma on size/quality) to stabilize the model's learning variance.

### 3. Model Ensembling
* Developed a **VotingRegressor** that balances the stability of **Ridge Regression** with the non-linear precision of **Gradient Boosting (Huber Loss)**.

##  Project Structure
* `Predictive_Housing_Price_Model.ipynb`: The primary end-to-end Python pipeline (Preprocessing, Training, Evaluation).
* `train.csv`: The raw Ames Housing Dataset.
* `README.md`: Project documentation and setup instructions.

##  How to Run
1. **Environment:** Ensure you have Python installed with the following libraries:
   `pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn`
2. **Execution:** Open the `Predictive_Housing_Price_Model.ipynb` file in Jupyter Notebook or VS Code and run all cells.
3. **Data Path:** Ensure the `train.csv` file is located in the same directory as the notebook.

##  Interpretability
The model utilizes **SHAP (SHapley Additive exPlanations)** values to provide transparency into how specific home features (like Square Footage and Quality) influence the final price prediction, ensuring the model's logic is grounded in real estate fundamentals.