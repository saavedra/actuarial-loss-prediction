## Actuarial Loss Prediction

### Description

The notebooks in this repo present a proposed solution to the [Actuarial Loss Prediction](https://www.kaggle.com/c/actuarial-loss-estimation) challenge on Kaggle. This solution is not fully developed, as it was completed within the time constraint of approximately one week (enough to test one approach of handling the categorical variables).

### File Contents
* `1. exploration.ipynb`: Thorough exploration of numerical and categorical variables in the dataset, along with key insights.
* `2. initial modelling.ipynb`: Initial modeling using the insights from the exploration of numerical and categorical variables.
* `3. NLP processing simple.ipynb`: Processing of claim descriptions— a critical step given the dataset’s nature.  
  The approach involves using an LLM to extract key events and affected body parts from descriptions. These extracted terms are then vectorized using an embedding model and one-hot encoding for the most common ones.  
  The extracted terms are not perfect, leaving room for improvement.
* `4. NLP exploration.ipynb`: Analyzes the impact of embeddings and one-hot encoded terms on prediction performance.  
  The descriptions alone have predictive power comparable to numerical and categorical variables.
* `5. training.ipynb`: Combines all processed features into a single dataset and trains using **XGBoost**, with hyperparameter tuning via **Optuna** and **K-Fold Cross-Validation (CV)**.
* `6. training xgboost - no outliers.ipynb`: A version of training excluding extreme values in the dataset (the target variable is heavily skewed to the right). Training is done using **K-Fold CV**.
* `7. training lightgbm.ipynb`: Trains using **LightGBM**, with hyperparameter tuning via **Optuna** and **K-Fold Cross-Validation (CV)**.
* `8. training on log.ipynb`: Trains using a **log transformation** on the target variable instead of **Box-Cox**, using **K-Fold CV**.
* `9. hybrid.ipynb`: Hybridization of some of the best-performing models, applying **K-Fold CV** and combining predictions from multiple models.

### Tools Used
The project was developed in **Python**, leveraging the standard Data Science stack (`pandas`, `scikit-learn`, `XGBoost`, `LightGBM`).  
- **`Optuna`**: Used for hyperparameter optimization.  
- **`MLFlow`**: Used for tracking experiments and model performance.  
- **`K-Fold Cross-Validation (CV)`**: Used for all training processes to ensure robust model evaluation.
- **`qwen2.5-0.5b-instruct`** Used for text extraction from the claim descriptions. Running locally on **LM Studio**.
- **`nomic-embed-text-v1.5`** Used to convert text to embeddings. Running locally on **LM Studio**.
