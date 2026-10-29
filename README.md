# Wine Quality Prediction and Customer Segmentation

Two machine learning tasks from my Machine Learning 1 module at Ngee Ann Polytechnic (Apr 2026 semester).

**Task 1: wine quality.** Predicts the quality score of white wines from 11 chemical measurements. I removed 937 duplicate rows, capped outliers with the IQR rule and compared Linear Regression with a Decision Tree. The untuned tree overfitted badly. Tuning it with GridSearchCV helped, but Linear Regression still did best (test RMSE 0.749, R² 0.289).

**Task 2: customer segmentation.** Cleans 541,909 online retail transactions down to 397,884, builds Recency, Frequency and Monetary features for 4,338 customers, and clusters them with K-Means and hierarchical (Ward) clustering using k = 4. K-Means separated the groups slightly better (silhouette 0.310 vs 0.282). The four segments are Champions, Occasional Mid-Value Shoppers, Low-Value Infrequent Shoppers and At-Risk/Lost Customers.

## Running it

You need pandas, numpy, matplotlib, seaborn, scikit-learn, scipy and statsmodels.

The datasets are not included. Put `winequality-white.csv` (White Wine Quality) and `online_retail.csv` (Online Retail) in the same folder as the notebook, then run all cells.
