**Problem 1:  Predictive Maintenance in Manufacturing (Regression with Overfitting Resolution**

a) Overfitting and the Bias-Variance Tradeoff
In your code, the goal is to predict the number of days until equipment failure using sensor data. Overfitting occurs when the model learns noise in the training data, such as random spikes in vibration or temperature, rather than generalizable patterns. This leads to poor performance on unseen data.

Bias-Variance Tradeoff:

- High bias models (e.g., linear regression) oversimplify relationships, leading to underfitting.
- High variance models (e.g., deep decision trees) memorize training data, leading to overfitting.

The use of Random Forest strikes a balance: It reduces variance by averaging predictions across multiple decision trees and It maintains low bias by capturing nonlinear relationships between features and failure time.


b) Random forest was used because it reduces overfitting by training multiple trees on bootstrapped samples and averaging their outputs and also handles noisy sensor data and complex feature interactions effectively.

Steps taken: 
1. Loaded data from "Question 1 dataests.csv"
2. _Feature Engineering_
   - Created interaction terms:
   - Vibration × Runtime
   - Temperature × Pressure
   - Normalized features using StandardScaler.
3. _Model Selection_
   - Applied **Random Forest Regressor** to reduce overfitting and capture nonlinear relationships.
4.  _Model Training_  
   - Split data into 80/20 train-test sets.
   - Trained model using RandomForestRegressor(n_estimators=100).
5. _Evaluation_
   - Calculated RMSE and R² on test data.
   - Performed 5-fold cross-validation.
   - Visualized:
   - Feature importance
   - Actual vs. predicted scatter plot
   - Residuals distribution

c) Performance evaluation:
- Test RMSE: 159.00
Measures average prediction error. A value of 159.00 indicates high prediction error on unseen data.

- Test R² Score: -0.20
Negative value (-0.20) means the model performs worse than simply predicting the mean(average failure time).

- Cross-Validated RMSE: 145.59
Confirms consistent underperformance across folds and Suggests that the model may still be overfitting or missing key predictive features, indicating poor generalization.


Visual Insights:
- Feature Importance Plot: Showed which sensor readings most influenced predictions, with Runtime and Vibration × Runtime likely being dominant.

- Actual vs Predicted Scatter Plot: Revealed weak correlation between predicted and actual values, confirming poor model fit.

- Residuals Histogram: Displayed wide error distribution, suggesting the model struggles to capture underlying patterns.


d) To further improve the predictive maintenance model, feature engineering plays a critical role in enhancing the model’s ability to capture meaningful patterns in sensor data. In the implementation, valuable interaction terms such as Vibration × Runtime and Temperature × Pressure are introduced , which helps the model understand compound effects that may signal mechanical stress or thermal overload. Building on this, additional engineered features could include polynomial terms like Vibration or Pressure to capture nonlinear relationships, especially where sensor readings escalate exponentially before failure. Applying log transformations to skewed features such as Runtime can also stabilize variance and make the model more sensitive to subtle changes in operating conditions.

If time-series data is available, rolling averages or lagged features could be introduced to reflect trends leading up to equipment failure. For example, a moving average of temperature over the last few readings might reveal gradual overheating that a single reading would miss. Another technique that could be used is Z-score normalization, which standardizes sensor values and highlights anomalies, readings that deviate significantly from the norm. These anomalies can be flagged and used as features or triggers for inspection.

Beyond regression, these engineered features open the door to anomaly detection applications. For instance, clustering techniques like K-Means could be used to group sensor behaviour into operational states, and any readings that fall outside these clusters could be treated as potential early warnings. These anomaly flags can then be integrated into supervised models like Random Forest or SVM to improve predictive accuracy. Ultimately, feature engineering not only strengthens the regression model but also transforms raw sensor data into actionable insights for maintenance teams, enabling earlier interventions and reducing costly downtime.


**Problem 2: Fraud Detection in Banking (Clustering and Classification with Feature Engineering)**

a) In the  code, K-Means clustering is applied to unlabelled transaction data using features like Amount, Time_Hour, Location, and Merchant. Numerical features are scaled and categorical ones are encoded using StandardScaler and OneHotEncoder.

Justification: K-Means is an unsupervised algorithm ideal for pattern discovery when fraud labels are missing. It groups transactions based on similarity, helping detect anomalies such as high-value transactions occurring late at night. These clusters can flag potential fraud without needing labelled examples.

Choosing k: The elbow method is used, plotting the within-cluster sum of squares (WCSS) across values of k from 1 to 9. The “elbow” point, where WCSS reduction slows, suggests the optimal number of clusters.


b) Supervised Learning with Naïve Bayes and Random Forest:
Both a Naïve Bayes classifier and a Random Forest classifier were trained on labelled data (Is_Fraud (Labeled Subset)), using the same pre-processed features. SMOTE was also applied, to balance the classes before training Random Forest.

Naïve Bayes Justification: Chosen for its simplicity and ability to handle categorical features probabilistically. It assumes feature independence, which reduces overfitting.

Bias-Variance Tradeoff:

- Naïve Bayes has high bias (simplified assumptions) and low variance, making it stable but less flexible.
- Random Forest has lower bias and moderate variance, and with SMOTE, it handled class imbalance effectively.


c) Feature Engineering to Improve Predictions:

- Time_Category: Binned Time_Hour into Night, Morning, Afternoon, Evening.
- Amount_Normalized: Scaled transaction amounts.
- High_Amount: Flagged transactions above the 95th percentile.

These features help the model distinguish normal from suspicious behaviour. For example: Late-night high-value transactions may indicate fraud.
Normalizing Amount reduces skew and improves sensitivity and Binning time captures behavioural patterns.

Feature engineering amplified signals from the minority class (fraud), improving classification performance and helping SMOTE generate more meaningful synthetic samples. It Improves model interpretability, enhances separation between fraud and non-fraud and Addresses class imbalance by amplifying minority class signals.


d) Evaluation and Comparison:

key:  

- F1 Score: Balances precision and recall, ideal for imbalanced data.
- Confusion Matrix: Shows classification accuracy.
- 10-fold Cross-Validation: Applied to Naïve Bayes.


Naïve Bayes:
- Confusion Matrix: [[7, 12], [0, 1]]
- Cross-Validated F1 Score: 0.17

Interpretation:

-  The model correctly identified only 1 fraudulent transaction and misclassified 12 legitimate ones.
-  High bias and poor separation between classes due to:
-  Strong feature independence assumption
-  Severe class imbalance (few fraud cases)
-  Naïve Bayes was not effective in this context, despite its simplicity and speed.

Random Forest (with SMOTE):
Confusion Matrix: [[20, 0], [0, 18]]
F1 Score: 1.00

Interpretation:

-  Perfect classification on the test set: all fraud and non-fraud cases correctly identified.
-  SMOTE (Synthetic Minority Oversampling Technique) successfully addressed class imbalance by generating synthetic fraud samples.
-  Random Forest handled feature interactions and categorical variables effectively.


Steps taken:

1. _Data loading_, using "Question 2 datasets.csv"
2. _Feature Engineering_
   - Binned time into categories (Time_Category).
   - Normalized transaction amounts (Amount_Normalized).
   - Flagged high-value transactions (High_Amount).
   - Encoded categorical features (Location, Merchant) using OneHotEncoder.
3. _Unsupervised Learning_ 
   - Applied K-Means Clustering to group transactions.
   - Used elbow method to determine optimal k.
4. _Supervised Learning_
   - Trained Naïve Bayes and Random Forest classifiers.
   - Applied SMOTE to balance fraud and non-fraud classes.
   - Evaluated using F1-score, confusion matrix, and ROC curve.
