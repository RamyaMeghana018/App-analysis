import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler

# Load datasets
apps_data = pd.read_csv('apps_data.csv')
reviews_data = pd.read_csv('reviews_data.csv')

# Function Definitions
def convert_size_to_numeric(size):
    if 'M' in size:
        return float(size.replace('M', '')) * 1e6
    elif 'k' in size:
        return float(size.replace('k', '')) * 1e3
    return float(size)

def analyze_sentiment(review):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(review)
    return sentiment['compound']

# Data preprocessing
apps_data.dropna(inplace=True)
reviews_data.dropna(inplace=True)

# Feature Engineering
apps_data['Size'] = apps_data['Size'].apply(lambda x: convert_size_to_numeric(x))
nltk.download('vader_lexicon')
reviews_data['Sentiment'] = reviews_data['Review'].apply(lambda x: analyze_sentiment(x))

# Merge datasets
merged_data = pd.merge(apps_data, reviews_data, on='App', how='inner')

# Data Visualization
sns.pairplot(merged_data, hue='Category')
plt.show()

# Model Training
X = merged_data[['Category', 'Size', 'Sentiment', 'Installs']]
y = merged_data['Rating']
X = pd.get_dummies(X, columns=['Category'], drop_first=True)  # One-hot encode categorical variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the model with modified hyperparameters
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R2 Score: {r2_score(y_test, y_pred)}')

# Visualize results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.show()

# Feature Importance
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print(importance_df)

# Fine-Tuning with different model (example: Gradient Boosting)
from sklearn.ensemble import GradientBoostingRegressor

# Initialize the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred_gb = gb_model.predict(X_test)
print(f'Mean Squared Error (GB): {mean_squared_error(y_test, y_pred_gb)}')
print(f'R2 Score (GB): {r2_score(y_test, y_pred_gb)}')

# Visualize results
plt.scatter(y_test, y_pred_gb)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings (Gradient Boosting)')
plt.show()
