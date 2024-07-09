import pandas as pd

# Create apps data
apps_data = {
    'App': ['App A', 'App B', 'App C', 'App D', 'App E'],
    'Category': ['Game', 'Education', 'Productivity', 'Game', 'Education'],
    'Rating': [4.5, 4.0, 4.7, 3.8, 4.2],
    'Size': ['25M', '15M', '30M', '40M', '20M'],
    'Installs': [100000, 50000, 200000, 30000, 75000]
}

# Create reviews data
reviews_data = {
    'App': ['App A', 'App B', 'App C', 'App D', 'App E'],
    'Review': ['Great game, really enjoyed it!', 'Useful app for students', 'Helps me stay organized, love it!', 'Could be better, crashes often', 'Informative and easy to use']
}

# Convert to DataFrame
apps_df = pd.DataFrame(apps_data)
reviews_df = pd.DataFrame(reviews_data)

# Save to CSV
apps_df.to_csv('apps_data.csv', index=False)
reviews_df.to_csv('reviews_data.csv', index=False)
