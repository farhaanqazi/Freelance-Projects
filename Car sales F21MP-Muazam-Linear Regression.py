#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import statistics
import matplotlib.pyplot as plt


# In[104]:


df = pd.read_csv('D:\Projects\car_prices.csv')


# In[105]:


df.head()


# In[106]:


print(df)


# In[107]:


df.head()


# In[108]:


df.describe()


# In[109]:


df.shape


# In[110]:


df.columns


# In[111]:


df.info()


# In[112]:


df.isnull().sum()


# In[113]:


df.duplicated().sum()


# In[114]:


# Check for null values in all columns
null_values_columns = df.isnull().any()

# Print columns with null values
print("Columns with null values:")
for column, has_null in null_values_columns.items():
    if has_null:
        print(column)


# In[115]:


# Check for null values in all columns
null_values_columns = df.isnull().sum()

# Print columns with null values and their respective count of null values
print("Columns with null values and their counts:")
for column, null_count in null_values_columns.items():
    if null_count > 0:
        print(f"{column}: {null_count}")


# In[116]:


# Identify non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
print(non_numeric_columns)


# In[117]:


# Calculate frequencies of each category
frequency_map_model = df['model'].value_counts(normalize=True)

# Map frequencies to the original categories
df['frequency_encoded_model'] = df['model'].map(frequency_map_model)

print(df['frequency_encoded_model'])


# In[118]:


# Calculate frequencies of each category
frequency_map_model = df['trim'].value_counts(normalize=True)

# Map frequencies to the original categories
df['frequency_encoded_trim'] = df['trim'].map(frequency_map_model)

print(df['frequency_encoded_trim'])


# In[119]:


# Calculate frequencies of each category
frequency_map_model = df['body'].value_counts(normalize=True)

# Map frequencies to the original categories
df['frequency_encoded_body'] = df['body'].map(frequency_map_model)

print(df['frequency_encoded_body'])


# In[120]:


# Calculate frequencies of each category
frequency_map_model = df['state'].value_counts(normalize=True)

# Map frequencies to the original categories
df['frequency_encoded_state'] = df['state'].map(frequency_map_model)

print(df['frequency_encoded_state'])


# In[121]:


# Calculate frequencies of each category
frequency_map_model = df['color'].value_counts(normalize=True)

# Map frequencies to the original categories
df['frequency_encoded_color'] = df['color'].map(frequency_map_model)

print(df['frequency_encoded_color'])


# In[122]:


# Calculate frequencies of each category
frequency_map_model = df['seller'].value_counts(normalize=True)

# Map frequencies to the original categories
df['frequency_encoded_seller'] = df['seller'].map(frequency_map_model)

print(df['frequency_encoded_seller'])


# In[123]:


# Calculate frequencies of each category
frequency_map_model = df['interior'].value_counts(normalize=True)

# Map frequencies to the original categories
df['frequency_encoded_interior'] = df['interior'].map(frequency_map_model)

print(df['frequency_encoded_interior'])


# In[124]:


# Calculate frequencies of each category
frequency_map_model = df['make'].value_counts(normalize=True)

# Map frequencies to the original categories
df['frequency_encoded_make'] = df['make'].map(frequency_map_model)

print(df['frequency_encoded_make'])


# In[125]:


df


# In[126]:


# Drop the specified columns
df.drop(columns='model', inplace=True)
df.drop(columns='trim', inplace=True)
df.drop(columns='state', inplace=True)
df.drop(columns='body', inplace=True)
df.drop(columns='condition', inplace=True)
df.drop(columns='color', inplace=True)
df.drop(columns='interior', inplace=True)
df.drop(columns='make', inplace=True)
df.drop(columns='seller', inplace=True)


# In[127]:


# Perform one-hot encoding
one_hot_encoded = pd.get_dummies(df['transmission'], prefix='transmission')

# Concatenate one-hot encoded columns with the original DataFrame
df_encoded = pd.concat([df, one_hot_encoded], axis=1)

print(df_encoded)


# In[130]:


df_encoded.head()


# In[131]:


df_new = df_encoded


# In[132]:


df_new.drop(columns='transmission', inplace=True)


# In[133]:


df_new.head()


# In[135]:


# Define a function to detect outliers using z-score
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    return (z_scores > threshold).any()

# Check for outliers in each column
columns_with_outliers_zscore = df_new.apply(detect_outliers_zscore)

# Print columns with outliers using z-score method
print("Columns with outliers (z-score):")
print(columns_with_outliers_zscore)


# In[137]:


# Define a function for Winsorization
def winsorize_series(series, lower_pct=0.05, upper_pct=0.95):
    lower_bound = series.quantile(lower_pct)
    upper_bound = series.quantile(upper_pct)
    series_winsorized = series.clip(lower=lower_bound, upper=upper_bound)
    return series_winsorized

# Apply Winsorization to specific columns (e.g., math_score, reading_score, writing_score)
df_new['year'] = winsorize_series(df_new['year'])
df_new['odometer'] = winsorize_series(df_new['odometer'])
df_new['mmr'] = winsorize_series(df_new['mmr'])
df_new['sellingprice'] = winsorize_series(df_new['sellingprice'])
df_new['frequency_encoded_model'] = winsorize_series(df_new['frequency_encoded_model'])


# In[139]:


# Define a function to detect outliers using z-score
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    return (z_scores > threshold).any()

# Check for outliers in each column
columns_with_outliers_zscore = df_new.apply(detect_outliers_zscore)

# Print columns with outliers using z-score method
print("Columns with outliers (z-score):")
print(columns_with_outliers_zscore)


# In[140]:


df_new.head()


# In[144]:


# Check for null values in all columns
null_values_columns = df_new.isnull().sum()

# Print columns with null values and their respective count of null values
print("Columns with null values and their counts:")
for column, null_count in null_values_columns.items():
    if null_count > 0:
        print(f"{column}: {null_count}")


# In[145]:


df_new['frequency_encoded_model'].fillna(df_new['frequency_encoded_model'].mean(), inplace=True)
df_new['frequency_encoded_trim'].fillna(df_new['frequency_encoded_trim'].median(), inplace=True)
df_new['frequency_encoded_body'].fillna(df_new['frequency_encoded_body'].median(), inplace=True)
df_new['frequency_encoded_color'].fillna(df_new['frequency_encoded_color'].mean(), inplace=True)
df_new['frequency_encoded_interior'].fillna(df_new['frequency_encoded_interior'].median(), inplace=True)
df_new['frequency_encoded_make'].fillna(df_new['frequency_encoded_make'].mean(), inplace=True)


# In[146]:


# Check for null values in all columns
null_values_columns = df_new.isnull().sum()

# Print columns with null values and their respective count of null values
print("Columns with null values and their counts:")
for column, null_count in null_values_columns.items():
    if null_count > 0:
        print(f"{column}: {null_count}")


# In[142]:


# Define a function to detect outliers using z-score
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    return (z_scores > threshold).any()

# Check for outliers in each column
columns_with_outliers_zscore = df_new.apply(detect_outliers_zscore)

# Print columns with outliers using z-score method
print("Columns with outliers (z-score):")
print(columns_with_outliers_zscore)


# In[147]:


df_new.head()


# In[148]:


df_new.info()


# In[151]:


df_new['year'] = pd.to_datetime(df_new['year'])

# Extract year from the datetime column and create a new column with just the year
df_new['year'] = df_new['year'].dt.year


# In[152]:


df_new.info()


# In[153]:


# # Separate the features and target variable
X = df_new.drop('sellingprice', axis=1)  
y = df_new['sellingprice']


# In[154]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[155]:


# Define columns to scale
columns_to_scale = ['odometer', 'mmr']

# Initialize StandardScaler
scaler = StandardScaler()

# Fit scaler on training data
scaler.fit(X_train[columns_to_scale])

# Transform both training and testing data
X_train[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])


# In[156]:


# Compute the correlation matrix
correlation_matrix = df_new.corr()

# Assuming your target variable is 'Rent', extract its correlations with other features
rent_correlations = correlation_matrix['sellingprice'].drop('sellingprice')  # Drop 'Rent' since we're interested in its correlations with other features

# Sort the correlations in descending order
sorted_correlations = rent_correlations.abs().sort_values(ascending=False)

# Print the top correlated features
print("Top correlated features with Rent:")
print(sorted_correlations)


# In[157]:


# Compute the correlation matrix
correlation_matrix = df_new.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Heatmap')
plt.show()


# In[158]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression



# Initialize the linear regression model
estimator = LinearRegression()

# Initialize RFE with the linear regression model and the desired number of features to select
num_features_to_select = 10  # Change this value based on your preference
rfe = RFE(estimator, n_features_to_select=num_features_to_select)

# Fit RFE on the training data
rfe.fit(X_train, y_train)

# Get the selected features
selected_features = X_train.columns[rfe.support_]

# Print the selected features
print("Selected features:")
print(selected_features)


# In[159]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time

start_time = time.time()

# Assuming selected_features contains the list of selected feature names
# Initialize the linear regression model
model = LinearRegression()

# Train the model using only the selected features
X_train_selected = X_train[selected_features]
model.fit(X_train_selected, y_train)

# Predict on the testing data using the selected features
X_test_selected = X_test[selected_features]
y_pred = model.predict(X_test_selected)


end_time = time.time()
total_time = end_time - start_time


# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("\nTotal time taken:", total_time, "seconds")


# In[160]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming you have already trained your model and obtained predictions
# model.predict(X_test) should return the predicted rent values

# Predictions from the model
y_pred = model.predict(X_test_selected)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

# R-squared (R²) Score
r2 = r2_score(y_test, y_pred)
print("R-squared (R²) Score:", r2)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

