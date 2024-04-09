#!/usr/bin/env python
# coding: utf-8

# In[482]:


get_ipython().system('pip install xgboost')


# In[483]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


# In[484]:


df = pd.read_csv('F:\Projects\Datasets\Energy Data\PJME_hourly.csv')


# In[485]:


df.head()


# In[486]:


df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)


# In[487]:


df.head()


# In[488]:


df.tail()


# In[489]:


df.columns


# In[490]:


color_pal = sns.color_palette()


# In[491]:


df.plot(style='.', figsize=(15,5), color=color_pal[0], title='Energry Consumption')
plt.show()


# In[492]:


df['PJME_MW'].plot(kind='hist', bins=500)


# In[493]:


df.query('PJME_MW < 19000').plot(figsize=(15,5), style='.')


# In[494]:


df = df.query('PJME_MW > 19000').copy()


# In[495]:


#Train/Test Split

train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']


# In[496]:


fig, ax = plt.subplots(figsize=(15,5))
train.plot(ax=ax, label='Training Set', title='Training/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()


# In[497]:


#Time Series Cross Validation


# In[498]:


from sklearn.model_selection import TimeSeriesSplit


# In[499]:


tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()


# In[500]:


fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

fold = 0
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]
    train['PJME_MW'].plot(ax=axs[fold],
                          label='Training Set',
                          title=f'Data Train/Test Split Fold {fold}')
    test['PJME_MW'].plot(ax=axs[fold],
                         label='Test Set')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    fold += 1
plt.show()


# In[501]:


(df.index > '01-01-2010') & (df.index < '01-07-2010')


# In[502]:


df.loc[(df.index > '01-01-2010') & (df.index < '01-07-2010')].plot(figsize=(15,5), title='Week of Data')


# In[503]:


#Feature Creation


# In[504]:


def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)


# In[505]:


def add_lags(df):
    target_map = df['PJME_MW'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df


# In[506]:


df = add_lags(df)


# In[507]:


df.head()


# In[508]:


fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]
    
    train = create_features(train)
    test = create_features(test)
    
    Feature_columns = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1','lag2','lag3'] 
    Target_column = ['PJME_MW']
    
    X_train = train[Feature_columns]
    y_train = train[Target_column]

    X_test = test[Feature_columns]
    y_test = test[Target_column]
    
    model = xgb.XGBRegressor(base_score=0.5, 
                             booster='gbtree',
                             n_estimators=1000,
                             early_stopping_rounds=50,
                             objective='reg:squarederror',
                             max_depth=3,
                             learning_rate=0.001)
    model.fit(X_train, y_train,
         eval_set=[(X_train, y_train), (X_test, y_test)],
         verbose=100)
    
    y_pred = model.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)


# In[509]:


print(f'Avg score across folds {np.mean(scores):0.2f}')
print(f'Fold scores: {scores}')


# In[510]:


#Retrain the model with all the data


# In[511]:


# Retrain on all data
df = create_features(df)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
            'lag1','lag2','lag3']
TARGET = 'PJME_MW'

X_all = df[FEATURES]
y_all = df[TARGET]

model = xgb.XGBRegressor(base_score=0.5,
                       booster='gbtree',    
                       n_estimators=1000,
                       objective='reg:squarederror',
                       max_depth=3,
                       learning_rate=0.01)
model.fit(X_all, y_all,
        eval_set=[(X_all, y_all)],
        verbose=100)


# In[512]:


df.index.max()


# In[513]:


# Create future dataframe
future = pd.date_range('2018-08-03', '2019-08-01', freq='1h')
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True
df['isFuture'] = False
df_and_future = pd.concat([df, future_df])


# In[514]:


df_and_future


# In[515]:


df_and_future = create_features(df_and_future)
df_and_future = add_lags(df_and_future)


# In[516]:


future_w_features = df_and_future.query('isFuture').copy()


# In[517]:


future_w_features


# In[535]:


# Predict the future


# In[536]:


Feature_columns


# In[537]:


FEATURES


# In[540]:


future_w_features['pred'] = model.predict(future_w_features[FEATURES])


# In[541]:


future_w_features['pred'].plot(figsize=(10, 5),
                               color=color_pal[4],
                               ms=1,
                               lw=1,
                               title='Future Predictions')


# In[ ]:





# In[ ]:





# In[ ]:





# In[542]:


#Visualise Feature/Target Relationship


# In[543]:


fig, ax= plt.subplots(figsize=(10,8)) 
sns.boxplot(data=df, x='hour', y = 'PJME_MW' )
ax.set_title('MW by Hour')


# In[544]:


fig, ax= plt.subplots(figsize=(10,8)) 
sns.boxplot(data=df, x='month', y = 'PJME_MW', palette='Blues' )
ax.set_title('MW by Month')


# In[545]:


#Create the Model


# In[546]:


#Feature Importance


# In[547]:


fi = pd.DataFrame(data=model.feature_importances_,
             index=model.feature_names_in_,
            columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')


# In[548]:


#Forecast on Test


# In[551]:


ax = df[['PJME_MW']].plot(figsize=(15,5))
future_w_features['pred'].plot(ax=ax, style='.')
ax.set_title('Raw Data and Predictions')
plt.legend(['True Data', 'Model Predictions'])


# In[553]:


df.loc[(df.index > '04-01-2018') & 
       (df.index < '04-08-2018')].plot(figsize=(15,5), 
                                       title='Week of Data')

