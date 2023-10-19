'''
`airbnb_preperation` prepares the airbnb dataset for training
@authors: Adam Koller
'''


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def airbnb_preperation(data_directory):
    '''This function prepares the dataset for training'''

    # Merging multiple datasets and adding city and listing_days features
    df = pd.DataFrame()
    for filename in os.listdir(data_directory):
        newdf = pd.read_csv(f'{data_directory}/{filename}')
        newdf['city'] = filename.split('_')[0]
        newdf['listing_days'] = filename.split('_')[1].split('.')[0]
        df = pd.concat([df, newdf ], ignore_index=True, axis=0)

    # Removing uninformative columns
    df = df.drop(['Unnamed: 0'], axis=1)

    # Building train and test sets
    X = df.drop('realSum', axis=1)
    y = df.realSum.copy()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=42)

    # Dropping outliers in X
    Xtrain_num = Xtrain.select_dtypes(include=[np.number])
    isolation_forest = IsolationForest(random_state=42, contamination=0.01).fit(Xtrain_num)
    outlier_indices = isolation_forest.predict(Xtrain_num)
    
    num_entries = Xtrain.shape[0]
    
    Xtrain = Xtrain.iloc[outlier_indices == 1]
    ytrain = ytrain.iloc[outlier_indices == 1]

    print(f'Dropped {num_entries - Xtrain.shape[0]} outliers')

    # Converting features to logarithms
    to_log_features = ['dist','metro_dist','attr_index','attr_index_norm','rest_index','rest_index_norm']
    for feature in to_log_features:
        Xtrain[feature] = np.log10(Xtrain[feature])

    # Log transforming target variable
    ytrain = np.log10(ytrain)
    ytest = np.log10(ytest)

    # One hot encoding categorical variables
    Xtrain = pd.get_dummies(Xtrain, columns = ['room_type','city','listing_days'])
    Xtest = pd.get_dummies(Xtest, columns = ['room_type','city','listing_days'])

    # Converting boolean values to 1 or 0 
    boolean_features = Xtrain.select_dtypes(include=[np.bool_]).columns
    for feature in boolean_features:
        Xtrain[feature] = Xtrain[feature]*1
        Xtest[feature] = Xtest[feature]*1

    # Standard scaling
    scaler = StandardScaler()
    Xtrain = pd.DataFrame(scaler.fit_transform(Xtrain), index = Xtrain.index, columns = Xtrain.columns)
    Xtest = pd.DataFrame(scaler.fit_transform(Xtest), index = Xtest.index, columns = Xtest.columns)

    # Splitting into train/dev sets
    Xtrain, Xdev, ytrain, ydev = train_test_split(Xtrain,ytrain, test_size=0.2)

    return Xtrain, ytrain, Xdev, ydev, Xtest, ytest

if __name__ == '__main__':
    Xtrain, ytrain, Xdev, ydev, Xtest, ytest = airbnb_preperation('./Data')
    #print(Xtrain.tail())
    #print(ytrain.tail())
    #print(Xtest.tail())
    #print(ytest.tail())
