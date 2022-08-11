import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf

from sklearn.metrics import r2_score

data = pd.read_csv('startup_funding.csv')

def preprocess_inputs(df):
    df = df.copy()
    
    # Drop ID and high-cardinality columns
    df = df.drop(['SNo', 'StartupName', 'SubVertical', 'InvestorsName'], axis=1)
    
    # Clean \\xc2\\xa0 examples
    df = df.applymap(lambda x: x.replace(r'\\xc2\\xa0', '') if type(x) == str else x)
    
    # Clean target column
    df['AmountInUSD'] = df['AmountInUSD'].apply(lambda x: x.replace(',', '') if str(x) != 'nan' else x)
    df['AmountInUSD'] = df['AmountInUSD'].replace({
        'undisclosed': np.NaN,
        'unknown': np.NaN,
        'Undisclosed': np.NaN,
        'N/A': np.NaN,
        '14342000+': '14342000'
    })
    
    # Drop missing target rows
    missing_target_rows = df[df['AmountInUSD'].isna()].index
    df = df.drop(missing_target_rows, axis=0).reset_index(drop=True)
    
    # Drop columns with more than 25% missing values
    df = df.drop('Remarks', axis=1)
    
    # Fill categorical missing values with most frequent occurence
    for column in ['IndustryVertical', 'CityLocation', 'InvestmentType']:
        df[column] = df[column].fillna(df[column].mode()[0])
    
    # Clean date column
    df['Date'] = df['Date'].replace({
        '05/072018': '05/07/2018',
        '01/07/015': '01/07/2015',
        '22/01//2015': '22/01/2015'
    })
    
    # Extract date features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['Day'] = df['Date'].apply(lambda x: x.day)
    df = df.drop('Date', axis=1)
    
    # Convert target column to float
    df['AmountInUSD'] = df['AmountInUSD'].astype(float)
    
    # Split df into X and y
    y = df['AmountInUSD']
    X = df.drop('AmountInUSD', axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)

def build_model():
    inputs = tf.keras.Input(shape=(422,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    
    return model

nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('nominal', nominal_transformer, ['IndustryVertical', 'CityLocation', 'InvestmentType'])
], remainder='passthrough')

regressor = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', regressor)
])


model.fit(
    X_train,
    y_train,
    regressor__validation_split=0.2,
    regressor__batch_size=32,
    regressor__epochs=100,
    regressor__callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=1,
            restore_best_weights=True
        )
    ]
)

y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print("     Test RMSE: {:.2f}".format(rmse))

r2 = r2_score(y_test, y_pred)
print("Test R^2 Score: {:.5f}".format(r2))