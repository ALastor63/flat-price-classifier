# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

df = pd.read_csv('flats_big.csv')
df = df[['price', 'area', 'rooms', 'floor']].dropna()

q1, q2 = df['price'].quantile([0.33, 0.66])

def price_to_class(p):
    if p <= q1:  return 0    
    if p <= q2:  return 1   
    return 2                 

df['price_class'] = df['price'].apply(price_to_class)

X = df[['area', 'rooms', 'floor']].values
y = df['price_class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Точность: {acc:.2f}')