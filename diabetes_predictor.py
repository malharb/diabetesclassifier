import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.show()
df = pd.read_csv('diabdf.csv')

plt.figure()
sns.countplot(x='Outcome',data=df)

#plt.figure()
#sns.heatmap(pd.isnull(df),yticklabels=False) #NO NULLS

from sklearn.model_selection import train_test_split
X = df.drop('Outcome',axis=1).values
y = df['Outcome'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=14)

#Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model = Sequential()

model.add(Dense(18,input_dim=8,activation='relu'))

model.add(Dense(30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(25,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(18,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=5)

model.fit(x=X_train,y=y_train,
          epochs=1000,validation_data=[X_test,y_test],
          callbacks=[early_stopping])

predictions = model.predict_classes(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))




