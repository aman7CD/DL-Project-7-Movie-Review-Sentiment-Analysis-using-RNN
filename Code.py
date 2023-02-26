
#importing the dependencies
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import Sequential
from keras.layers import Dense,SimpleRNN,Embedding,Flatten

(X_train,y_train),(X_test,y_test) = imdb.load_data()

X_train = pad_sequences(X_train,padding='post',maxlen=50)
X_test = pad_sequences(X_test,padding='post',maxlen=50)

X_train.shape

model = Sequential()
model.add(Embedding(10000, 2, input_length=50))
model.add(SimpleRNN(32,return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train,epochs=5,validation_data=(X_test,y_test))

loss, accuracy = model.evaluate(X_test, y_test)

#Prediction
input_data = X_test[4]
input_data = input_data.reshape(1,-1)

prediction = model.predict(input_data)

if(prediction[0] < 0.5):
  print('The Movie is Bad')

else:
  print('The Movie is Good')
  
