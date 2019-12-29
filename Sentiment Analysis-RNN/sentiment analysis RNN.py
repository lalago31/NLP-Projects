from keras.datasets import imdb  # import the built-in imdb dataset in Keras
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,Bidirectional

# load data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

print("Loaded dataset with {} training samples, {} test samples".format(len(X_train), len(X_test)))
# Map word IDs back to words
word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print("--- Review (with words) ---")
print([id2word.get(i, " ") for i in X_train[7]])
print("--- Label ---")
print(y_train[7])


# Set the maximum number of words per document (for both training and testing)
max_words = 500
# Pad sequences in X_train and X_test
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# RNN model architecture
embedding_size = 32
vocabulary_size = 5000

model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length = max_words))
model.add(Bidirectional(LSTM(50, dropout=0.3, return_sequences=False), merge_mode='concat'))
# model.add(LSTM(64, dropout=0.3, return_sequences=False))
# model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
batch_size = 256
num_epochs = 10

X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]  # first batch_size samples
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]  # rest for training

model.fit(X_train, y_train,
          validation_data=(X_valid, y_valid),
          batch_size=batch_size, epochs=num_epochs)

# Evaluate model on the test set
scores = model.evaluate(X_test, y_test, verbose=0)  # returns loss and other metrics specified in model.compile()
print("Test accuracy:", scores[1])  # scores[1] correspond to accuracy with metrics=['accuracy']