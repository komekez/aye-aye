
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

def lstm_classifier(MAX_NB_WORDS, EMBEDDING_DIM, input_length):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=input_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
