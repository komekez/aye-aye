from util.data_processing import * 
import pandas as pd
import warnings as wrn
wrn.filterwarnings('ignore')

from models.lstm_classifier import lstm_classifier

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
epochs=2
batch_size=64

def processed_data():
    fetched_data = fetch_data()
    parameters = ['utterance_index', 'subutterance_index', 'text', 'act_tag']
    removed_columns = remove_unwanted_params(fetched_data, parameters)

    backchannel_classify = mark_backchannels(removed_columns)
    
    swda_df = preprocess_data(backchannel_classify)

    X, y = text_label_gen(swda_df)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.10, random_state = 42)

    return X_train, X_test, y_train, y_test, X.shape[1]


if __name__ == "__main__":
    model_type = 'lstm'

    X_train, X_test, y_train, y_test, X_shape = processed_data()
    print('here')
    if(model_type == 'lstm'):
        lstm_classifier_model = lstm_classifier(MAX_NB_WORDS, EMBEDDING_DIM, X_shape)


        lstm_classifier_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        accr = lstm_classifier_model.evaluate(X_test, y_test)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
