from util.data_processing import * 
import pandas as pd
import warnings as wrn
wrn.filterwarnings('ignore')


def processed_data():
    fetched_data = fetch_data()
    parameters = ['utterance_index', 'subutterance_index', 'text', 'act_tag']
    removed_columns = remove_unwanted_params(fetched_data, parameters)

    backchannel_classify = mark_backchannels(removed_columns)
    
    swda_df = preprocess_data(backchannel_classify)
    print(swda_df)
    return swda_df


processed_data()