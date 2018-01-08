import librosa
import librosa.display
% pylab inline
import os
import pandas as pd
import glob 

DATA_PATH = '../Data/data_speech_commands_v0.01/'

def feature_extractor():
    X_train = []
    X_val = []
    X_test = []
    
    Y_train = []
    Y_val = []
    Y_test = []
    
    convert_label_id = {}
    
    labels = next(os.walk(DATA_PATH))[1]
    
    for label_id in xrange(len(labels)):
        convert_label_id[labels[label_id]] = label_id 
    
    with open(DATA_PATH+"training_list.txt") as f:
        train_list = f.readlines()
        train_list = [ele.split('\n')[0] for ele in train_list]
    with open(DATA_PATH+"validation_list.txt") as f:
        val_list = f.readlines()
        val_list = [ele.split('\n')[0] for ele in val_list]
    with open(DATA_PATH+"testing_list.txt") as f:
        test_list = f.readlines()
        test_list = [ele.split('\n')[0] for ele in test_list]
    
    train_list = list(set(train_list) - set(val_list + test_list))
    
    ##Example - Limiting dataset
    train_list = train_list[:10]
    val_list = val_list[:10]
    test_list = test_list[:10]
     
    for audio_file in train_list:     
        data, sampling_rate = librosa.load(DATA_PATH+audio_file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T,axis=0)
        X_train.append(mfccs)
        Y_train.append(convert_label_id[audio_file.split('/')[0]])

    for audio_file in val_list:     
        data, sampling_rate = librosa.load(DATA_PATH+audio_file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T,axis=0)
        X_val.append(mfccs)
        Y_val.append(convert_label_id[audio_file.split('/')[0]])
        
    for audio_file in test_list:     
        data, sampling_rate = librosa.load(DATA_PATH+audio_file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T,axis=0)
        X_test.append(mfccs)
        Y_test.append(convert_label_id[audio_file.split('/')[0]])        
    
    split_data = [np.array(X_train),np.array(Y_train),np.array(X_val),np.array(Y_val),np.array(X_test),np.array(Y_test)]
    return [split_data,labels]   


##Visualise
#plt.figure(figsize=(12, 4))
#librosa.display.waveplot(data, sr=sampling_rate)

##Feature Extraction
[[X_train, Y_train, X_val, Y_val, X_test, Y_test],labels] = feature_extractor()
