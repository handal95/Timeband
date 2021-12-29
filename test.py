import enum
import io
import os
import time
import urllib
import zipfile
import datetime

DATA_URL = 'https://s3-us-west-2.amazonaws.com/telemanom/data.zip'

if not os.path.exists('./smap'):
    response = urllib.request.urlopen(DATA_URL)
    bytes_io = io.BytesIO(response.read())
    
    with zipfile.ZipFile(bytes_io) as zf:
        zf.extractall()
        
train_signals = os.listdir('./smap/train')
test_signals = os.listdir('./smap/test')

print(train_signals == test_signals)

import pandas as pd
import numpy as np

NASA_DIR = os.path.join('smap', '{}', '{}')

def build_df(data, start=0):
    index = np.array(range(start, start + len(data)))
    timestamp = index * 21600 + 1222819200
    
    return pd.DataFrame({'timestamp': timestamp, 'value': data[:, 0]}) #, 'value1': data[:, 1], 'value2': data[:, 2]})

npy_file = NASA_DIR.format('train', 'S-1.npy')
data = build_df(np.load(npy_file))

print(data.head())

os.makedirs('csv', exist_ok=True)
PATH_DIR = os.path.join('csv', '{}')

for signal in train_signals:
    name = signal[:-4]
    train_np = np.load(NASA_DIR.format('train', signal))
    test_np = np.load(NASA_DIR.format('test', signal))
    
    data = build_df(np.concatenate([train_np, test_np]))
    data.to_csv(PATH_DIR.format(name + '.csv'), index=False)
    
    train = build_df(train_np)
    train.to_csv(PATH_DIR.format(name + '-train.csv'), index=False)
    
    test = build_df(test_np, start=len(train))
    test.to_csv(PATH_DIR.format(name + '-test.csv'), index=False)
    
s1 = pd.read_csv(PATH_DIR.format('S-1.csv'))
print(s1.head())


import os
import json
import pandas as pd

CSV_URL = 'https://github.com/khundman/telemanom/raw/master/labeled_anomalies.csv'

df = pd.read_csv(CSV_URL)
df.head()

import os
import json

labels_data = list()

def format_csv(df, timestamp_column=None, value_columns=None):
    timestamp_column_name = df.columns[timestamp_column] if timestamp_column else df.columns[0]
    value_column_names = df.columns[value_columns] if value_columns else df.columns[1:]

    data = dict()
    data['timestamp'] = df[timestamp_column_name].astype('int64').values
    for column in value_column_names:
        data[column] = df[column].astype(float).values

    return pd.DataFrame(data)

def load_csv(path, timestamp_column=None, value_column=None):
    header = None if timestamp_column is not None else 'infer'
    data = pd.read_csv(path, header=header)

    if timestamp_column is None:
        return data

    return format_csv(data, timestamp_column, value_column)

def load_signal(signal, test_size=None, timestamp_column=None, value_column=None):
    data = load_csv(signal, timestamp_column, value_column)
    data = format_csv(data)

    if test_size is None:
        return data

    test_length = round(len(data) * test_size)
    train = data.iloc[:-test_length]
    test = data.iloc[-test_length:]

    return train, test

for _, row in df.iterrows():
    signal = row.chan_id
    data = load_signal(os.path.join('csv', signal + '.csv'))
    test = data[-row.num_values:]
    
    events = list()
    for start, end in json.loads(row.anomaly_sequences):
        start_ts = test.iloc[start].timestamp.astype(int)
        end_ts = test.iloc[end].timestamp.astype(int)
        events.append([start_ts, end_ts])
    
    labels_data.append({
        'signal': signal,
        'events': events
    })
    
labels = pd.DataFrame(labels_data)[['signal','events']]
labels.set_index(['signal'], inplace=True)
labels.sort_index(inplace=True)
print(labels.head())
labels.reset_index(inplace=True)
labels.to_csv('labels.csv', index=False)


print("~~~~~~")
for _, row in df.iterrows():
    print(">>>", signal)
    signal = row.chan_id
    
    data = pd.read_csv(os.path.join('csv', signal + '.csv'))
    try:
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    except:
        pass
    
    data.to_csv(os.path.join('data', signal + '.csv'), index=False)
    

labels = pd.DataFrame(labels_data)[['signal','events']]
for i, signal in enumerate(labels['signal']):
    print(signal)
    
    events = labels["events"].iloc[i]
    
    timestamp_events = list()
    for j, event in enumerate(events):
        start, end = event
        start = datetime.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S")
        end = datetime.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S")
        
        print(start, end)
        timestamp_events.append([start, end])
    print(timestamp_events)

    labels["events"].iloc[i] = timestamp_events    
    # for t in labels['events'].iloc[i]:
    #     labels['events'].iloc[i][0] = pd.to_datetime(t[0], unit='s')
    #     labels['events'].iloc[i][1] = pd.to_datetime(t[1], unit='s')

labels.to_csv("labels.csv", index=False)
