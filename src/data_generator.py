import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DataGen():
  def __init__(self, filepath, split):
    data = pd.read_csv(filepath)
    data.loc[:, 'DATETIME'] = pd.to_datetime(data.loc[:, 'DATETIME'], infer_datetime_format=True)

    mRange = int(len(data)*split[0])

    self.train = data[:mRange]
    self.test = data[mRange:]

    self.E_paras = [np.min(self.train.E.values),np.max(self.train.E.values)]
    self.Avg_paras = [np.min(self.train.Avg.values),np.max(self.train.Avg.values)]
    self.Std_paras = [np.min(self.train.Std.values),np.max(self.train.Std.values)]
    self.Avg_w_paras = [np.min(self.train.Avg_w.values),np.max(self.train.Avg_w.values)]
    self.Std_w_paras = [np.min(self.train.Std_w.values),np.max(self.train.Std_w.values)]

  def get_data(self, data_type, window, normalize_basic = True, normalize_derived = True):
    
    if(data_type=='train'):
      data = self.train

    else:
      data = self.test

    #Basic features
    E = data.E.values
    if (normalize_basic):
      E = (E - self.E_paras[0]) / (self.E_paras[1] - self.E_paras[0])
    E = E.reshape(-1, 1)

    enc = OneHotEncoder(handle_unknown='ignore')

    I = data.I.values
    enc.fit(I.reshape(-1, 1))
    I = enc.transform(I.reshape(-1, 1)).toarray()

    D = data.D.values
    enc.fit(D.reshape(-1, 1))
    D = enc.transform(D.reshape(-1, 1)).toarray()

    H = data.H.values
    enc.fit(H.reshape(-1, 1))
    H = enc.transform(H.reshape(-1, 1)).toarray()

    basic_data = np.concatenate((E, D, H, I), axis=1)

    #Derived features
    Avg = data.Avg.values
    Std = data.Std.values
    Avg_w = data.Avg_w.values
    Std_w = data.Std_w.values

    Avg[:-1] = Avg[1:]
    Std[:-1] = Std[1:]
    
    if(normalize_derived):
      Avg = (Avg - self.Avg_paras[0]) / (self.Avg_paras[1] - self.Avg_paras[0])
      Std = (Std - self.Std_paras[0]) / (self.Std_paras[1] - self.Std_paras[0])

      Avg_w = (Avg_w - self.Avg_w_paras[0]) / (self.Avg_w_paras[1] - self.Avg_w_paras[0])
      Std_w = (Std_w - self.Std_w_paras[0]) / (self.Std_w_paras[1] - self.Std_w_paras[0])
    
    Avg = Avg.reshape(-1, 1)
    Std = Std.reshape(-1, 1)
    Avg_w = Avg_w.reshape(-1, 1)
    Std_w = Std_w.reshape(-1, 1)

    derived_data = np.concatenate((E, Avg_w, Std_w, Avg, Std), axis=1)

    seq_len = window + 1
    seq_basic = []
    seq_derived = []

    for i in range(len(data) - seq_len):
      seq_basic.append(basic_data[i: i + seq_len])
      seq_derived.append(derived_data[i: i + window])
    
    seq_basic = np.asarray(seq_basic)
    seq_derived = np.asarray(seq_derived)

    x_data = seq_basic[:, :-1]
    y_data = seq_basic[:, -1][:, 0]

    if(normalize_basic):
      y_data = (y_data* (self.E_paras[1] - self.E_paras[0])) + self.E_paras[0]    
    
    return x_data, y_data, seq_derived