import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def breast_load_two_party_data(data_dir):
    print("[INFO] load two party data")
    Guest_data, Host_data = np.array(pd.read_csv(data_dir + 'guest.csv')), np.array(pd.read_csv(data_dir + 'host.csv'))
    Xa, Xb, y = Guest_data[:,2:], Host_data[:,1:], Guest_data[:,1]

    y = np.expand_dims(y, axis=1)
    print("# of train samples:", )
    Xa_train, Xb_train = Xa[:], Xb[:]
    Xa_test, Xb_test = Xa[:], Xb[:]
    y_train, y_test = y[:], y[:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))
    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]

def default_credit_load_two_party_data(data_dir):
    print("[INFO] load two party data")
    Guest_data, Host_data = np.array(pd.read_csv(data_dir + 'guest.csv')), np.array(pd.read_csv(data_dir + 'host.csv'))
    Xa, Xb, y = Guest_data[:,2:], Host_data[:,1:], Guest_data[:,1]

    y = np.expand_dims(y, axis=1)
    print("# of train samples:", )
    Xa_train, Xb_train = Xa[:], Xb[:]
    Xa_test, Xb_test = Xa[:], Xb[:]
    y_train, y_test = y[:], y[:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))
    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]

def give_credit_load_two_party_data(data_dir):
    print("[INFO] load two party data")
    Guest_data, Host_data = np.array(pd.read_csv(data_dir + 'guest.csv')), np.array(pd.read_csv(data_dir + 'host.csv'))
    Xa, Xb, y = Guest_data[:,2:], Host_data[:,1:], Guest_data[:,1]

    y = np.expand_dims(y, axis=1)
    print("# of train samples:", )
    Xa_train, Xb_train = Xa[:], Xb[:120000]
    Xa_test, Xb_test = Xa[:], Xb[:]
    if not (Xa_test.shape[0] == Xb_test.shape[0]) :
        maxl = min(Xa_test.shape[0],Xb_test.shape[0])
        Xa_test, Xb_test = Xa_test[:maxl], Xb_test[:maxl]

    y_train, y_test = y[:], y[:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))
    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]

def dvisits_load_two_party_data(data_dir):
    pass 
def motor_load_two_party_data(data_dir):
    pass 
def vehicle_scale_load_two_party_data(data_dir,which_to_be_1):
    print("[INFO] load two party data")
    Guest_data, Host_data = np.array(pd.read_csv(data_dir + 'guest.csv')), np.array(pd.read_csv(data_dir + 'host.csv'))
    Xa, Xb, y = Guest_data[:,2:], Host_data[:,1:], Guest_data[:,1]

    y = np.expand_dims(y, axis=1)
    y = np.array((y == which_to_be_1)).reshape(-1,1)

    print("# of train samples:", )
    Xa_train, Xb_train = Xa[:], Xb[:]
    Xa_test, Xb_test = Xa[:], Xb[:]
    if not (Xa_test.shape[0] == Xb_test.shape[0]) :
        maxl = min(Xa_test.shape[0],Xb_test.shape[0])
        Xa_test, Xb_test = Xa_test[:maxl], Xb_test[:maxl]
        
    y_train, y_test = y[:], y[:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))
    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]

