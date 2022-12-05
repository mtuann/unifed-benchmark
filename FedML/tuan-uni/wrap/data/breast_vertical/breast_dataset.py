import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pandas as pd


def breast_load_two_party_data(data_dir):
    print("[INFO] load two party data")
    # processed_loan_df = load_processed_data(data_dir)
    # party_a_feat_list = qualification_feat + loan_feat
    # party_b_feat_list = debt_feat + repayment_feat + multi_acc_feat + mal_behavior_feat
    Guest_data, Host_data = np.array(pd.read_csv(data_dir + 'guest.csv')), np.array(pd.read_csv(data_dir + 'host.csv'))
    Xa, Xb, y = Guest_data[:,2:], Host_data[:,1:], Guest_data[:,1]

    y = np.expand_dims(y, axis=1)
    n_train = int(0.75 * Xa.shape[0])
    print("# of train samples:", n_train)
    Xa_train, Xb_train = Xa[:n_train], Xb[:n_train]
    Xa_test, Xb_test = Xa[n_train:], Xb[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))
    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]

# if __name__ == '__main__':
#     data_dir = "../../../data/lending_club_loan/"
#     loan_load_two_party_data(data_dir)
