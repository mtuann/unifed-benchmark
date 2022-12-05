import os
import sys
import argparse

from sklearn.utils import shuffle

from fedml.simulation.sp.classical_vertical_fl.vfl_fixture import FederatedLearningFixture
from fedml.simulation.sp.classical_vertical_fl.party_models import VFLGuestModel, VFLHostModel

from fedml.simulation.sp.classical_vertical_fl.vfl import VerticalMultiplePartyLogisticRegressionFederatedLearning
# from fedml_api.model.finance.vfl_models_standalone import LocalModel, DenseModel

from wrap.trainer.preprocessing_data import breast_load_two_party_data, default_credit_load_two_party_data, give_credit_load_two_party_data, dvisits_load_two_party_data, motor_load_two_party_data, vehicle_scale_load_two_party_data
from fedml.model.finance.vfl_models_standalone import LocalModel, DenseModel

import flbenchmark.logging
from time import sleep

import json
config = json.load(open('config.json', 'r'))

def add_args(parser):
    # Training settings

    dimension_ab = {
        'breast_vertical': (10, 20),
        'default_credit_vertical': (13, 10),
        'give_credit_vertical': (5, 5)
    }

    parser.add_argument('--outdim_a', type=int, default=dimension_ab[config["dataset"]][0], metavar='N',
                        help='output dimension of local model A')
    parser.add_argument('--outdim_b', type=int, default=dimension_ab[config["dataset"]][1], metavar='N',
                        help='output dimension of local model B')                       
    parser.add_argument('--dataset', type=str, default=config["dataset"], metavar='N',
                        help='dataset used for training')
    parser.add_argument('--batch_size', type=int, default=config["training_param"]["batch_size"], metavar='N',
                        help='input batch size for training (default: 32)')  
    parser.add_argument('--lr', type=float, default=config["training_param"]["learning_rate"], metavar='LR',
                        help='learning rate (default: 0.001)')            
    parser.add_argument('--epochs', type=int, default=config["training_param"]["epochs"], metavar='EP',
                        help='how many epochs will be trained locally (default 40)')

    return parser

def load_data(dataset_name):
    if dataset_name == 'breast_vertical':
        data_dir = "../csv_data/breast_vertical/breast_hetero_"
        train, test = breast_load_two_party_data(data_dir)
    elif dataset_name == 'default_credit_vertical':
        data_dir = "../csv_data/default_credit_vertical/default_credit_hetero_"
        train, test = default_credit_load_two_party_data(data_dir)
    elif dataset_name == 'give_credit_vertical':
        data_dir = "../csv_data/give_credit_vertical/give_credit_hetero_"
        train, test = give_credit_load_two_party_data(data_dir)
    return train, test

def run_experiment(train_data, test_data, batch_size, learning_rate, epoch, args):
    print("hyper-parameters:")
    print("batch size: {0}".format(batch_size))
    print("learning rate: {0}".format(learning_rate))

    Xa_train, Xb_train, y_train = train_data
    Xa_test, Xb_test, y_test = test_data


    print("################################ Wire Federated Models ############################")

    party_a_local_model = LocalModel(input_dim=Xa_train.shape[1], output_dim=args.outdim_a, learning_rate=learning_rate, optim_param = config["training_param"]["optimizer_param"])
    party_b_local_model = LocalModel(input_dim=Xb_train.shape[1], output_dim=args.outdim_b, learning_rate=learning_rate, optim_param = config["training_param"]["optimizer_param"])

    party_a_dense_model = DenseModel(party_a_local_model.get_output_dim(), 1, learning_rate=learning_rate, optim_param = config["training_param"]["optimizer_param"], bias=True)
    party_b_dense_model = DenseModel(party_b_local_model.get_output_dim(), 1, learning_rate=learning_rate, optim_param = config["training_param"]["optimizer_param"], bias=False)
    partyA = VFLGuestModel(local_model=party_a_local_model)
    partyA.set_dense_model(party_a_dense_model)
    partyB = VFLHostModel(local_model=party_b_local_model)
    partyB.set_dense_model(party_b_dense_model)

    party_B_id = "B"
    federatedLearning = VerticalMultiplePartyLogisticRegressionFederatedLearning(partyA)
    federatedLearning.add_party(id=party_B_id, party_model=partyB)
    federatedLearning.set_debug(is_debug=False)

    print("################################ Train Federated Models ############################")

    fl_fixture = FederatedLearningFixture(federatedLearning)

    train_data = {federatedLearning.get_main_party_id(): {"X": Xa_train, "Y": y_train},
                  "party_list": {party_B_id: Xb_train}}
    test_data = {federatedLearning.get_main_party_id(): {"X": Xa_test, "Y": y_test},
                 "party_list": {party_B_id: Xb_test}}

    print(epoch, batch_size)
    fl_fixture.fit(train_data=train_data, test_data=test_data, epochs=epoch, batch_size=batch_size)


if __name__ == '__main__':
    parser = add_args(argparse.ArgumentParser(description='classical_vertical-standalone'))
    args = parser.parse_args()

    train, test = load_data(args.dataset)
    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    batch_size = args.batch_size
    epoch = args.epochs
    lr = args.lr

    Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
    Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
    train = [Xa_train, Xb_train, y_train] 
    test = [Xa_test, Xb_test, y_test]
    run_experiment(train_data=train, test_data=test, batch_size=batch_size, learning_rate=lr, epoch=epoch, args = args)