import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import json

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from fedml_api.data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
from fedml_api.data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.cv.cnn import CNN_DropOut, CNN_OriginalFedAvg
from fedml_api.model.cv.lenet import lenet
from fedml_api.model.cv.alexnet import AlexNet
from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow

from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from fedml_api.data_preprocessing.breast_horizontal.data_loader import load_partition_data_breast_horizontal
from fedml_api.data_preprocessing.default_credit_horizontal.data_loader import load_partition_data_default_credit_horizontal
from fedml_api.data_preprocessing.give_credit_horizontal.data_loader import load_partition_data_give_credit_horizontal
from fedml_api.data_preprocessing.vehicle_scale_horizontal.data_loader import load_partition_data_vehicle_scale_horizontal
from fedml_api.data_preprocessing.student_horizontal.data_loader import load_partition_data_student_horizontal
from fedml_api.model.linear.lr import LogisticRegression, LinearRegression
from fedml_api.model.non_linear.mlp import MLP
from fedml_api.model.cv.resnet_gn import resnet18

from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from fedml_api.standalone.fedavg.my_model_trainer_regression import MyModelTrainer as MyModelTrainerRGR
from fedml_api.standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from fedml_api.standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG

from fedml_api.data_preprocessing.our_femnist.data_loader import load_partition_data_femnist
from fedml_api.data_preprocessing.our_reddit.data_loader import load_partition_data_reddit
from fedml_api.data_preprocessing.our_celeba.data_loader import load_partition_data_celeba

config = json.load(open('config.json', 'r'))

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default=config["model"], metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default=config["dataset"], metavar='N',
                        help='dataset used for training')

    if config["dataset"] == "mnist" :
        parser.add_argument('--data_dir', type=str, default='./data/mnist',
                            help='data directory')
    elif config["dataset"] == "cifar10" :
        parser.add_argument('--data_dir', type=str, default='./data/cifar10',
                            help='data directory')
    elif config["dataset"] == "shakespeare" :
        parser.add_argument('--data_dir', type=str, default='./data/shakespeare',
                            help='data directory')
    else:
        parser.add_argument('--data_dir', type=str, default='../data/'+config["dataset"],
                            help='data directory')


    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=config["training_param"]["batch_size"], metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default=config["training_param"]["optimizer"],
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=config["training_param"]["learning_rate"], metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=config["training_param"]["optimizer_param"]["weight_decay"])

    parser.add_argument('--epochs', type=int, default=config["training_param"]["inner_step"], metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=2, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=config["training_param"]["client_per_round"], metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=config["training_param"]["epochs"],
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--optim_param', type=dict, default=config["training_param"]["optimizer_param"])

    return parser


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size)
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num

    elif dataset_name == "breast_horizontal":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_breast_horizontal(args.batch_size)
        args.client_num_in_total = client_num
    
    elif dataset_name == "default_credit_horizontal":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_default_credit_horizontal(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "give_credit_horizontal":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_give_credit_horizontal(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "vehicle_scale_horizontal":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_vehicle_scale_horizontal(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "student_horizontal":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_student_horizontal(args.batch_size)
        args.client_num_in_total = client_num
        
    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_femnist(args.batch_size)
        print('CLASS NUM = ',class_num)
        args.client_num_in_total = client_num

    elif dataset_name == "reddit":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_reddit(args.batch_size)
        print('CLASS NUM = ',class_num)
        args.client_num_in_total = client_num

    elif dataset_name == "celeba":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        logging.info("Begin to load Celeba")
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_celeba(args.batch_size)
        print('CLASS NUM = ',class_num)
        args.client_num_in_total = client_num
        logging.info("END loading Celeba")

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "ILSVRC2012":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_ImageNet(dataset=dataset_name, data_dir=args.data_dir,
                                                 partition_method=None, partition_alpha=None,
                                                 client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_dir, 'mini_gld_train_split.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'mini_gld_test.csv')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_dir, 'federated_train.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'test.csv')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)

    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size)

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None

    if model_name[:3] == 'mlp':
        hidden = list(model_name.split('_'))[1:]
        model_name = 'mlp'
        hidden = [int(x) for x in hidden]

    # print(hidden)

    if model_name == "logistic_regression" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "mlp" and args.dataset == 'mnist':
        logging.info("MLP + mnist")
        model = MLP(28 * 28, output_dim, hidden)
    elif model_name == "logistic_regression" and args.dataset == 'breast_horizontal':
        logging.info("LogisticRegression + breast_horizontal")
        model = LogisticRegression(30, output_dim)
    elif model_name == "logistic_regression" and args.dataset == 'default_credit_horizontal':
        logging.info("LogisticRegression + default_credit_horizontal")
        model = LogisticRegression(23, output_dim)
    elif model_name == "mlp" and args.dataset == 'default_credit_horizontal':
        logging.info("MLP + default_credit_horizontal")
        model = MLP(23, output_dim, hidden)
    elif model_name == "logistic_regression" and args.dataset == 'give_credit_horizontal':
        logging.info("LogisticRegression + give_credit_horizontal")
        model = LogisticRegression(10, output_dim)
    elif model_name == "logistic_regression" and args.dataset == 'vehicle_scale_horizontal':
        logging.info("LogisticRegression + vehicle_scale_horizontal")
        model = LogisticRegression(18, output_dim)
    elif model_name == "mlp" and args.dataset == 'breast_horizontal':
        logging.info("MLP + breast_horizontal")
        model = MLP(30, output_dim, hidden)
    elif model_name == "mlp" and args.dataset == 'give_credit_horizontal':
        logging.info("MLP + give_credit_horizontal")
        model = MLP(10, output_dim, hidden)
    elif model_name == "mlp" and args.dataset == 'vehicle_scale_horizontal':
        logging.info("MLP + vehicle_scale_horizontal")
        model = MLP(18, output_dim, hidden)
    elif model_name == 'linear_regression' and args.dataset == 'student_horizontal':
        logging.info("LinearRegression + student_horizontal")
        model = LinearRegression(13, 1)
    elif model_name == "logistic_regression" and args.dataset == "femnist":
        logging.info("LogisticRegression + FederatedEMNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "mlp" and args.dataset == "femnist":
        logging.info("MLP + FederatedEMNIST")
        model = MLP(28 * 28, output_dim, hidden)
    elif model_name == "lenet" and args.dataset == "femnist":
        logging.info("LeNet5 + FederatedEMNIST")
        model = lenet(output_dim)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "mlp" and args.dataset == "celeba":
        logging.info("MLP + celeba")
        model = MLP(224 * 224 * 3, output_dim, hidden)
    elif model_name == "alexnet" and args.dataset == "celeba":
        logging.info("AlexNet + celeba")
        model = AlexNet()
    elif model_name == "cnn_o" and args.dataset == "femnist":
        logging.info("CNN_ORIGINAL + FederatedEMNIST")
        model = CNN_OriginalFedAvg(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "logistic_regression" and args.dataset == "stackoverflow_logistic_regression":
        logging.info("logistic_regression + stackoverflow_logistic_regression")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "lstm" and args.dataset == "reddit":
        logging.info("LSTM + reddit")
        model = RNN_StackOverFlow(vocab_size = output_dim - 4) 
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)

    # print(model)
    return model


def custom_model_trainer(args, model):
    if args.dataset == "stackoverflow_logistic_regression":
        return MyModelTrainerTAG(model)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp", "reddit"]:
        return MyModelTrainerNWP(model)
    elif args.dataset in ['student_horizontal']:
        return MyModelTrainerRGR(model)
    else: # default model trainer is for classification problem
        return MyModelTrainerCLS(model)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    dataset = load_data(args, args.dataset)

    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model_trainer = custom_model_trainer(args, model)
    logging.info(model)

    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer, is_regression = (args.dataset == 'student_horizontal'))
    fedavgAPI.train()
