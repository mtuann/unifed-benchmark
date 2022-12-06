import copy
import logging
import random

import numpy as np
import torch
import wandb

from fedml import mlops
# from fedml.ml.trainer.trainer_creator import create_model_trainer
# from ...trainer.trainer_creator import create_model_trainer
from wrap.trainer.trainer_creator import custom_model_trainer
from .client import Client
import json
from sklearn.metrics import roc_auc_score

from time import time
from time import sleep
import flbenchmark.logging

AUC = ['breast_horizontal', 'default_credit_horizontal', 'give_credit_horizontal',
       'breast_vertical', 'default_credit_vertical', 'give_credit_vertical', ]
def getbyte(w):
    ret = 0
    for key, value in w.items():
        ret += value.numel() * value.element_size()
    return ret

class FedAvgAPI(object):
    def __init__(self, args, device, dataset, model, is_regression):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.is_regression = is_regression
        
        logging.info("model = {}".format(model))
        # self.model_trainer = create_model_trainer(model, args)
        self.model_trainer = custom_model_trainer(model, args)
        
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )

    def _setup_clients(
        self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")
        
    def train(self):
        communication_time, communication_bytes = 0, 0
        report_my = 0
        config = json.load(open('config.json', 'r'))

        w_global = self.model_trainer.get_model_params()
        with flbenchmark.logging.BasicLogger(id=0, agent_type='aggregator') as logger:
            with logger.training():
                for idx, client in enumerate(self.client_list):
                    # client.logger.start()
                    client.logger.training_start()

                for round_idx in range(self.args.comm_round):
                    with logger.training_round() as tr:
                        tr.report_metric('client_num', config["training_param"]["client_per_round"])
                        logging.info("################Communication round : {}".format(round_idx))
                        
                        for idx, client in enumerate(self.client_list):
                            # client.logger.start()
                            client.logger.training_round_start()

                        w_locals = []

                        """
                        for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
                        Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
                        """
                        client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                            self.args.client_num_per_round)
                        logging.info("client_indexes = " + str(client_indexes))

                        for idx, client in enumerate(self.client_list):
                            # update dataset
                            client_idx = client_indexes[idx]
                            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                                        self.test_data_local_dict[client_idx],
                                                        self.train_data_local_num_dict[client_idx])



                            # train on new dataset
                            w = client.train(copy.deepcopy(w_global))
                            communication_bytes += getbyte(w)
                            # self.logger.info("local weights = " + str(w))
                            w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                            
                            # client back to server
                            communication_time += time() - client.time

                        # update global weights
                        with logger.computation() as c:
                            w_global = self._aggregate(w_locals)

                        for idx, client in enumerate(self.client_list):
                            with logger.communication(target_id=idx + 1) as c:
                                c.report_metric('byte', getbyte(w_global))

                        timea = time()
                        self.model_trainer.set_model_params(w_global)
                        # server to client time
                        communication_time += time() - timea

                        # test results
                        # at last round
                        if round_idx == self.args.comm_round - 1:
                            btime = time()
                            report_my = self._local_test_on_all_clients(round_idx)
                            finaltime = time() - btime
                        # per {frequency_of_the_test} round
                        elif round_idx % self.args.frequency_of_the_test == 0:
                            pass # if self.args.dataset.startswith("stackoverflow"):
                                # self._local_test_on_validation_set(round_idx)
                            # else:
                                # self._local_test_on_all_clients(round_idx)
                        
                        for idx, client in enumerate(self.client_list):
                            # client.logger.start()
                            client.logger.training_round_end()

                for idx, client in enumerate(self.client_list):
                    client.logger.end()
                # print("Total communication time is {}".format(communication_time))
                # print("Total communication cost is {}".format(communication_bytes * 2))
                # print("Total communication round is {}".format(self.args.comm_round))
                for idx, client in enumerate(self.client_list):
                    # client.logger.start()
                    client.logger.training_end()

            with logger.model_evaluation() as e:
                sleep(finaltime)
                if self.is_regression:
                    e.report_metric('mse', report_my)
                elif self.args.dataset in AUC:
                    # e.report_metric('accuracy', report_my[1] * 100)
                    e.report_metric('auc', report_my)
                else:
                    e.report_metric('accuracy', report_my)

    # def train(self):
    #     logging.info("self.model_trainer = {}".format(self.model_trainer))
    #     w_global = self.model_trainer.get_model_params()
    #     mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
    #     mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
    #     mlops.log_round_info(self.args.comm_round, -1)
    #     for round_idx in range(self.args.comm_round):

    #         logging.info("################Communication round : {}".format(round_idx))

    #         w_locals = []

    #         """
    #         for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
    #         Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
    #         """
    #         client_indexes = self._client_sampling(
    #             round_idx, self.args.client_num_in_total, self.args.client_num_per_round
    #         )
    #         logging.info("client_indexes = " + str(client_indexes))

    #         for idx, client in enumerate(self.client_list):
    #             # update dataset
    #             client_idx = client_indexes[idx]
    #             client.update_local_dataset(
    #                 client_idx,
    #                 self.train_data_local_dict[client_idx],
    #                 self.test_data_local_dict[client_idx],
    #                 self.train_data_local_num_dict[client_idx],
    #             )

    #             # train on new dataset
    #             mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
    #             w = client.train(copy.deepcopy(w_global))
    #             mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
    #             # self.logging.info("local weights = " + str(w))
    #             w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

    #         # update global weights
    #         mlops.event("agg", event_started=True, event_value=str(round_idx))
    #         w_global = self._aggregate(w_locals)
    #         self.model_trainer.set_model_params(w_global)
    #         mlops.event("agg", event_started=False, event_value=str(round_idx))

    #         # test results
    #         # at last round
    #         if round_idx == self.args.comm_round - 1:
    #             self._local_test_on_all_clients(round_idx)
    #         # per {frequency_of_the_test} round
    #         elif round_idx % self.args.frequency_of_the_test == 0:
    #             if self.args.dataset.startswith("stackoverflow"):
    #                 self._local_test_on_validation_set(round_idx)
    #             else:
    #                 self._local_test_on_all_clients(round_idx)

    #         mlops.log_round_info(self.args.comm_round, round_idx)

    #     mlops.log_training_finished_status()
    #     mlops.log_aggregation_finished_status()

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        # logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _aggregate_noniid_avg(self, w_locals):
        """
        The old aggregate method will impact the model performance when it comes to Non-IID setting
        Args:
            w_locals:
        Returns:
        """
        (_, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            temp_w = []
            for (_, local_w) in w_locals:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            if not self.is_regression:
                train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))
            
            if self.args.dataset in AUC:
                if predict_my == None:
                    predict_my = test_local_metrics['predict']
                else:
                    predict_my = torch.cat((predict_my, test_local_metrics['predict']))
                if target_my == None:
                    target_my = test_local_metrics['targety']
                else:
                    target_my = torch.cat((target_my, test_local_metrics['targety']))
        # test on training dataset
        train_acc = 0
        if not self.is_regression:
            train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
            
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = 0
        if not self.is_regression:
            test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
         
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})

        # mlops.log({"Train/Acc": train_acc, "round": round_idx})
        # mlops.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

        # mlops.log({"Test/Acc": test_acc, "round": round_idx})
        # mlops.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)
        
        if self.is_regression:
            return test_loss
        elif self.args.dataset not in AUC:
            return test_acc
        else:
            return roc_auc_score(target_my.cpu(), predict_my.cpu())
        

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Pre": test_pre, "round": round_idx})
                wandb.log({"Test/Rec": test_rec, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Pre": test_pre, "round": round_idx})
            mlops.log({"Test/Rec": test_rec, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)
