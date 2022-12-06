import logging
from time import time
import flbenchmark.logging

class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.time = time()

        self.logger = flbenchmark.logging.BasicLogger(id=client_idx + 1, agent_type='client')
        print(client_idx)
        

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    def getbyte(self, w):
        ret = 0
        for key, value in w.items():
            ret += value.numel() * value.element_size()
        return ret

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        # self.model_trainer.train(self.local_training_data, self.device, self.args)
        self.model_trainer.train(self.local_training_data, self.device, self.args, self.logger)
        weights = self.model_trainer.get_model_params()
        with self.logger.communication(target_id=0) as c:
            c.report_metric('byte', self.getbyte(weights)) 
        self.time = time()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
