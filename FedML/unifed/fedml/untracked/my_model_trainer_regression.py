import logging

import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, logger):
        model = self.model

        model.to(device)
        model.train()



        # train and update
        criterion = nn.MSELoss().to(device)
        optim_param = args.optim_param
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, momentum=optim_param['momentum'], weight_decay=optim_param['weight_decay'], dampening=optim_param['dampening'], nesterov=optim_param['nesterov'])
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(args.epochs):
            with logger.computation() as c:
                batch_loss, tot_sample = [], 0
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    model.zero_grad()
                    log_probs = model(x).reshape(-1)
                    loss = criterion(log_probs, labels)
                    loss.backward()

                    optimizer.step()
                    batch_loss.append(loss.item() * x.shape[0])
                    tot_sample += x.shape[0]
                epoch_loss.append(sum(batch_loss) / tot_sample)
                logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
                c.report_metric('loss',epoch_loss[-1] )

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            # 'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.MSELoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x).reshape(-1)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
