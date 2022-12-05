import flbenchmark.logging

class VerticalMultiplePartyLogisticRegressionFederatedLearning(object):

    def __init__(self, party_A, main_party_id="_main"):
        super(VerticalMultiplePartyLogisticRegressionFederatedLearning, self).__init__()
        self.main_party_id = main_party_id
        # party A is the parity with labels
        self.party_a = party_A
        # the party dictionary stores other parties that have no labels
        self.party_dict = dict()
        self.is_debug = False
    
    def set_logger(self, logger0, logger1):
        self.logger0 = logger0
        self.logger1 = logger1

    def set_debug(self, is_debug):
        self.is_debug = is_debug

    def get_main_party_id(self):
        return self.main_party_id

    def add_party(self, *, id, party_model):
        self.party_dict[id] = party_model

    def fit(self, X_A, y, party_X_dict, global_step):
        if self.is_debug: print("==> start fit")

        # set batch data for party A.
        # only party A has label.
        self.party_a.set_batch(X_A, y, global_step)

        # set batch data for all other parties
        for idx, party_X in party_X_dict.items():
            self.party_dict[idx].set_batch(party_X, global_step)

        if self.is_debug: print("==> Guest receive intermediate computing results from hosts")
        comp_list = []
        for party in self.party_dict.values():
            logits = party.send_components(self.logger1)
            comp_list.append(logits)
        self.party_a.receive_components(component_list=comp_list, logger = self.logger1)



        if self.is_debug: print("==> Guest train and computes loss")
        with self.logger0.computation() as c:
            self.party_a.fit()
            loss = self.party_a.get_loss()
            c.report_metric('loss', loss)



        if self.is_debug: print("==> Guest sends out common grad")
        grad_result = self.party_a.send_gradients()



        if self.is_debug: print("==> Hosts receive common grad from guest and perform training")
        for party in self.party_dict.values():
            party.receive_gradients(grad_result, self.logger0, self.logger1)

        return loss

    def predict(self, X_A, party_X_dict):
        comp_list = []
        for id, party_X in party_X_dict.items():
            comp_list.append(self.party_dict[id].predict(party_X))
        return self.party_a.predict(X_A, component_list=comp_list)
