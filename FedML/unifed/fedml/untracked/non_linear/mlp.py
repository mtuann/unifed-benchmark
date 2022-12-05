from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = [8]):
        super(MLP, self).__init__()
        self.model = nn.Sequential()

        fc = nn.Linear(input_dim, hidden_dim[0])
        self.model.add_module('fc1', fc)
        self.model.add_module('Relu1', nn.ReLU())

        hidden_width = len(hidden_dim)
        for i in range(1,hidden_width):
            fc = nn.Linear(hidden_dim[i-1], hidden_dim[i])
            self.model.add_module('fc' + str(i+1), fc)
            self.model.add_module('Relu' + str(i+1), nn.ReLU())
        
        fc = nn.Linear(hidden_dim[-1], output_dim)
        self.model.add_module('fc_last', fc)

    def forward(self, x):
        return self.model(x)

