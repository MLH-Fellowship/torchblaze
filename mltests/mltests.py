import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_params(model):
    """Gets the list of all the named parameters in a model.

    Arguments:
        model::torch.nn.Module: The deep learning model.
    
    Returns:
        param_list::list: List of all the named parameters in the model.
    """
    return [(name, params) for name, params in model.named_parameters() if params.requires_grad]


def assert_nan(parameters):
    """Checks if a given list of model parameters consist of NaN values.

    Arguments:
        parameters::list: List of trainable named model parameters
    
    Returns:

    """
    for name, param in parameters:
        try:
            assert not param.isnan().any()
        except AssertionError:
            print(f"NaN values found in the layer: {name}")

        

if __name__ == "__main__":
    net = Net()
    for _, param in get_params(net):
        print(param.shape)
    assert_nan(get_params(net))  
    # print("Test successful")  
    # try:
    #     print("Works for single layer too?")
    #     conv = nn.Linear(in_features=32, out_features=64)
    #     maxpool = nn.MaxPool2d(kernel_size=4)
    #     get_params(conv)
    #     print("YESSS!")
    # except:
    #     pass
    