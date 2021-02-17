import torch
import torch.nn as nn
import torch.nn.functional as F


def get_params(model):
    """Retrieves list of all the named parameters in a model.

    Arguments:
        model::torch.nn.Module- Ideally the deep learning model, but can be a set of layers/layer too.

    Returns:
        param_list::list- List of all the named parameters in the model.
    """
    return [(name, params) for name, params in model.named_parameters() if params.requires_grad]


def assert_nan(name, params):
    """Tests for the presence of NaN values [torch.tensor(float("nan"))] in the given tensor of model parameter.

    Arguments:
        name::str- Name of the parameter
        params::torch.Tensor- Trainable named parameters associated with a layer

    Returns:
        None- Throws an exception in case any parameter is a NaN value.
    """
    try:
        assert not params.isnan().any()
    except AssertionError:
        print(f"NaN values found in the layer: {name}")


def assert_infinite(name, params):
    """Tests for the presence of infinite values [torch.tensor(float("Inf"))] in the given tensor of model parameter.

    Arguments:
        name::str- Name of the parameter
        params::torch.Tensor- Trainable named parameters associated with a layer

    Returns:
        None- Throws an exception in case any parameter is a Inf value.
    """
    try:
        assert not params.isinf().any()
    except AssertionError:
        print(f"Infinite values found in the layer: {name}")


if __name__ == "__main__":
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

    net = Net()
    for name, params in get_params(net):
        print(name)
        assert_nan(name, params)
        assert_infinite(name, params)
