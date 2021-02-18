import torch
import torch.nn as nn
import torch.nn.functional as F


class ParamsTooLargeException(Exception):
    pass 

class ParamsTooSmallException(Exception):
    pass

class GradientAboveThresholdException(Exception):
    pass

class NaNParamsException(Exception):
    pass

class InfParamsExceptions(Exception):
    pass

def get_params(model):
    """Retrieves list of all the named parameters in a model.

    Arguments:
        model::torch.nn.Module- Ideally the deep learning model, but can be a set of layers/layer too.

    Returns:
        param_list::list- List of all the named parameters in the model.
    """
    return [(name, params) for name, params in model.named_parameters() if params.requires_grad]


def check_nan(name, params):
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
        print(f"\nNaN values found in the layer: {name}")


def check_infinite(name, params):
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
        print(f"\nInfinite values found in the layer: {name}")


def check_smaller(name, params, upper_limit=0):
    """Tests if the absolute value of any parameter exceeds a certain threshold.

    Arguments:
        name::str- Name of the parameter
        params::torch.Tensor- Trainable named parameters associated with a layer
        upper_limit::float- The threshold value every parameter should be smaller than in terms of its absolute value.

    Returns:
        None- Throws an exception in case any parameter is a exceeds threshold value.
    """
    try:
        assert params.abs().less(upper_limit).any()
    except AssertionError:
        print(f"\nCertain parameters in layer '{name}' found to be greater than the threshold value = {upper_limit}.")



def check_greater(name, params, lower_limit=0):
    """Tests if the absolute value of any parameter falls below a certain threshold.

    Arguments:
        name::str- Name of the parameter
        params::torch.Tensor- Trainable named parameters associated with a layer
        lower_limit::float- The threshold value every parameter should be greater than in terms of its absolute value.

    Returns:
        None- Throws an exception in case any parameter is a NaN value.
    """
    try:
        assert params.abs().greater(lower_limit).any()
    except AssertionError:
        print(f"\nCertain parameters in layer '{name}' found to be smaller than the threshold value = {lower_limit}.")


def check_gradient_smaller(name, params, grad_limit=1e3):
    """Tests if the absolute gradient value for any parameter exceed a certain threshold. Can be used to check for gradient explosion.

    Arguments:
        name::str- Name of the parameter
        params::torch.Tensor- Trainable named parameters associated with a layer
        grad_limit::float- A threshold value, such that, |params.grad| < grad_limit

    Returns:
        None- Throws an exception in case the gradient for any parameter exceeds the threshold value (grad_limit)
        OR
        None- Throws an exception in case the method is used without running the loss.backward() method for backprop.
    """
    grads = params.grad
    try:
        assert not (grads == None)
    except AssertionError:
        print("\nModel gradients not initialized. Kindly run loss.backwards() to initialize gradients first.")
        return 

    try:
        assert not grads.abs().greater(grad_limit).any()
    except AssertionError:
        print(f"\nGradients (absolute) for certain parameters in layer '{name}' found to be greater than the threshold grad_limit value = {grad_limit}.")
        return 



def model_test(model, batch_label, batch_target, 
                optim=torch.optim.Adam, loss=torch.nn.CrossEntropyLoss, 
                check_gradient_smaller=True, check_greater=True, 
                check_smaller=True, check_infinite=True, check_nan=True,
                upper_limit=1e2, lower_limit=1e-2, grad_limit=1e4):
    """Executes a suite of tests on the ML model.
    Set <test_name> = False if you want to exclude a certain test from the test suite.

    Arguments:
        model::nn.Module- The model that you want to test.

        batch_x::torch.Tensor- A single batch of data features to perform model checks
        
        batch_y::torch.Tensor- A single batch of data labels to perform model checks

        optim::torch.optim- default=torch.optim.Adam, Optimizer algorithm to be used during model training 
        
        loss- default=torch.nn.CrossEntropyLoss, Loss function to be used for model evaluation during training

        check_gradient_smaller::bools- default=True, Asserts if gradients exceed a certain threshold  
        
        check_greater::bools- default=True, Asserts if all parameters > threshold limit
        
        check_smaller::bools- default=True, Asserts if all parameters < threshold limit
        
        check_infinite::bools- default=True, Asserts that no parameters == infinite
        
        check_nan::bools- default=True, Asserts that no parameters == NaN
        
        upper_limit::float- default=1e2, Absolute value of all parameters should be smaller than this threshold value
        
        lower_limit::float- default=1e-2, Absolute value of all parameters should be greater than this threshold value
        
        grad_limit::float- default=1e4, Absolute value of all gradients should be smaller than this threshold value
    """
    pass



if __name__ == "__main__":
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output


    def train(args, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break


    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    net = Net()
    for name, params in get_params(net):
        print(name)
        check_nan(name, params)
        check_infinite(name, params)
        check_greater(name, params, lower_limit=0.001)
        check_smaller(name, params, upper_limit=0.1)
        check_gradient_smaller(name, params, grad_limit=1e3)
