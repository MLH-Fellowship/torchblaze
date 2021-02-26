---
id: mltests
title: MLTests
---

TorchBlaze comes with a PyTorch-based model testing suite, which is essentially a set of methods that can be used to perform a variety of integrity checks on your PyTorch models before and during the model training process in a bit to ensure that no unexpected results show up in your model outputs. 

## Getting Started with MLTests
---

For using the set of automated model-testing methods available in the __mltests__ module, the first step is to import it. This can be achieved with this simple command:

```py
import torchblaze.mltests as mls
```

To get a list of all the methods available in the module, you can use the following command:

```py
dir(mls)
```

Here's the list of methods, along with their usage, provided in the __mltests__ package.

## Model Tests
---

### # mls.get_params


Retrieves the list of all the named parameters associated with a model.

Arguments:
    model::torch.nn.Module- Ideally the deep learning model, but can be a singel model layer/set of layers too.

Returns:
    param_list::list- List of all the named parameters in the model.


#### Usage:

```py
import torchblaze.mltests as mls

# assuming that your model class object is stored in variable 'model' 
param_list = mls.get_params(model)

print(param_list)
```

Each element in the param_list will be a tuple, the first element of which is the parameter name, and the second element is the tensor of parameters associated with the named parameter. 

---

### # mls.check_cuda

Checks if the training device is of type CUDA or not.

Arguments:
    params::torch.Tensor- The parameters associated with a model layer.

Returns:
    None: Throws an exception if the training device is not CUDA-enabled. 


#### Usage:

Here is a an example of how you can implement this test. In order to use it as a standalone test, you need to provide the method, as an argument, a tensor of named parameters associated with the model. It can be any named parameter associated with the model.

```py
import torchblaze.mltests as mls

# getting the list of named parameters of the model 
param_list = mls.get_params(model)

# getting a single named parameter tensor from the list
param = param_list[0][1]

# performing the test
mls.check_cude(param)

```

In case the model is not training on a CUDA-enabled device, you will get a `DeviceNotCudaException` exception. 

---

### # mls.check_nan


Tests for the presence of NaN values [torch.tensor(float("nan"))] in the given tensor of model parameter.

Arguments:
    name::str- Name of the parameter
    params::torch.Tensor- Trainable named parameters associated with a layer

Returns:
    None- Throws an exception in case any parameter is a NaN value. 


#### Usage:


```py
import torchblaze.mltests as mls

# getting the list of named parameters of the model 
param_list = mls.get_params(model)

# performing nan-value check on every named parameter 
for name, param in param_list:
    check_nan(name, param)

```

In case any model parameter is a NaN value you will get a `NaNParamsException` exception. 

---

### # mls.check_infinite


Tests for the presence of infinite values [torch.tensor(float("Inf"))] in the given tensor of model parameter.

Arguments:
    name::str- Name of the parameter
    params::torch.Tensor- Trainable named parameters associated with a layer

Returns:
    None- Throws an exception in case any parameter is a Inf value. 


#### Usage:


```py
import torchblaze.mltests as mls

# getting the list of named parameters of the model 
param_list = mls.get_params(model)

# performing infinite-value check on every named parameter 
for name, param in param_list:
    check_infinite(name, param)

```

In case any model parameter is a infinite value you will get a `InfParamsException` exception. 

---

### # mls.check_smaller


Tests if the absolute value of any parameter exceeds a certain threshold.

Arguments:
    name::str- Name of the parameter
    params::torch.Tensor- Trainable named parameters associated with a layer
    upper_limit::float- The threshold value every parameter should be smaller than in terms of its absolute value.

Returns:
    None- Throws an exception in case any parameter is a exceeds threshold value. 


#### Usage:


```py
import torchblaze.mltests as mls

# getting the list of named parameters of the model 
param_list = mls.get_params(model)

# performing the check on every named parameter 
for name, param in param_list:
    check_smaller(name, param, upper_limit=1e2)

```

In case any parameter exceeds a upper_limit threshold value you will get a `ParamsTooLargeException` exception. 

---

### # mls.check_greater


Tests if the absolute value of any parameter falls below a certain threshold.

Arguments:
    name::str- Name of the parameter
    params::torch.Tensor- Trainable named parameters associated with a layer
    lower_limit::float- The threshold value every parameter should be greater than in terms of its absolute value.

Returns:
    None- Throws an exception in case any parameter is a NaN value. 


#### Usage:


```py
import torchblaze.mltests as mls

# getting the list of named parameters of the model 
param_list = mls.get_params(model)

# performing the check on every named parameter 
for name, param in param_list:
    check_greater(name, param, lower_limit=1e-2)

```

In case any parameter falls below the lower_limit threshold value you will get a `ParamsTooSmallException` exception. 

---

### # mls.check_gradient_smaller


Tests if the absolute gradient value for any parameter exceed a certain threshold. Can be used to check for gradient explosion.

Arguments:
    name::str- Name of the parameter
    params::torch.Tensor- Trainable named parameters associated with a layer
    grad_limit::float- A threshold value, such that, |params.grad| < grad_limit

Returns:
    None- Throws an exception in case the gradient for any parameter exceeds the threshold value (grad_limit)
    OR
    None- Throws an exception in case the method is used without running the loss.backward() method for backprop. 


#### Usage:

:::caution

Running this method for a model whose gradients have not been initialized (i.e., a model that has not gone training) will result in GradientsUninitializedException.

:::

```py
import torchblaze.mltests as mls

# getting the list of named parameters of the model 
param_list = mls.get_params(model)

# performing gradient check on every named parameter (assuming model has undergone at least one step of training) 
for name, param in param_list:
    check_gradient_smaller(name, params, grad_limit=1e4)

```

In case any parameter's gradient value exceeds the grad_limit threshold value, you will get a `GradientAboveThresholdException` exception. 

---

### # mls.check_params_changing


Tests if the parameters in the model/certain layer are changing after a training cycle.

Arguments:
    name::str- Name of the parameter
    params_list_old::list- List of trainable model parameters BEFORE training cycle.
    params_list_new::list- List of trainable model parameters AFTER training cycle.

Returns:
    None- Throws an exception in case the parameters are not changing 


#### Usage:


```py
import torchblaze.mltests as mls

# getting the list of named parameters of the model BEFORE training
param_list_old = mls.get_params(model)

...

# getting the list of named parameters of the model AFTER training
param_list_new = mls.get_params(model)

# performing the parameter-change test 
check_params_changing(params_list_old, params_list_new)

```

In case none of the model parameters change after training the model, you get a `ParamsNotChangingException` exception. 



## Automated Test
---

While the methods given above can be used to define your own model tests, the mltests module comes with an automated test too.

### # mls.model_test


Executes a suite of automated tests on the ML model.
Set <test_name> = False if you want to exclude a certain test from the test suite.

Arguments:
    model::nn.Module- The model that you want to test.

    batch_x::torch.Tensor- A single batch of data features to perform model checks

    batch_y::torch.Tensor- A single batch of data labels to perform model checks

    optim_fn::torch.optim- default=torch.optim.Adam, Optimizer algorithm to be used during model training 

    loss_fn- default=torch.nn.CrossEntropyLoss, Loss function to be used for model evaluation during training

    test_gradient_smaller::bool- default=True, Asserts if gradients exceed a certain threshold  

    test_greater::bool- default=True, Asserts if all parameters > threshold limit

    test_smaller::bool- default=True, Asserts if all parameters < threshold limit

    test_infinite::bool- default=True, Asserts that no parameters == infinite

    test_nan::bool- default=True, Asserts that no parameters == NaN

    test_params_changing::bool- Default=False, Asserts that all the parameters change/update after the training is complete

    test_cuda::bool- Default=False, Asserts that the model is training on a cuda-enabled GPU

    upper_limit::float- default=1e2, Absolute value of all parameters should be smaller than this threshold value

    lower_limit::float- default=1e-2, Absolute value of all parameters should be greater than this threshold value

    grad_limit::float- default=1e4, Absolute value of all gradients should be smaller than this threshold value 


#### Usage:


```py
import torchblaze.mltests as mls

# loss function
criterion = torch.nn.CrossEntropyLoss()

# optimizer function
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# running automated tests
model_test(model=net, batch_x=x, batch_y=y, optim_fn=optimizer, loss_fn=criterion)

```

In the example given above, we assume that the model class object is stored under that variable name `net`. 
x: A batch of training features
y: A batch of training lables, corresponding to the feature batch x. 