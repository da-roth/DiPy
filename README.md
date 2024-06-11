# DiPy (Differentiation Interface for Python)

An interface to various automatic (adjoint) differentiation (AAD) backends in Python

## Goal

This package provides a backend-agnostic interface for the differentiation of scientific computations. The two main goals are to provide:

- Enable differentiation (e.g. through .grad()) without code refactoring. Hence, allowing for AAD for all kind of scripts even if not in the form `f(x)=y`
- Automatic creation of functions that allow for re-evaluation/differentiation the code for new input values.

## Features

- First- and second-order derivatives (gradients, Jacobians, Hessians and more)
- TBA

## Compatibility

- numpy (using finite differences)
- JAX
- PyTorch
- TensorFlow

## Example


```python
import numpy as np

x = 1.7
a = np.sqrt(2)

y = np.exp(x * a)

#Backend      Result       
#numpy        11.069162  
```

```python
import dipy.dipy as di

x_value = 1.7
x= di.variable(x_value)
a = np.sqrt(2)

y = di.exp(x * a)

result = y.eval()
derivative = y.grad(x)

#Backend      Result       Gradient    
#numpy        11.069162    15.654270   
```

```python
import dipy.dipy as di
import numpy as np

backend_array = ['numpy', 'torch', 'tensorflow', 'jax']

for backend in backend_array:
    di.setBackend(backend)

    x_value = 1.7
    x = di.variable(x_value)
    a = np.sqrt(2)
    y = di.exp(x * a)
    
    result = y.eval()
    derivative = y.grad(x)
#Backend      Result       Gradient    
#numpy        11.069162    15.654270   
#torch        11.069163    15.654160   
#tensorflow   11.069163    15.654160   
#jax          11.069162    15.654160     
```

## General remarks

- Inspired by [DifferentiationInterface for Julia](https://github.com/gdalle/DifferentiationInterface.jl?tab=readme-ov-file)