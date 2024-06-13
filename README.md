# DiPy (Differentiation Interface for Python)

An interface to various automatic (adjoint) differentiation (AAD) backends in Python

## Goal

This package provides a backend-agnostic interface for the differentiation of scientific computations. The main goal is to:

- Enable automatic (adjoint) differentiation for all sorts of backends through minimal code refactoring. This allows for AAD in all kinds of codes by supporting not only functions of the general form `f(x)=y` but also complete scripts.
- Simplify the integration of automatic differentiation into existing codebases by maintaining a familiar syntax and interface, akin to NumPy.

## Features

- **Minimal Refactoring Required**: Transitioning from NumPy to DiPy involves simple changes, making it easy to integrate into existing projects.
- **Backend-Agnostic**: Supports various AAD backends, providing flexibility and extensibility.
- **Familiar Syntax**: Maintains a syntax and interface similar to NumPy, allowing users to leverage their existing knowledge.
- **Graph Recording Tool**: Provides a graph recording tool using a syntax identical to NumPy, enabling straightforward transition and enhanced computational graph analysis.

## Example

Refactoring your existing NumPy code to use DiPy is straightforward. Here are two simple options to demonstrate this:

# Original Code Using NumPy
```python
import numpy as np

x = 1.7
a = np.sqrt(2)
y = np.exp(x * a)

print(y)
#11.069162135812496
```
# Option 1: Direct Replacement of Prefixes
Refactor by replacing the np. prefixes with di. prefixes:
```python
import dipy as di
x = 1.7
a = di.sqrt(2)
y = di.exp(x * a)

print(y)
#exp((1.7 * sqrt(2)))
print(y.eval())
#11.069162135812496
```
# Option 2: Import DiPy as 'np'
Refactor by importing DiPy as 'np' to keep the code almost identical:
```python
import dipy as np

x = 1.7
a = np.sqrt(2)
y = np.exp(x * a)

print(y)
#exp((1.7 * sqrt(2)))
print(y.eval())
#11.069162135812496
```
Note: The addition of y.eval() is necessary to obtain the float result, otherwise, it will display the computational graph.

## General remarks

- Inspired by [DifferentiationInterface for Julia](https://github.com/gdalle/DifferentiationInterface.jl?tab=readme-ov-file)

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
a = di.sqrt(2)
y = di.exp(x * a)

result = y.eval()
derivative = y.grad(x)

#Backend      Result       Gradient    
#numpy        11.069162    15.654270   
```

```python
import dipy.dipy as di

backend_array = ['numpy', 'torch', 'tensorflow', 'jax']

for backend in backend_array:
    di.setBackend(backend)

    x_value = 1.7
    x = di.variable(x_value)
    a = di.sqrt(2)
    y = di.exp(x * a)
    
    result = y.eval()
    derivative = y.grad(x)
#Backend      Result       Gradient    
#numpy        11.069162    15.654270   
#torch        11.069163    15.654160   
#tensorflow   11.069163    15.654160   
#jax          11.069162    15.654160     
```
