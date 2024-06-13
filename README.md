# DiffiPy (Differentiation interface for Python)

DiffiPy offers a versatile interface to multiple automatic (adjoint) differentiation (AAD) backends in Python. It utilizes a syntax inspired by numpy, ensuring straightforward integration and enhanced computational graph analysis.

## Goal

This package provides a backend-agnostic interface for the differentiation of scientific computations. The main goal is to:

- Enable automatic (adjoint) differentiation for all sorts of backends through minimal code refactoring. This allows for AAD in all kinds of codes by supporting not only functions of the general form `f(x)=y` but also complete scripts.
- Simplify the integration of automatic differentiation into existing code by maintaining a familiar syntax and interface, akin to numpy.

## Features

- **Graph Recording Tool**: Provides a graph recording tool using a syntax identical to numpy, enabling straightforward transition and enhanced computational graph analysis.
- **Minimal Refactoring Required**: Transitioning from numpy to diffipy involves simple changes, making it easy to integrate into existing projects.
- **Backend-Agnostic**: Supports various AAD backends, providing flexibility and extensibility.
- **Familiar Syntax**: Maintains a syntax and interface similar to numpy, allowing users to leverage their existing knowledge.

## Example

Refactoring your existing numpy code to use Diffipy is straightforward. Here are two simple options to demonstrate this:

# Original Code Using numpy
```python
import numpy as np

x = 1.7
a = np.sqrt(2)
y = np.exp(x * a)

print(y)
#11.069162135812496
```
# Option 1: Direct Replacement of Prefixes
Refactor by replacing the np. prefixes with dp. prefixes:
```python
import diffipy as dp
x = 1.7
a = dp.sqrt(2)
y = dp.exp(x * a)

print(y)
#exp((1.7 * sqrt(2)))
print(y.eval())
#11.069162135812496
```
# Option 2: Import 'diffipy' as 'np'
Refactor by importing diffipy as 'np' to keep the code almost identical:
```python
import diffipy as dp

x = 1.7
a = np.sqrt(2)
y = np.exp(x * a)

print(y)
#exp((1.7 * sqrt(2)))
print(y.eval())
#11.069162135812496
```
Notes: 
- The addition of y.eval() is necessary to obtain the float result, otherwise, it will display the computational graph.
- [Colab example](https://github.com/da-roth/DiffiPy/blob/main/DifferentiationInterface/examples-colab/introduction_colab.ipynb) for the above and the following example.

## Compatibility

- numpy (using finite differences)
- JAX
- PyTorch
- TensorFlow

# Example: 

```python
import diffipy as dp

backend_array = ['numpy', 'torch', 'tensorflow', 'jax']

for backend in backend_array:
    dp.setBackend(backend)

    x_value = 1.7
    x = dp.variable(x_value)
    a = dp.sqrt(2)
    y = dp.exp(x * a)
    
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

## Contributing

I welcome contributions and suggestions to improve the framework. Please feel free to open an issue or submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

