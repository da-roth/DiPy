# Diffipy Test

Testing and benchmarking utilities for automatic differentiation (AD) in Python, based on DiffiPy (Differentiation interface for Python).

## Main Features

Diffipy Test provides convenient utilities for evaluating the performance of automatic differentiation (AD) backends in Python. It supports two primary methods for differentiation:

1. **Direct Differentiation**: If the backend supports it (e.g., TensorFlow, PyTorch), Diffipy Test directly differentiates the recorded computational graph. This method is efficient and leverages the backend's native capabilities for gradient computation.

2. **Executable Creation**: For backends that require a function as input (e.g., JAX), Diffipy Test provides a method `get_executable()` that translates the recorded graph into an executable function. This approach allows seamless integration with diverse AD frameworks, ensuring consistent and accurate gradient calculations. Even for backends that support direct differentiation, using executables reduces overhead by avoiding repeated graph object evaluations and enables further backend-native optimizations (e.g., JIT compilation).

## Example

Here's an example ((Colab version[https://github.com/da-roth/DiffiPy/blob/main/DifferentiationInterfaceTest/examples-colab/introduction_colab.ipynb]) demonstrating the usage of Diffipy Test for performance testing across different AD backends:


```python
backend_array = ['numpy', 'torch',  'tensorflow', 'jax']

for backend in backend_array:
    dp.setBackend(backend)
    # Initialize variables and constants
    s0_value = 100.0
    K_value = 110.0
    r_value = 0.05
    sigma_value = 0.3
    dt_value = 1.0
    z_value = 0.5
    N = 1000000
    seed = 1

    # Define variables
    s0 = dp.variable(s0_value, 'input','s0')
    K = dp.variable(K_value, 'input','K')
    r = dp.variable(r_value, 'input','r')
    sigma = dp.variable(sigma_value, 'input','sigma')
    dt = dp.variable(dt_value, 'input','dt')
    z = dp.variable(z_value, 'randomVariableNormal','z')
    input_variables = [s0, K, r, sigma, dt, z]

    # Record Tape: Standard Monte Carlo

    s = s0 * dp.exp((r - sigma **2 / 2) * dt + sigma * dp.sqrt(dt) * z)
    payoff =  dp.if_(s > K, s - K, 0)
    PV_standard = dp.exp(-r * dt) * dp.sum(payoff) / N

    ### Test performance
    dp.seed(seed)
    pre_computed_random_variables = z.NewSample(N) # pre-evaluate random samples

    PV_standard.run_performance_test(input_variables, warmup_iterations = 10, test_iterations = 100)

# Backend              Result       Gradient (1. entry)  mean runtime    variance runtime
# nump                 10.023817    0.500207             0.036675        0.000002       
# numpy_jit            10.024070    0.500215             0.031361        0.000036       
# torch                10.020878    0.499224             0.002437        0.000020       
# torch_optimized      10.020878    0.499224             0.001283        0.000014       
# tensorflow           9.999600     0.499357             0.024196        0.000058       
# tensorflow_optimized 9.999600     0.499357             0.016357        0.000000       
# jax                  10.033216    0.499721             0.017167        0.002474 
```


