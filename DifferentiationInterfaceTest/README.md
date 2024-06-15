# Diffipy Test

Testing and benchmarking utilities for automatic differentiation (AD) in Python, based on DiffiPy (Differentiation interface for Python).

## Main Features

Diffipy Test provides convenient utilities for evaluating the performance of automatic differentiation (AD) backends in Python. It supports two primary methods for differentiation:

1. **Direct Differentiation**: If the backend supports it (e.g., TensorFlow, PyTorch), Diffipy Test directly differentiates the recorded computational graph. This method is efficient and leverages the backend's native capabilities for gradient computation.

2. **Executable Creation**: For backends that require a function as input (e.g., JAX), Diffipy Test provides a method `get_executable()` that translates the recorded graph into an executable function. This approach allows seamless integration with diverse AD frameworks, ensuring consistent and accurate gradient calculations. Even for backends that support direct differentiation, using executables reduces overhead by avoiding repeated graph object evaluations and enables further backend-native optimizations (e.g., JIT compilation).

## Example

Here's an example ([Colab version](https://github.com/da-roth/DiffiPy/blob/main/DifferentiationInterfaceTest/examples-colab/introduction_colab.ipynb)) demonstrating the usage of Diffipy Test for performance testing across different AD backends:


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

    func_input_variables = [s0, K, r, sigma, dt, z]
    diff_variables = [s0, K, r, sigma, dt] 

    # Record Tape: Standard Monte Carlo

    s = s0 * dp.exp((r - sigma **2 / 2) * dt + sigma * dp.sqrt(dt) * z)
    payoff =  dp.if_(s > K, s - K, 0)
    PV_standard = dp.exp(-r * dt) * dp.sum(payoff) / N

    ### Test performance
    dp.seed(seed)
    pre_computed_random_variables = z.NewSample(N) # pre-evaluate random samples

    PV_standard.run_performance_test(func_input_variables, diff_variables, warmup_iterations = 10, test_iterations = 40)

# Backend              Eval-Result  mean runtime    variance runtime    gradient directions: {'s0', 'r', 'K', 'sigma', 'dt'}
# numpy                10.023520    0.054356        0.000020            [0.5002012963828406, -0.36360554602765655, 39.996882379789156, 39.89751398876251, 7.9844421986052785]
# numpy_as_func        10.023520    0.056490        0.000057            {'s0': 0.5002013217847434, 'K': -0.36360553590242256, 'r': 39.99688241371757, 'sigma': 39.89751398751906, 'dt': 7.984442234842958}
# torch                10.020878    0.007389        0.000000            [0.4992237389087677, -0.3627409040927887, 39.9015007019043, 39.90696334838867, 7.981120586395264]
# torch_as_func        10.020875    0.001370        0.000000            [tensor(0.4992), tensor(-0.3627), tensor(39.9015), tensor(39.9070), tensor(7.9811)]
# tensorflow           9.999600     0.080152        0.000012            [0.49935734, -0.36305577, 39.936134, 39.80618, 7.9677343]
# tensorflow_as_func   9.999602     0.016340        0.000001            [<tf.Tensor: shape=(), dtype=float32, numpy=0.49935734>, <tf.Tensor: shape=(), dtype=float32, numpy=-0.36305577>, <tf.Tensor: shape=(), dtype=float32, numpy=39.936134>, <tf.Tensor: shape=(), dt....
# jax_as_func          10.033226    0.012359        0.000002            {'s0': Array(0.49972072, dtype=float32, weak_type=True), 'K': Array(-0.36308047, dtype=float32, weak_type=True), 'r': Array(39.938858, dtype=float32, weak_type=True), 'sigma': Array(39.954735, ....
```


