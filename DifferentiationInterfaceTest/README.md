# DiPy 

Testing and benchmarking utilities for automatic differentiation (AD) in Python, based on DiPy (DifferentiationInterface for Python).

## Goal

Make performance testing for a given function using different AAD backends easy

## Example


```python
backend_array = ['numpy', 'torch',  'tensorflow', 'jax']

for backend in backend_array:
    di.setBackend(backend)
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
    s0 = di.variable(s0_value, 'input','s0')
    K = di.variable(K_value, 'input','K')
    r = di.variable(r_value, 'input','r')
    sigma = di.variable(sigma_value, 'input','sigma')
    dt = di.variable(dt_value, 'input','dt')
    z = di.variable(z_value, 'randomVariableNormal','z')
    input_variables = [s0, K, r, sigma, dt, z]

    # Record Tape: Standard Monte Carlo

    s = s0 * di.exp((r - sigma **2 / 2) * dt + sigma * di.sqrt(dt) * z)
    payoff =  di.if_(s > K, s - K, 0)
    PV_standard = di.exp(-r * dt) * di.sum(payoff) / N

    ### Test performance
    di.seed(seed)
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
