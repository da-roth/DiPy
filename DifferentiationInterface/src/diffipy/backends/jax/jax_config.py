from .NodesJax import *

backend_classes = {
    "jax": {
        "exp": ExpNodeJAX,
        "pow": PowNodeJAX,
        "log": LogNodeJAX,
        "sqrt": SqrtNodeJAX,
        "cdf": CdfNodeJAX,
        "erf": ErfNodeJAX,
        "erfinv": ErfinvNodeJAX,
        "max": MaxNodeJAX,
        "sumVectorized": SumNodeVectorizedJAX,
        "seed": lambda value: jax.random.PRNGKey(seed=value),
        "if": IfNodeJAX,
        "sin": SinNodeJAX,
        "cos": CosNodeJAX
    }
}

backend_variable_classes = {
    "jax": {
        "randomVariable": RandomVariableNodeJAX,
        "constant": ConstantNodeJAX,
        "input": VariableNodeJAX,
        "randomVariableNormal": RandomVariableNodeJAXNormal
    }
}

backend_valuation_and_grad_classes = {
    "jax": {
        "grad": DifferentiationNodeJAX
    }
}

backend_result_classes = {
    "jax": {
        "result": ResultNodeJAX
    }
}

backend_graph_differentiation_bool = False