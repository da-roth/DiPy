from .NodesNumpy import *

backend_classes = {
    "numpy": {
        "exp": ExpNodeNumpy,
        "pow": PowNodeNumpy,
        "log": LogNodeNumpy,
        "sqrt": SqrtNodeNumpy,
        "cdf": CdfNodeNumpy,
        "erf": ErfNodeNumpy,
        "erfinv": ErfinvNodeNumpy,
        "max": MaxNodeNumpy,
        "sumVectorized": SumNodeVectorizedNumpy,
        "seed": lambda value: np.random.seed(seed=value),
        "if": IfNodeNumpy,
        "sin": SinNodeNumpy,
        "cos": CosNodeNumpy
    }
}

backend_variable_classes = {
    "numpy": {
        "randomVariable": RandomVariableNodeNumpy,
        "constant": ConstantNode,
        "input": VariableNode,
        "randomVariableNormal": RandomVariableNodeNumpyNormal
    }
}

backend_valuation_and_grad_classes = {
    "numpy": {
        "grad": DifferentiationNodeNumpy
    }
}

backend_result_classes = {
    "numpy": {
        "result": ResultNodeNumpy
    }
}

backend_graph_differentiation_bool = {
    "numpy": {
        "differentiation_bool": True
    }
}

