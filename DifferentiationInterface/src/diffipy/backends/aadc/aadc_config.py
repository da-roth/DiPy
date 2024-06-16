from .NodesAadc import *

backend_classes = {
    "aadc": {
        "exp": ExpNodeAadc,
        "pow": PowNodeAadc,
        "log": LogNodeAadc,
        "sqrt": SqrtNodeAadc,
        "cdf": CdfNodeAadc,
        "erf": ErfNodeAadc,
        "erfinv": ErfinvNodeAadc,
        "max": MaxNodeAadc,
        "sumVectorized": SumNodeVectorizedAadc,
        "seed": lambda value: Aadc.manual_seed(value),
        "if": IfNodeAadc,
        "sin": SinNodeAadc,
        "cos": CosNodeAadc
    }
}

backend_variable_classes = {
    "aadc": {
        "randomVariable": RandomVariableNodeAadc,
        "constant": ConstantNodeAadc,
        "input": VariableNodeAadc,
        "randomVariableNormal": RandomVariableNodeAadcNormal
    }
}

backend_valuation_and_grad_classes = {
    "aadc": {
        "grad": DifferentiationNodeAadc
    }
}

backend_result_classes = {
    "aadc": {
        "result": ResultNodeAadc
    }
}

backend_graph_differentiation_bool = True

backend_function_mappings = {
            "constant" : "aadc.idouble",
            "exp": "np.exp",
            "sin": "np.sin",
            "cos": "np.cos",
            "pow": "np.pow",
            "log": "np.log",
            "sqrt": "np.sqrt",
            "cdf": "np.cdf",
            "erf": "np.erf",
            "erfinv": "np.erfinv",
            "max": "np.max",
            "sumVectorized": "np.sum",
            "seed": "np.seed",
            "if": "aadc.iif"
    }
    