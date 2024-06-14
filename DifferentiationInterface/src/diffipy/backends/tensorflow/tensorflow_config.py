from .NodesTensorflow import *

backend_classes = {
    "tensorflow": {
        "exp": ExpNodeTF,
        "pow": PowNodeTF,
        "log": LogNodeTF,
        "sqrt": SqrtNodeTF,
        "cdf": CdfNodeTF,
        "erf": ErfNodeTF,
        "erfinv": ErfinvNodeTF,
        "max": MaxNodeTF,
        "sumVectorized": SumNodeVectorizedTF,
        "seed": lambda value: tf.random.set_seed(value),
        "if": IfNodeTF,
        "sin": SinNodeTF,
        "cos": CosNodeTF
    }
}

backend_variable_classes = {
    "tensorflow": {
        "randomVariable": RandomVariableNodeTF,
        "constant": ConstantNodeTF,
        "input": VariableNodeTF,
        "randomVariableNormal": RandomVariableNodeTFNormal
    }
}

backend_valuation_and_grad_classes = {
    "tensorflow": {
        "grad": DifferentiationNodeTF
    }
}

backend_result_classes = {
    "tensorflow": {
        "result": ResultNodeTF
    }
}

backend_graph_differentiation_bool = {
    "tensorflow": {
        "differentiation_bool": True
    }
}
