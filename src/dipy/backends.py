from .NodesPytorch import *
from .NodesNumpy import *
from .NodesTensorflow import *
from .NodesJax import *

class BackendConfig:
    backend = 'numpy'  # default

    backend_classes = {
        "torch": {"exp": ExpNodeTorch, "pow": PowNodeTorch},
        "numpy": {"exp": ExpNodeNumpy, "pow": PowNodeNumpy},
        "tensorflow": {"exp": ExpNodeTF, "pow": PowNodeTF},
        "jax": {"exp": ExpNodeJAX, "pow": PowNodeJAX}
    }

    backend_variable_classes = {
        "torch": {"randomVariable": RandomVariableNodeTorch, "constant": ConstantNodeTorch, "input": VariableNodeTorch, "randomVariableNormal": RandomVariableNodeTorchNormal},
        "numpy": {"randomVariable": RandomVariableNodeNumpy, "constant": ConstantNode, "input": VariableNode, "randomVariableNormal": RandomVariableNodeNumpyNormal},
        "tensorflow": {"randomVariable": RandomVariableNodeTF, "constant": ConstantNodeTF, "input": VariableNodeTF, "randomVariableNormal": RandomVariableNodeTFNormal},
        "jax": {"randomVariable": RandomVariableNodeJAX, "constant": ConstantNodeJAX, "input": VariableNodeJAX, "randomVariableNormal": RandomVariableNodeJAXNormal}
    }

    backend_valuation_and_grad_classes = {
        "torch": {"grad": GradNodeTorch},
        "numpy": {"grad": GradNodeNumpy},
        "tensorflow": {"grad": GradNodeTF},
        "jax": {"grad": GradNodeJAX}
    }

    backend_result_classes = {
        "torch": {"result": ResultNodeTorch},
        "numpy": {"result": ResultNodeNumpy},
        "tensorflow": {"result": ResultNodeTF},
        "jax": {"result": ResultNodeJAX}
    }
