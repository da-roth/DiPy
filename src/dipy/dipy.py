from .Node import Node
from .NodesVariables import *
from .NodesOperations import *
from .NodesPytorch import *
from .NodesNumpy import *
from .backends import *

class dipy:

    @staticmethod
    def setBackend(backend):
        BackendConfig.backend = backend
        
    @staticmethod
    def variable(value, var_type='input'):
        if var_type == 'randomVariable':
            random_variable_class = BackendConfig.backend_variable_classes[BackendConfig.backend]["randomVariable"]
            return random_variable_class(value)
        elif var_type == 'constant':
            constant_class = BackendConfig.backend_variable_classes[BackendConfig.backend]["constant"]
            return constant_class(value)
        else:
            variable_class = BackendConfig.backend_variable_classes[BackendConfig.backend]["input"]
            return variable_class(value)
    
    @staticmethod
    def variable(value, var_type='input', var_name=None):
        if var_type == 'randomVariable':
            random_variable_class = BackendConfig.backend_variable_classes[BackendConfig.backend]["randomVariable"]
            return random_variable_class(value, var_name)
        elif var_type == 'constant':
            constant_class = BackendConfig.backend_variable_classes[BackendConfig.backend]["constant"]
            return constant_class(value)
        elif var_type == 'randomVariableNormal':
            random_variable_class = BackendConfig.backend_variable_classes[BackendConfig.backend]["randomVariableNormal"]
            return random_variable_class(value, var_name)
        else:
            variable_class = BackendConfig.backend_variable_classes[BackendConfig.backend]["input"]
            return variable_class(value, var_name)
    
    @staticmethod
    def constant(value):
        return ConstantNode(value)

    @staticmethod
    def sin(operand):
        return SinNodeNumpy(operand)

    @staticmethod
    def exp(operand):
        exp_class = BackendConfig.backend_classes[BackendConfig.backend]["exp"]
        return exp_class(operand)
      # if BackendConfig.backend == 'torch':
      #   return ExpNodeTorch(operand)
      # else:
      #   return ExpNodeNumpy(operand)

    @staticmethod
    def add(left, right):
        return AddNode(left, right)

    @staticmethod
    def sub(left, right):
        return SubNode(left, right)

    @staticmethod
    def mul(left, right):
        return MulNode(left, right)

    @staticmethod
    def div(left, right):
        return DivNode(left, right)

    @staticmethod
    def neg(operand):
        return NegNode(operand)

    @staticmethod
    def log(operand):
        if BackendConfig.backend == 'torch':
            return LogNodeTorch(operand)
        else:
            return LogNodeNumpy(operand)

    @staticmethod
    def sqrt(operand):
        if BackendConfig.backend == 'torch':
            return SqrtNodeTorch(operand)
        else:
            return SqrtNodeNumpy(operand)

    @staticmethod
    def cdf(operand):
        if BackendConfig.backend == 'torch':
            return CdfNodeTorch(operand)
        else:
            return CdfNodeNumpy(operand)

    @staticmethod
    def erf(operand):
        if BackendConfig.backend == 'torch':
            return ErfNodeTorch(operand)
        else:
            return ErfNodeNumpy(operand)

    @staticmethod
    def erfinv(operand):
        if BackendConfig.backend == 'torch':
            return ErfinvNodeTorch(operand)
        else:
            return ErfinvNodeNumpy(operand)

    @staticmethod
    def max(left, right):
        if BackendConfig.backend == 'torch':
            return MaxNodeTorch(left, right)
        else:
            return MaxNodeNumpy(left, right)

    @staticmethod
    def zeros(N):
        return [ConstantNode(0) for _ in range(N)]

    @staticmethod
    def sum(operands):
      if isinstance(operands, Node):
        if BackendConfig.backend == 'torch':
          return SumNodeVectorizedTorch(operands)
        else:
          return SumNodeVectorizedNumpy(operands)
      else:
        return SumNode(operands)

    @staticmethod
    def seed(value):
        if BackendConfig.backend == 'torch':
            torch.manual_seed(value)
        else:
            np.random.seed(seed=value)

    @staticmethod
    def if_(condition, true_value, false_value):
      if BackendConfig.backend == 'torch':
        return IfNodeTorch(condition, true_value, false_value)
      else:
        return IfNodeNumpy(condition, true_value, false_value)


