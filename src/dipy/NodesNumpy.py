from .Node import *
from .NodesVariables import *
from .NodesOperations import *

# Numerical and statistic computations
import numpy as np
import scipy.stats
import scipy.special



# Subclass for NumPy
class ExpNodeNumpy(ExpNode):
    def Run(self):
        return np.exp(self.operand.Run())

    
# Logarithm function node
class SinNodeNumpy(SinNode):
    def Run(self):
        return np.sin(self.operand.Run())
    
# Subclass for NumPy
class LogNodeNumpy(LogNode):
    def Run(self):
        return np.log(self.operand.Run())
    
# Subclass for NumPy
class SqrtNodeNumpy(SqrtNode):
    def Run(self):
        return np.sqrt(self.operand.Run())

# Subclass for NumPy
class PowNodeNumpy(PowNode):
    def Run(self):
        return self.left.Run() ** self.right.Run()
    

# Subclass for NumPy
class CdfNodeNumpy(CdfNode):
    def Run(self):
        return scipy.stats.norm.cdf(self.operand.Run())
    
# Subclass for NumPy
class ErfNodeNumpy(ErfNode):
    def Run(self):
        return scipy.special.erf(self.operand.Run())



# Subclass for NumPy
class ErfinvNodeNumpy(ErfinvNode):
    def Run(self):
        return scipy.special.erfinv(self.operand.Run())


# Subclass for NumPy
class MaxNodeNumpy(MaxNode):
    def Run(self):
        return np.maximum(self.left.Run(), self.right.Run())

# Subclass for Numpy
class RandomVariableNodeNumpy(RandomVariableNode):
    def NewSample(self, sampleSize = 1):
        self.value = np.random.uniform(size = sampleSize)

# Subclass for Numpy
class RandomVariableNodeNumpyNormal(RandomVariableNode):
    def NewSample(self, sampleSize = 1):
        self.value = np.random.normal(size = sampleSize)

# Summation node
class SumNodeVectorizedNumpy(Node):
    def __init__(self, operand):
        super().__init__()
        self.operand = self.ensure_node(operand)
        self.parents = [self.operand]

    def __str__(self):
        return f"erf({str(self.operand)})"

    def Run(self):
        return np.sum(self.operand.Run())
    

class IfNodeNumpy(IfNode):
        def Run(self):
            condition_value = self.condition.Run()
            true_value = self.true_value.Run()
            false_value = self.false_value.Run()
            return np.where(condition_value, true_value, false_value)
            # if self.condition.Run():
            #     return self.true_value.Run()
            # else:
            #     return self.false_value.Run()


# Subclass for PyTorch
class GradNodeNumpy(GradNode):
    def __init__(self, operand, diffDirection):
        super().__init__(operand, diffDirection)

    def grad(self):
        result = self.Run()
        h = 0.00001
        self.diffDirection.value = self.diffDirection.value + h
        result_h = self.Run()
        return (result_h - result) / h
    
class ResultNodeNumpy(ResultNode):
    def __init__(self, operationNode):
        super().__init__(operationNode)

    def eval(self):
        return self.operationNode.Run()