from .Node import *
from .NodesVariables import *
from .NodesOperations import *

# Numerical and statistic computations
import torch

# Subclass for PyTorch
class VariableNodeTorch(VariableNode):
    def __init__(self, value, identifier=None):
        super().__init__(value, identifier)
        self.value_torch = torch.tensor(self.value, requires_grad=True)

    def Run(self):
        return self.value_torch
    

# Subclass for PyTorch
class RandomVariableNodeTorch(RandomVariableNode):
    def NewSample(self, sampleSize = 1):
        self.SampleSize = sampleSize
        z_torch = torch.normal(mean=0, std=1, size=(1,sampleSize))
        self.value = 0.5 * (1 + torch.erf(z_torch / torch.sqrt(torch.tensor(2.0))))

# Subclass for PyTorch
class RandomVariableNodeTorchNormal(RandomVariableNode):
    def NewSample(self, sampleSize = 1):
        self.SampleSize = sampleSize
        self.value = torch.normal(mean=0, std=1, size=(1,sampleSize))



# Subclass for PyTorch
class ConstantNodeTorch(ConstantNode):
    def Run(self):
        return torch.tensor(self.value)
    


# Subclass for PyTorch
class ExpNodeTorch(ExpNode):
    def Run(self):
        return torch.exp(self.operand.Run())



# Subclass for PyTorch
class LogNodeTorch(LogNode):
    def Run(self):
        return torch.log(self.operand.Run())
    

# Subclass for PyTorch
class SqrtNodeTorch(SqrtNode):
    def Run(self):
        return torch.sqrt(self.operand.Run())



# Subclass for PyTorch
class PowNodeTorch(PowNode):
    def Run(self):
        return torch.pow(self.left.Run(), self.right.Run())



# Subclass for PyTorch
class CdfNodeTorch(CdfNode):
    def Run(self):
        return 0.5 * (torch.erf(self.operand.Run() / torch.sqrt(torch.tensor(2.0))) + 1.0 )



# Subclass for PyTorch
class ErfNodeTorch(ErfNode):
    def Run(self):
        return torch.erf(self.operand.Run())
    

# Subclass for PyTorch
class ErfinvNodeTorch(ErfinvNode):
    def Run(self):
        return torch.erfinv(self.operand.Run())
    

# Subclass for PyTorch
class MaxNodeTorch(MaxNode):
    def Run(self):
        return torch.maximum(self.left.Run(), self.right.Run())
    



class SumNodeVectorizedTorch(Node):
    def __init__(self, operand):
        super().__init__()
        self.operand = self.ensure_node(operand)
        self.parents = [self.operand]

    def __str__(self):
        return f"erf({str(self.operand)})"

    def Run(self):
        return torch.sum(self.operand.Run())



class IfNodeTorch(IfNode):
    def __init__(self, condition, true_value, false_value):
      super().__init__(condition, true_value, false_value)

    def Run(self):
      condition_value = self.condition.Run()
      true_value = self.true_value.Run()
      false_value = self.false_value.Run()
      return torch.where(condition_value, true_value, false_value)
    

# Subclass for PyTorch
class GradNodeTorch(GradNode):
    def __init__(self, operand, diffDirection):
        super().__init__(operand, diffDirection)

    def grad(self):
        # Reset derivative graph
        self.diffDirection.value_torch.grad = None
        forwardevaluation = self.Run()

        # Backward
        forwardevaluation.backward()

        # Return S0 derivative
        derivative = self.diffDirection.value_torch.grad.item()
        return derivative
    
class ResultNodeTorch(ResultNode):
    def __init__(self, operationNode):
        super().__init__(operationNode)

    def eval(self):
        return self.operationNode.Run().item()