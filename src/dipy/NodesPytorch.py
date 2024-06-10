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
class SinNodeTorch(SinNode):
    def Run(self):
        return torch.sin(self.operand.Run())
    
# Subclass for PyTorch
class CosNodeTorch(CosNode):
    def Run(self):
        return torch.cos(self.operand.Run())

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
    
    def create_optimized_executable(self):
        def create_function_from_expression(expression_string, expression_inputs, backend):
            # Generate the function definition as a string
            inputs = ", ".join(expression_inputs)
            function_code = f"def myfunc({inputs}):\n    return {expression_string}\n"
            
            # Print the generated function code
            print("Generated Function Code:")
            print(function_code)

            # Compile the function code
            compiled_code = compile(function_code, "<string>", "exec")
            
            # Combine the provided backend with an empty dictionary to serve as the globals
            namespace = {**backend}
            exec(compiled_code, namespace)
            
            # Retrieve the dynamically created function
            created_function = namespace["myfunc"]
        
            # Return the dynamically created function
            return namespace["myfunc"]

        expression = str(self.operationNode)

            # Replace function names in the expression string
        function_mappings = {
            "exp": "torch.exp",
            "sin": "torch.sin",
            "cos": "torch.cos",
            "pow": "torch.pow",
            "log": "torch.log",
            "sqrt": "torch.sqrt",
            "cdf": "torch.cdf",
            "erf": "torch.erf",
            "erfinv": "torch.erfinv",
            "max": "torch.max",
            "sumVectorized": "torch.sumVectorized",
            "seed": "torch.seed",
            "if": "torch.if"
        }
        
        for key, value in function_mappings.items():
            expression = expression.replace(key, value)

        #expression = expression.replace('exp', 'torch.exp').replace('sqrt', 'torch.sqrt').replace('log', 'torch.log').replace('sin', 'torch.sin')
        input_names = self.operationNode.get_input_variables()

        torch_func = create_function_from_expression(expression, input_names,  {'torch': torch})
        return torch_func