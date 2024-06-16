# Import the nodes from which the following classes will inherit
from ...Node import *
from ...NodesVariables import *
from ...NodesOperations import *
from ...NodesDifferentiation import *
from ..BackendHelper import *

# Import backend specific packages
import aadc
import numpy as np
import scipy.stats
import scipy.special

###
### PyAadc specific nodes.
###

class VariableNodeAadc(VariableNode):
    def __init__(self, value, identifier=None):
        super().__init__(value, identifier)
        self.value = aadc.idouble(self.value)
        self.require_grad = True
        #self.value = self.value

    def Run(self):
        return self.value

class ConstantNodeAadc(ConstantNode):
    def Run(self):
        return aadc.idouble(self.value)
    def __str__(self):
        return f"constant({str(self.value)})"

class ExpNodeAadc(ExpNode):
    def Run(self):
        return np.exp(self.operand.Run())
    
class SinNodeAadc(SinNode):
    def Run(self):
        return np.sin(self.operand.Run())
    
class CosNodeAadc(SinNode):
    def Run(self):
        return np.cos(self.operand.Run())
    
class LogNodeAadc(LogNode):
    def Run(self):
        return np.log(self.operand.Run())
    
class SqrtNodeAadc(SqrtNode):
    def Run(self):
        return np.sqrt(self.operand.Run())
    
class PowNodeAadc(PowNode):
    def Run(self):
        return self.left.Run() ** self.right.Run()
    
class CdfNodeAadc(CdfNode):
    def Run(self):
        return scipy.stats.norm.cdf(self.operand.Run())
    
class ErfNodeAadc(ErfNode):
    def Run(self):
        return scipy.special.erf(self.operand.Run())

class ErfinvNodeAadc(ErfinvNode):
    def Run(self):
        return scipy.special.erfinv(self.operand.Run())

class MaxNodeAadc(MaxNode):
    def Run(self):
        return np.maximum(self.left.Run(), self.right.Run())

class RandomVariableNodeAadc(RandomVariableNode):
    def NewSample(self, sampleSize = 1):
        self.value = np.random.uniform(size = sampleSize)

class RandomVariableNodeAadcNormal(RandomVariableNode):
    def NewSample(self, sampleSize = 1):
        self.value = np.random.normal(size = sampleSize)

class SumNodeVectorizedAadc(Node):
    def __init__(self, operand):
        super().__init__()
        self.operand = self.ensure_node(operand)
        self.parents = [self.operand]
    def __str__(self):
        return f"sumVectorized({str(self.operand)})"
    def Run(self):
        return np.sum(self.operand.Run())
    def get_inputs(self):
        return self.operand.get_inputs()
    def get_inputs_with_diff(self):
        return self.operand.get_inputs_with_diff()
    def get_input_variables(self):
        return self.operand.get_input_variables()

class IfNodeAadc(IfNode):
        def Run(self):
            condition_value = self.condition.Run()
            true_value = self.true_value.Run()
            false_value = self.false_value.Run()
            return aadc.iif(condition_value, true_value, false_value)
##
## Differentiation node is created on the graph when .grad() is called for on a node
##
class DifferentiationNodeAadc(DifferentiationNode):
    def __init__(self, operand, diffDirection):
        super().__init__(operand, diffDirection)

    def backend_specific_grad(self):
        # Handle the case where self.diffDirection is a list
        if isinstance(self.diffDirection, list):
            derivatives = []
            for direction in self.diffDirection:
                # Reset derivative graph for each tensor
                if direction.value.grad is not None:
                    direction.value.grad.zero_()
                forward_evaluation = self.Run()

                # Backward pass
                forward_evaluation.backward()

                # Get the gradient
                derivative = direction.value.grad.item()
                derivatives.append(derivative)
            return derivatives
        else:
            # Handle the case where self.diffDirection is a single object
            # Reset derivative graph
            if self.diffDirection.value.grad is not None:
                self.diffDirection.value.grad.zero_()
            forward_evaluation = self.Run()

            # Backward pass
            forward_evaluation.backward()

            # Get the gradient
            derivative = self.diffDirection.value.grad.item()
            return derivative
    
##
## Result node is used within performance testing. It contains the logic to create optimized executables and eval/grad of these.
##
class ResultNodeAadc(ResultNode):
    def __init__(self, operationNode):
        super().__init__(operationNode)

    def eval(self):
        return self.operationNode.Run().item()
        
    def eval_and_grad_of_function(sef, myfunc, input_dict, diff_dict):
        
        result = myfunc(**input_dict)

        for key in diff_dict: #Reset all gradients first
            diff_dict[key].grad = None

        result.backward()

        gradient = []
        for key in diff_dict:
            gradient_entry = diff_dict[key].grad
            gradient.append( gradient_entry)
        return result, gradient


    def create_optimized_executable(self):
            expression = str(self.operationNode)

            function_mappings = self.get_function_mappings()

            for key, value in function_mappings.items():
                expression = expression.replace(key, value)

            #expression = expression.replace('exp', 'Aadc.exp').replace('sqrt', 'Aadc.sqrt').replace('log', 'Aadc.log').replace('sin', 'Aadc.sin')
            input_names = self.operationNode.get_input_variables()

            Aadc_func = BackendHelper.create_function_from_expression(expression, input_names,  {'Aadc': Aadc})

            # Wrap it such that it can get values as inputs
            def myfunc_wrapper(func):
                def wrapped_func(*args):#, **kwargs):
                    # Convert all positional arguments to Aadc.tensor
                    converted_args = [Aadc.tensor(arg.value) for arg in args]
                    
                    # # Convert all keyword arguments to Aadc.tensor
                    # converted_kwargs = {key: Aadc.tensor(value) for key, value in kwargs.items()}
                    
                    # Call the original function with converted arguments
                    return func(*converted_args)#, **converted_kwargs)
                
                return wrapped_func

            return Aadc_func#myfunc_wrapper(Aadc_func) #returning it in such a way that it needs tensor inputs for now