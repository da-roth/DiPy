# Import the nodes from which the following classes will inherit
from ...Node import *
from ...NodesVariables import *
from ...NodesOperations import *
from ...NodesDifferentiation import *

# Import backend specific packages
import tensorflow as tf
import numpy as np

###
### TensorFlow specific nodes.
###

class VariableNodeTF(VariableNode):
    def __init__(self, value, identifier=None):
        super().__init__(value, identifier)
        self.value = tf.Variable(self.value, dtype=tf.float32)

    def Run(self):
        return self.value

class RandomVariableNodeTF(RandomVariableNode):
    def NewSample(self, sampleSize=1):
        self.SampleSize = sampleSize
        z_tf = tf.random.normal(shape=(1, sampleSize))
        self.value = 0.5 * (1 + tf.math.erf(z_tf / tf.sqrt(2.0)))

class RandomVariableNodeTFNormal(RandomVariableNode):
    def NewSample(self, sampleSize=1):
        self.SampleSize = sampleSize
        self.value = tf.random.normal(shape=(1, sampleSize))

class ConstantNodeTF(ConstantNode):
    def Run(self):
        return tf.constant(self.value, dtype=tf.float32)
    def __str__(self):
        return f"constant({str(self.value)})"

class ExpNodeTF(ExpNode):
    def Run(self):
        return tf.exp(self.operand.Run())
    
class LogNodeTF(LogNode):
    def Run(self):
        return tf.log(self.operand.Run())
    
class SinNodeTF(SinNode):
    def Run(self):
        return tf.sin(self.operand.Run())

class CosNodeTF(CosNode):
    def Run(self):
        return tf.cos(self.operand.Run())

class SqrtNodeTF(SqrtNode):
    def Run(self):
        return tf.sqrt(self.operand.Run())

class PowNodeTF(PowNode):
    def Run(self):
        return tf.pow(self.left.Run(), self.right.Run())

class CdfNodeTF(CdfNode):
    def Run(self):
        return 0.5 * (tf.math.erf(self.operand.Run() / tf.sqrt(2.0)) + 1.0)

class ErfNodeTF(ErfNode):
    def Run(self):
        return tf.math.erf(self.operand.Run())

class ErfinvNodeTF(ErfinvNode):
    def Run(self):
        return tf.math.erfinv(self.operand.Run())

class MaxNodeTF(MaxNode):
    def Run(self):
        return tf.maximum(self.left.Run(), self.right.Run())

class SumNodeVectorizedTF(Node):
    def __init__(self, operand):
        super().__init__()
        self.operand = self.ensure_node(operand)
        self.parents = [self.operand]
    def __str__(self):
        return f"sumVectorized({str(self.operand)})"
    def Run(self):
        return tf.reduce_sum(self.operand.Run())
    def get_inputs(self):
        return self.operand.get_inputs()
    def get_inputs_with_diff(self):
        return self.operand.get_inputs_with_diff()
    def get_input_variables(self):
        return self.operand.get_input_variables()

class IfNodeTF(IfNode):
    def __init__(self, condition, true_value, false_value):
        super().__init__(condition, true_value, false_value)

    def Run(self):
        condition_value = self.condition.Run()
        true_value = self.true_value.Run()
        false_value = self.false_value.Run()
        return tf.where(condition_value, true_value, false_value)

class DifferentiationNodeTF(DifferentiationNode):
    def __init__(self, operand, diffDirection):
        super().__init__(operand, diffDirection)

    def backend_specific_grad(self):
        # Handle the case where self.diffDirection is a list
        if isinstance(self.diffDirection, list):
            gradients = []
            for direction in self.diffDirection:
                with tf.GradientTape() as tape:
                    forward_evaluation = self.Run()

                gradient = tape.gradient(forward_evaluation, direction.value).numpy()
                gradients.append(gradient)
            
            return gradients
        else:
            # Handle the case where self.diffDirection is a single object
            with tf.GradientTape() as tape:
                forward_evaluation = self.Run()

            gradient = tape.gradient(forward_evaluation, self.diffDirection.value).numpy()
            return gradient

class ResultNodeTF(ResultNode):
    def __init__(self, operationNode):
        super().__init__(operationNode)

    def eval(self):
        return self.operationNode.Run().numpy().item()
    
    def eval_and_grad_of_function(sef, myfunc, input_dict, diff_dict):
        with tf.GradientTape() as tape:
            result = myfunc(**input_dict)
        gradients = tape.gradient(result, [diff_dict[key] for key in diff_dict])
        return result, gradients

    def create_optimized_executable(self):
            def create_function_from_expression(expression_string, expression_inputs, backend):
                # Generate the function definition as a string
                inputs = ", ".join(expression_inputs)
                function_code = f"def myfunc({inputs}):\n    return {expression_string}\n"
                
                # Print the generated function code
                #print("Generated Function Code:")
                #print(function_code)

                # Compile the function code
                compiled_code = compile(function_code, "<string>", "exec")
                
                # Combine the provided backend with an empty dictionary to serve as the globals
                namespace = {**backend}
                exec(compiled_code, namespace)
                
                # Retrieve the dynamically created function
                #created_function = namespace["myfunc"]
            
                # Return the dynamically created function
                return namespace["myfunc"]

            expression = str(self.operationNode)

            function_mappings = self.get_function_mappings()

            import re
            
            for key, value in function_mappings.items():
                expression = expression.replace(key, value)

            # Function to replace 'constant' with 'tf.Variable'
            def replace_constant(match):
                value = match.group(1)
                # return f"tf.Variable({value}, dtype=tf.float32)"
                return f"tf.Variable({value}, dtype=tf.float32)"
            expression = re.sub(r'constant\(([^)]+)\)', replace_constant, expression)

            #expression = expression.replace('exp', 'tf.exp').replace('sqrt', 'tf.sqrt').replace('log', 'tf.log').replace('sin', 'tf.sin')
            input_names = self.operationNode.get_input_variables()

            tensorflow_func = create_function_from_expression(expression, input_names,  {'tf': tf})

            return tensorflow_func#myfunc_wrapper(tensorflow_func) #returning it in such a way that it needs tensor inputs for now
