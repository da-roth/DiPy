from .Node import *
from .NodesVariables import *
from .NodesOperations import *

import tensorflow as tf

class VariableNodeTF(VariableNode):
    def __init__(self, value, identifier=None):
        super().__init__(value, identifier)
        self.value_tf = tf.Variable(self.value, dtype=tf.float32)

    def Run(self):
        return self.value_tf

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

class ExpNodeTF(ExpNode):
    def Run(self):
        return tf.exp(self.operand.Run())

class LogNodeTF(LogNode):
    def Run(self):
        return tf.math.log(self.operand.Run())

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
        return f"erf({str(self.operand)})"

    def Run(self):
        return tf.reduce_sum(self.operand.Run())

class IfNodeTF(IfNode):
    def __init__(self, condition, true_value, false_value):
        super().__init__(condition, true_value, false_value)

    def Run(self):
        condition_value = self.condition.Run()
        true_value = self.true_value.Run()
        false_value = self.false_value.Run()
        return tf.where(condition_value, true_value, false_value)

class GradNodeTF(GradNode):
    def __init__(self, operand, diffDirection):
        super().__init__(operand, diffDirection)

    def grad(self):
        with tf.GradientTape() as tape:
            forward_evaluation = self.Run()
        return tape.gradient(forward_evaluation, self.diffDirection.value_tf).numpy()

class ResultNodeTF(ResultNode):
    def __init__(self, operationNode):
        super().__init__(operationNode)

    def eval(self):
        return self.operationNode.Run().numpy().item()
