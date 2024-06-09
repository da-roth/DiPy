from .Node import *
from .NodesVariables import *
from .NodesOperations import *

import jax
import jax.numpy as jnp

class VariableNodeJAX(VariableNode):
    def __init__(self, value, identifier=None):
        super().__init__(value, identifier)
        self.value_jax = jnp.array(self.value)

    def Run(self):
        return self.value_jax

class RandomVariableNodeJAX(RandomVariableNode):
    def NewSample(self, sampleSize=1):
        self.SampleSize = sampleSize
        z_jax = jax.random.normal(jax.random.PRNGKey(0), shape=(1, sampleSize))
        self.value = 0.5 * (1 + jax.scipy.special.erf(z_jax / jnp.sqrt(2.0)))

class RandomVariableNodeJAXNormal(RandomVariableNode):
    def NewSample(self, sampleSize=1):
        self.SampleSize = sampleSize
        self.value = jax.random.normal(jax.random.PRNGKey(0), shape=(1, sampleSize))

class ConstantNodeJAX(ConstantNode):
    def Run(self):
        return jnp.array(self.value)

class ExpNodeJAX(ExpNode):
    def Run(self):
        return jnp.exp(self.operand.Run())

class LogNodeJAX(LogNode):
    def Run(self):
        return jnp.log(self.operand.Run())

class SqrtNodeJAX(SqrtNode):
    def Run(self):
        return jnp.sqrt(self.operand.Run())

class PowNodeJAX(PowNode):
    def Run(self):
        return jnp.power(self.left.Run(), self.right.Run())

class CdfNodeJAX(CdfNode):
    def Run(self):
        return 0.5 * (jax.scipy.special.erf(self.operand.Run() / jnp.sqrt(2.0)) + 1.0)

class ErfNodeJAX(ErfNode):
    def Run(self):
        return jax.scipy.special.erf(self.operand.Run())

class ErfinvNodeJAX(ErfinvNode):
    def Run(self):
        return jax.scipy.special.erfinv(self.operand.Run())

class MaxNodeJAX(MaxNode):
    def Run(self):
        return jnp.maximum(self.left.Run(), self.right.Run())

class SumNodeVectorizedJAX(Node):
    def __init__(self, operand):
        super().__init__()
        self.operand = self.ensure_node(operand)
        self.parents = [self.operand]

    def __str__(self):
        return f"erf({str(self.operand)})"

    def Run(self):
        return jnp.sum(self.operand.Run())

class IfNodeJAX(IfNode):
    def __init__(self, condition, true_value, false_value):
        super().__init__(condition, true_value, false_value)

    def Run(self):
        condition_value = self.condition.Run()
        true_value = self.true_value.Run()
        false_value = self.false_value.Run()
        return jnp.where(condition_value, true_value, false_value)

class GradNodeJAX(GradNode):
    def __init__(self, operand, diffDirection):
        super().__init__(operand, diffDirection)

    def grad(self):
        return np.nan

class ResultNodeJAX(ResultNode):
    def __init__(self, operationNode):
        super().__init__(operationNode)

    def eval(self):
        return self.operationNode.Run().item()
