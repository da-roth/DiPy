# For performance testing:
import time

#
# Base class for nodes in the computation graph
#

class Node:
    
    # Mapping dictionary to associate backend strings with classes
    backend = 'numpy'

    ## Constructor, evaluate and string representation
    def __init__(self):
        self.parents = []
    
    def Run(self):
        raise NotImplementedError("Must be implemented in subclasses")

    def __str__(self):
        raise NotImplementedError("Must be implemented in subclasses")

    ## Operator overloading to e.g. allow my.exp(x) + 3
    def __add__(self, other):
        from .NodesOperations import AddNode
        return AddNode(self, other)

    def __radd__(self, other):
        from .NodesOperations import AddNode
        return AddNode(other, self)

    def __sub__(self, other):
        from .NodesOperations import SubNode
        return SubNode(self, other)

    def __rsub__(self, other):
        from .NodesOperations import SubNode
        return SubNode(other, self)

    def __mul__(self, other):
        from .NodesOperations import MulNode
        return MulNode(self, other)

    def __rmul__(self, other):
        from .NodesOperations import MulNode
        return MulNode(other, self)

    def __truediv__(self, other):
        from .NodesOperations import DivNode
        return DivNode(self, other)

    def __rtruediv__(self, other):
        from .NodesOperations import DivNode
        return DivNode(other, self)

    def __neg__(self):
        from .NodesOperations import NegNode
        return NegNode(self)

    def __pow__(self, other):
        from .backends import BackendConfig
        pow_class = BackendConfig.backend_classes[BackendConfig.backend]["pow"]
        return pow_class(self, other)

    def __rpow__(self, other):
        from .backends import BackendConfig
        pow_class = BackendConfig.backend_classes[BackendConfig.backend]["pow"]
        return pow_class(self, other)

    def ensure_node(self, other):
        from .backends import BackendConfig
        if isinstance(other, Node):
            return other
        else:
            constant_class = BackendConfig.backend_variable_classes[BackendConfig.backend]["constant"]
            return constant_class(other)

    ## Comparison operators
    def __gt__(self, other):
        from .NodesOperations import ComparisonNode
        return ComparisonNode(self, other, '>')

    def __lt__(self, other):
        from .NodesOperations import ComparisonNode
        return ComparisonNode(self, other, '<')

    def __ge__(self, other):
        from .NodesOperations import ComparisonNode
        return ComparisonNode(self, other, '>=')

    def __le__(self, other):
        from .NodesOperations import ComparisonNode
        return ComparisonNode(self, other, '<=')

    def __eq__(self, other):
        from .NodesOperations import ComparisonNode
        return ComparisonNode(self, other, '==')

    def __ne__(self, other):
        from .NodesOperations import ComparisonNode
        return ComparisonNode(self, other, '!=')
    

    ## First draft performance benchmark
    def PerformanceIteration(self):
        a = self.Run()
        b = self.grad()
        return a + b

    def DoPerformanceTest(self, diffDirection, warmup_iterations=5, test_iterations=100):
        print('Starting performance test:')
        # Warm-up phase
        for _ in range(warmup_iterations):
            start_time = time.time()
            self.grad(diffDirection)
            end_time = time.time()
            warmup_duration = end_time - start_time

        # Test phase
        total_time = 0.0
        times = []
        for _ in range(test_iterations):
            start_time = time.time()
            self.grad(diffDirection)
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            times.append(execution_time)

        mean_time = total_time / test_iterations
        variance_time = sum((time - mean_time) ** 2 for time in times) / (test_iterations - 1)

        print(f"Mean execution time: {mean_time:.6f} seconds")
        print(f"Variance in execution time: {variance_time:.6f} seconds")

    ## Valuation and derivative logics.
    # Todo:
    # - Node.backend is strange since it's from another class
    # - finite differences into another class
    # - more than just s0 derivative
    
    def eval(self):
        from .backends import BackendConfig
        eval_class = BackendConfig.backend_result_classes[BackendConfig.backend]["result"]
        instance_eval_class = eval_class(self)
        return instance_eval_class.eval()
        
    def grad(self, diffDirection):
        from .backends import BackendConfig
        grad_class = BackendConfig.backend_valuation_and_grad_classes[BackendConfig.backend]["grad"]
        instance_grad_class = grad_class(self, diffDirection)
        return instance_grad_class.grad()

# Unary and binary node
class UnitaryNode(Node):
    def GetVariableList(self):
        return self.operand.GetVariableList()

class BinaryNode(Node):
    def GetVariableList(self):
        variableStrings = [self.left.GetVariableList(), self.right.GetVariableList()]
        return [x for x in variableStrings if x]

# Result node for nice outputs
class ResultNode(Node):
    def __init__(self, operationNode):
        super().__init__()
        self.operationNode = operationNode



