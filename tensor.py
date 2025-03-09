


class Tensor:
    def __init__(self, data):
        """
        Initialize the tensor.
        If data is not a list, it is treated as a scalar.
        """
        self.data = data if isinstance(data, list) else data
        self.shape = self._infer_shape(self.data)

    def _infer_shape(self, data):
        """Recursively infer the shape of the tensor data."""
        if isinstance(data, list):
            if not data:  # empty list
                return (0,)
            # Check if elements are lists (i.e. 2D tensor)
            if isinstance(data[0], list):
                return (len(data), len(data[0]))
            else:
                return (len(data),)
        else:
            return ()  # scalar

    def __repr__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._elementwise_op(other, lambda a, b: a + b))
        else:
            # Assume other is a scalar
            return Tensor(self._elementwise_op_scalar(lambda a: a + other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._elementwise_op(other, lambda a, b: a - b))
        else:
            return Tensor(self._elementwise_op_scalar(lambda a: a - other))

    def __rsub__(self, other):
        # For scalar - tensor
        return Tensor(self._elementwise_op_scalar(lambda a: other - a))

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._elementwise_op(other, lambda a, b: a * b))
        else:
            return Tensor(self._elementwise_op_scalar(lambda a: a * other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def _elementwise_op(self, other, op):
        if self.shape != other.shape:
            raise ValueError("Shapes do not match for element-wise operation")
        return self._recursive_op(self.data, other.data, op)

    def _recursive_op(self, a, b, op):
        if isinstance(a, list) and isinstance(b, list):
            return [self._recursive_op(x, y, op) for x, y in zip(a, b)]
        else:
            return op(a, b)

    def _elementwise_op_scalar(self, op):
        def recursive_apply(x):
            if isinstance(x, list):
                return [recursive_apply(item) for item in x]
            else:
                return op(x)
        return recursive_apply(self.data)

    def __matmul__(self, other):
        """
        Matrix multiplication for 2D tensors using the @ operator.
        For tensors A (m x n) and B (n x p), the result will be (m x p).
        """
        if not (len(self.shape) == 2 and len(other.shape) == 2):
            raise ValueError("Matrix multiplication is only defined for 2D tensors")
        m, n = self.shape
        n2, p = other.shape
        if n != n2:
            raise ValueError("Inner dimensions must match for matrix multiplication")
        result = []
        for i in range(m):
            row = []
            for j in range(p):
                s = 0
                for k in range(n):
                    s += self.data[i][k] * other.data[k][j]
                row.append(s)
            result.append(row)
        return Tensor(result)


def linear_regression(X, W, b):
    """
    Simple linear regression model: Y = X @ W + b
    X: input features as a Tensor (2D: [samples x features])
    W: weights as a Tensor (2D: [features x 1])
    b: bias as a Tensor (scalar or a Tensor that broadcasts)
    """
    return X @ W + b


if __name__ == "__main__":
    # Example usage:
    # Suppose we have 4 data points with 1 feature each.
    X = Tensor([[1], [2], [3], [4]])
    W = Tensor([[2]])  # weight (for the single feature)
    b = Tensor(1)      # bias (scalar)
    
    predictions = linear_regression(X, W, b)
    print("Predictions:", predictions)
