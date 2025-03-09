def linear_regression(X, W, b):
    """
    Simple linear regression model: Y = X @ W + b
    X: input features as a Tensor (2D: [samples x features])
    W: weights as a Tensor (2D: [features x 1])
    b: bias as a Tensor (scalar or a Tensor that broadcasts)
    """
    return X @ W + b
