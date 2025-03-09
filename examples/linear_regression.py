import os
import sys

from core.tensor import Tensor


def linear_regression(X, W, b):
    """
    Simple linear regression model: Y = X @ W + b
    X: input features as a Tensor (2D: [samples x features])
    W: weights as a Tensor (2D: [features x 1])
    b: bias as a Tensor (scalar or a Tensor that broadcasts)
    """
    return X @ W + b


def main():
    # Define the input data (4 samples with 1 feature each)
    X = Tensor([[1], [2], [3], [4]])

    # Define the weight and bias for our linear model
    # This sets up a model where Y = 2 * X + 1
    W = Tensor([[2]])
    b = Tensor(1)

    # Compute predictions using our linear regression model
    predictions = linear_regression(X, W, b)

    # Display the input and the predicted outputs
    print("Input X:")
    print(X)
    print("\nPredictions (Y = 2 * X + 1):")
    print(predictions)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    main()
