# example_usage.py

from tensor import Tensor
from linear_regression import linear_regression


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


if __name__ == "__main__":
    main()
