#!/usr/bin/env python
import csv
import os
import sys

from core.tensor import Tensor

# Ensure the project root is in the sys.path for proper module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def linear_regression(X, W, b):
    """
    Simple linear regression model: Y = X @ W + b
    X: input features as a Tensor (2D: [samples x features])
    W: weights as a Tensor (2D: [features x 1])
    b: bias as a Tensor (scalar or a Tensor that broadcasts)
    """
    return X @ W + b


def load_csv_to_tensors(filepath):
    """
    Reads a CSV file and returns two Tensors:
    - features Tensor (all columns except the last)
    - target Tensor (last column)

    Assumes that the first row might be a header, and that all data can be converted to float.
    """
    data = []
    with open(filepath, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    # Optionally, remove header if present (assume header row cannot be converted to float)
    try:
        # Try converting the first cell to float
        float(data[0][0])
    except ValueError:
        data = data[1:]  # Remove header row

    # Process each row: convert each value to float
    features = []
    target = []
    for row in data:
        # Assuming last column is target and the rest are features
        features.append([float(val) for val in row[:-1]])
        target.append([float(row[-1])])

    # Create Tensor objects from the lists
    tensor_features = Tensor(features)
    tensor_target = Tensor(target)

    return tensor_features, tensor_target


def main():
    # Define the path to the CSV file
    csv_path = os.path.join("data", "Salary_Data.csv")

    # Load data from CSV without using pandas
    tensor_X, tensor_y = load_csv_to_tensors(csv_path)

    # For example purposes, assume a model where Y = 2 * X + 1.
    # For a single feature, we set:
    #   W = Tensor([[2]])
    #   b = Tensor(1)
    X = tensor_X
    W = Tensor([[2]])
    b = Tensor(1)

    predictions = linear_regression(tensor_X, W, b)

    # Display the input and the predicted outputs
    print("Input X:")
    print(X)
    print("\nPredictions (Y = 2 * X + 1):")
    print(predictions)


if __name__ == "__main__":
    main()
