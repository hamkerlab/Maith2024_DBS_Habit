import pytensor.tensor as pt
from concurrent.futures import ProcessPoolExecutor


# Define your PyTorch function
def my_function(data):
    # Perform computations with data
    result = data**2  # Example operation
    return result


# List of data inputs for each function call
data_inputs = pt.ones(10)

# Use ProcessPoolExecutor to run function calls in parallel
with ProcessPoolExecutor() as executor:
    results = list(executor.map(my_function, data_inputs))

# The results list now holds the output of each function call
print(results)
