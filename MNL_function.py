#Importing necessary libraries for vectorization and visualizaiton
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
    Function to calculate the probabilities of evey datapoint in each alternative in a multinomial logit model.
    Has the following parameters:
        parameters - Dictionary of b coefficients.
        data - Dictionary of independent variables X's.
        utilities - List of functions to calculate deterministic utilities for each alternative.
    Return type:
    Dictionary with keys as alternatives and values as lists of probabilities for each data point.
"""
def calculate_probabilities(parameters, data, utilities):  
    try:
        # Checking for empty inputs
        if not parameters or not data or not utilities:
            raise ValueError("Parameters, data, or utilities are empty")
        
        # Checking for consistency in data lengths for each independent variable (not missing data)
        if len(set(len(val) for val in data.values())) != 1:
            raise ValueError("Inconsistent data lengths in the input data")

        # Converting data to a numpy matrix for vectorized operations
        data_matrix = np.array([data[key] for key in data])

        # Calculating deterministic utilities for each alternative
        utilities_values = []
        for utility in utilities:
            try:
                utilities_values.append(utility(parameters, data_matrix))
            except Exception as e:
                raise ValueError(f"Error in utility function: {e}")

        # Converting list to numpy array
        utilities_values = np.array(utilities_values)

        # Checking for the correct number of utilities(same as number of alternatives)
        if utilities_values.shape[0] != len(utilities):
            raise ValueError("Number of utility functions does not match the number of alternatives")

        # Vectorizing utilities values to exponential
        exp_utilities = np.exp(utilities_values)

        # Sum of exponentials of utilities values across all alternatives to provide for denominator in probabilty calculation
        sum_exp_utilities = np.sum(exp_utilities, axis=0)

        # Calculating probabilities for each alternative
        probabilities = exp_utilities / sum_exp_utilities

        # Return dictionary of the probabilities for each alternative
        return {f"P{i+1}": probabilities[i].tolist() for i in range(len(utilities))}
    
    #Printing error if any exception raiswd
    except Exception as e:
        print(f"Error occurred: {e}")
        return {}

#------------------------------------------------------------------------
"""
    Function to plot a bar chart for the probability distribution of each alternative for a given data point.
    Has the parameters:
        probabilities: Dictionary of calculated probabilities from `calculate_probabilities` function.
        data_point_index: Index of the data point to visualize.
"""
def plot_probability_distribution(probabilities, data_point_index):
    # Extracting probabilities for the specified data point
    labels = probabilities.keys()
    values = [prob[data_point_index] for prob in probabilities.values()]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Alternatives')
    plt.ylabel('Probability')
    plt.title(f'Probability Distribution for Data Point {data_point_index}')
    plt.show()
#------------------------------------------------------------------------
# Defining utility functions and data according to specific specific problem 

# Sample utility functions
def utility1(params, data):
    return params['b01'] + params['b1'] * data[0] + params['b2'] * data[1]

def utility2(params, data):
    return params['b02'] + params['b1'] * data[0] + params['b2'] * data[1]

def utility3(params, data):
    return params['b03'] + params['b1'] * data[2] + params['b2'] * data[2]
    
utilities = [utility1, utility2, utility3]

# Sample data and parameters based on PEP8
data = {
    'X1': [2, 3, 5, 7, 1, 8, 4, 5, 6, 7],
    'X2': [1, 5, 3, 8, 2, 7, 5, 9, 4, 2],
    'Sero': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
parameters = {
    'b01': 0.1, 
    'b1': 0.5, 
    'b2': 0.5, 
    'b02': 1, 
    'b03': 0
}

# Function call for testing
probabilities = calculate_probabilities(parameters, data, utilities)
print(probabilities)

plot_probability_distribution(probabilities, 1)