import torch
from Utilities import nth_derivative
# # in this case it is: u_xx + 0.2* u_x + u = -0.2 * exp(-x/5) * cos(x)
# differential_operator = lambda function, variable: nth_derivative(function,variable,2) + 0.2 * nth_derivative(function,variable,1) +\
#             function + 0.2 * torch.exp(-variable/5) * torch.cos(variable)

# # representing approximation of function so that it satisfies the boundary conditions
# # f(0) = 0  f_x(0) = 1
# approximation_of_function = lambda x, nn_model_value: x + x * x * nn_model_value

# # analytical solution of the equation
# def true_analytical_solution(x):
#     return torch.exp(-x/5)*torch.sin(x)


left_bound = 0
right_bound = 1

# in this case it is: u_xx + 0.2* u_x + u = -0.2 * exp(-x/5) * cos(x)
differential_operator = (
    lambda function, variable: nth_derivative(function, variable, 2)
    + 0.2 * nth_derivative(function, variable, 1)
    + function
    + 0.2 * torch.exp(-variable / 5) * torch.cos(variable)
)

# representing approximation of function so that it satisfies the boundary conditions
# f(0) = 0  f(1) = 1
approximation_of_function = (
    lambda x, nn_model_value: x
    * torch.sin(torch.tensor([1.0]))
    * torch.exp(torch.tensor([-0.2]))
    + x * (1 - x) * nn_model_value
)

# analytical solution of the equation
def true_analytical_solution(x):
    return torch.exp(-x / 5) * torch.sin(x)
