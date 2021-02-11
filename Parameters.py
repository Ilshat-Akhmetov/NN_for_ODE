import torch
from CustomClass import CustomClass
import numpy as  np
import random
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



#equation_rest = equation_rest.to(device)

num_epochs = 35
batch_size = 1
HiddenNeurons=200

number_of_inputs = 1
number_of_outputs = 1
model = CustomClass(1,HiddenNeurons,number_of_outputs)

test_size = 0.3 # 30 % of data belongs to the test domain
#model.to(device)

# domain of function
number_of_points = 50



optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 1)



# utility calculating function's derivative
def nth_derivative(function, variable, derivatives_degree):
    for i in range(derivatives_degree):
        grads = torch.autograd.grad(function, variable, create_graph=True)[0]
        function = grads.sum()
    return grads

left_bound = 0
right_bound = 1

# # in this case it is: u_xx + 0.2* u_x + u = -0.2 * exp(-x/5) * cos(x)
# differential_operator = lambda function, variable: nth_derivative(function,variable,2) + 0.2 * nth_derivative(function,variable,1) +\
#             function + 0.2 * torch.exp(-variable/5) * torch.cos(variable)

# # representing approximation of function so that it satisfies the boundary conditions
# # f(0) = 0  f_x(0) = 1
# approximation_of_function = lambda x, nn_model_value: x + x * x * nn_model_value

# # analytical solution of the equation
# def true_analytical_solution(x):
#     return torch.exp(-x/5)*torch.sin(x)



# in this case it is: u_xx + 0.2* u_x + u = -0.2 * exp(-x/5) * cos(x)
differential_operator = lambda function, variable: nth_derivative(function,variable,2) + 0.2 * nth_derivative(function,variable,1) +\
            function + 0.2 * torch.exp(-variable/5) * torch.cos(variable)

# representing approximation of function so that it satisfies the boundary conditions
# f(0) = 0  f(1) = 1
approximation_of_function = lambda x, nn_model_value: x*torch.sin(torch.tensor([1.0]))*torch.exp(torch.tensor([-0.2])) + x * (1-x) * nn_model_value

# analytical solution of the equation
def true_analytical_solution(x):
    return torch.exp(-x/5)*torch.sin(x)


