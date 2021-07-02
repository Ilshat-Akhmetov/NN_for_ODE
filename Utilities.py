import torch

# Metrics
LossMSE = torch.nn.MSELoss()
AbsoluteError = lambda true_solution, approximation: torch.abs(
    true_solution - approximation
)
MaxAbsoluteError = lambda true_solution, approximation: torch.max(
    AbsoluteError(true_solution, approximation)
)

# def relative_error(true_sol, approximation):
#     return 100 * torch.sum(AbsoluteError(true_sol,approximation))/torch.sum(torch.abs(true_sol))

## calctulate relative error excluding cases when solution equals zero
def relative_error(true_sol, approximation):
    not_null_condition = true_sol != 0
    return 100 * torch.max(
        torch.abs(
            (true_sol[not_null_condition] - approximation[not_null_condition])
            / true_sol[not_null_condition]
        )
    )

def get_domain(left_bound, right_bound, number_of_points):
    train_x = torch.linspace(
        left_bound, right_bound, number_of_points, requires_grad=True
    )
    train_x = train_x.unsqueeze(dim=1)
    valid_x = torch.cat(
        (
            train_x[0:1],
            train_x[1:number_of_points - 1] + 1 / (2 * number_of_points),
            train_x[number_of_points - 1 : number_of_points],
        ),
        dim=0,
    )
    return train_x, valid_x

def nth_derivative(function, variable, derivatives_degree):
    for i in range(derivatives_degree):
        grads = torch.autograd.grad(function, variable, create_graph=True)[0]
        function = grads.sum()
    return grads

import matplotlib.pyplot as plt


def plot_function(domain, function, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.plot(domain.detach().numpy(), function.detach().numpy())
    plt.show()

def plot_results(epochs, mse_loss_train, mse_loss_valid, abs_err_train, abs_err_valid, relative_err_train, relative_err_valid):
    plot_function(epochs, mse_loss_train, "MSE Loss on train set", "Epoch", "Loss")
    plot_function(epochs, mse_loss_valid, "MSE Loss on validation set", "Epoch", "Loss")
    plot_function(epochs, abs_err_train, "Absolute error on train", "Epoch", "Loss")
    plot_function(epochs, abs_err_valid, "Absolute error on test", "Epoch", "Loss")
    plot_function(epochs, relative_err_train, "Relative error on train %", "Epoch", "Loss")
    plot_function(epochs, relative_err_valid, "Relative error on test %", "Epoch", "Loss")

def make_report(nn_model, approximation_of_function, train_domain, valid_domain, true_analytical_solution):
    nn_returned_value_train = nn_model(train_domain)
    approximated_function_train = approximation_of_function(
        train_domain, nn_returned_value_train
    )
    true_solution_train = true_analytical_solution(train_domain)
    print(
        "Train absolute error: {}".format(
            MaxAbsoluteError(true_solution_train, approximated_function_train)
        )
    )
    print(
        "Train relative error: {} %".format(
            relative_error(true_solution_train, approximated_function_train)
        )
    )

    nn_returned_value_valid = nn_model(valid_domain)
    approximated_function_valid = approximation_of_function(
        valid_domain, nn_returned_value_valid
    )
    true_solution_valid = true_analytical_solution(valid_domain)

    print(
        "Valid absolute error: {}".format(
            MaxAbsoluteError(true_solution_valid, approximated_function_valid)
        )
    )
    print(
        "Valid relative error: {} %".format(
            relative_error(true_solution_valid, approximated_function_valid)
        )
    )
    plot_function(
        valid_domain,
        AbsoluteError(true_solution_valid, approximated_function_valid),
        "Absolute error: true sol - Approximation",
        "X",
        "Y"
    )

    plot_function(valid_domain, true_solution_valid, "True Solution", "X", "Y")
    plot_function(valid_domain, approximated_function_valid, "Approximation", "X", "Y")