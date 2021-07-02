from NN_train_parameters import *
from Train import train_model
from ODE_parameters import *
from Utilities import *


if __name__ == "__main__":
    train_domain, valid_domain = get_domain(left_bound, right_bound, number_of_points)

    nn_model, epochs, mse_loss_train, mse_loss_valid, \
    abs_err_train, abs_err_valid, relative_err_train, relative_err_valid = train_model(
        model,
        LossMSE,
        optimizer,
        scheduler,
        num_epochs,
        train_domain,
        valid_domain,
        approximation_of_function,
        differential_operator,
        true_analytical_solution
    )
    plot_results(epochs, mse_loss_train, mse_loss_valid, abs_err_train, abs_err_valid, relative_err_train, relative_err_valid)
    make_report(nn_model, approximation_of_function, train_domain, valid_domain, true_analytical_solution)

