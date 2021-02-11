from Utilities import *
from Metrics import *
from Plotting import plot_function




if __name__ == '__main__':
    train_domain, valid_domain = get_domain()

    nn_model = get_model(train_domain, valid_domain)


    nn_returned_value_train = nn_model(train_domain)
    approximated_function_train = approximation_of_function(train_domain,nn_returned_value_train)
    true_solution_train = true_analytical_solution(train_domain)
    print("Train absolute error: {}".format(MaxAbsoluteError(true_solution_train, approximated_function_train)))
    print("Train relative error: {}".format(relative_error(true_solution_train, approximated_function_train)))

    nn_returned_value_valid = nn_model(valid_domain)
    approximated_function_valid = approximation_of_function(valid_domain, nn_returned_value_valid)
    true_solution_valid = true_analytical_solution(valid_domain)

    print("Valid absolute error: {} %".format(MaxAbsoluteError(true_solution_valid, approximated_function_valid)))
    print("Valid relative error: {} %".format(relative_error(true_solution_valid, approximated_function_valid)))
    plot_function(valid_domain, AbsoluteError(true_solution_valid,approximated_function_valid),
                  "Absolute error: true sol - Approximation")

    plot_function(valid_domain,true_solution_valid, "True Solution")
    plot_function(valid_domain,approximated_function_valid, "Approximation")