import torch


LossMSE = torch.nn.MSELoss()
AbsoluteError = lambda true_solution, approximation: torch.abs(true_solution - approximation)
MaxAbsoluteError =  lambda true_solution, approximation: torch.max(AbsoluteError(true_solution, approximation))

# def relative_error(true_sol, approximation):
#     return 100 * torch.sum(AbsoluteError(true_sol,approximation))/torch.sum(torch.abs(true_sol))

## calctulate relative error excluding cases when solution equals zero
def relative_error(true_sol, approximation):
     not_null_condition = true_sol!=0
     return 100 * torch.max(torch.abs((true_sol[not_null_condition]-approximation[not_null_condition])
                                      /true_sol[not_null_condition]))