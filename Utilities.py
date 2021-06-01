from Train import train_model
from Parameters import *
from Metrics import LossMSE


def get_domain():
    train_x = torch.linspace(
        left_bound, right_bound, number_of_points, requires_grad=True
    )
    train_x = train_x.unsqueeze(dim=1)
    valid_x = torch.cat(
        (
            train_x[0:1],
            train_x[1:9] + 1 / (2 * number_of_points),
            train_x[number_of_points - 1 : number_of_points],
        ),
        dim=0,
    )
    return train_x, valid_x


def get_model(train_domain, valid_domain):
    train_model(
        model,
        LossMSE,
        optimizer,
        scheduler,
        num_epochs,
        train_domain,
        valid_domain,
        approximation_of_function,
        differential_operator,
    )
    return model
