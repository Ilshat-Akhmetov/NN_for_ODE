from tqdm import tqdm
from CustomDataSet import CustomDataSet
from NN_train_parameters import batch_size
from Utilities import relative_error, MaxAbsoluteError
import torch


def train_model(
    model,
    loss,
    optimizer,
    scheduler,
    num_epochs,
    train_domain,
    val_domain,
    func_approximation,
    differential_operator,
    true_analytical_solution
):
    TorchZero = torch.Tensor([[0.0]])
    train_dataloader = torch.utils.data.DataLoader(
        CustomDataSet(train_domain), batch_size=batch_size, shuffle=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        CustomDataSet(val_domain), batch_size=batch_size, shuffle=False
    )
    abs_err_train = torch.zeros(num_epochs)
    abs_err_valid = torch.zeros(num_epochs)
    relative_err_train = torch.zeros(num_epochs)
    relative_err_valid = torch.zeros(num_epochs)
    mse_loss_train = torch.zeros(num_epochs)
    mse_loss_valid = torch.zeros(num_epochs)

    true_analytical_solution_train = true_analytical_solution(train_domain)
    true_analytical_solution_valid = true_analytical_solution(val_domain)

    model.train()
    for epoch in range(num_epochs):
        print("Epoch {}/{}:".format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                dataloader = train_dataloader
                # model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                # model.eval()  # Set model to evaluate mode
            running_loss = 0.0

            # Iterate over data.
            for inputs in tqdm(dataloader):
                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(True):
                    # with torch.set_grad_enabled(phase == 'train'):
                    nn_model_pred = model(inputs)
                    preds = func_approximation(inputs, nn_model_pred)

                    residual = differential_operator(preds, inputs)
                    loss_value = loss(residual, TorchZero)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss_value.backward(retain_graph=True)
                        optimizer.step()
                # statistics
                running_loss += loss_value.item()
            epoch_loss = running_loss / len(dataloader)
            if phase == "train":
                scheduler.step()
                mse_loss_train[epoch] = epoch_loss
                nn_preds = model(train_domain)
                func_preds = func_approximation(train_domain, nn_preds)
                relative_err_train[epoch] = relative_error(true_analytical_solution_train, func_preds).detach()
                abs_err_train[epoch] = MaxAbsoluteError(true_analytical_solution_train, func_preds).detach()
            #     writer.add_scalar("Loss train: ", epoch_loss, epoch)
            else:
                mse_loss_valid[epoch] = epoch_loss
                nn_preds = model(val_domain)
                func_preds = func_approximation(val_domain, nn_preds)
                relative_err_valid[epoch] = relative_error(true_analytical_solution_valid, func_preds).detach()
                abs_err_valid[epoch] = MaxAbsoluteError(true_analytical_solution_valid, func_preds).detach()
            #     writer.add_scalar("Loss validation: ", epoch_loss, epoch)

            print("{} Loss: {:.4f}".format(phase, epoch_loss), flush=True)
    # writer.close()

    model.eval()
    epochs = torch.arange(num_epochs)
    return model, epochs, mse_loss_train, mse_loss_valid, abs_err_train, abs_err_valid, relative_err_train, relative_err_valid
