from tqdm import tqdm
from CustomDataSet import CustomDataSet
from Parameters import batch_size, true_analytical_solution
from Plotting import plot_metrics
from Metrics import *


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
):
    TorchZero = torch.Tensor([[0.0]])
    train_dataloader = torch.utils.data.DataLoader(
        CustomDataSet(train_domain), batch_size=batch_size, shuffle=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        CustomDataSet(val_domain), batch_size=batch_size, shuffle=False
    )
    abs_err_train = []
    abs_err_valid = []
    relative_err_train = []
    relative_err_valid = []
    mse_loss_train = []
    mse_loss_valid = []
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
                mse_loss_train.append(epoch_loss)
                nn_preds = model(train_domain)
                func_preds = func_approximation(train_domain, nn_preds)
                relative_err_train.append(
                    relative_error(true_analytical_solution_train, func_preds)
                    .detach()
                    .numpy()
                )
                abs_err_train.append(
                    MaxAbsoluteError(true_analytical_solution_train, func_preds)
                    .detach()
                    .numpy()
                )
            #     writer.add_scalar("Loss train: ", epoch_loss, epoch)
            else:
                mse_loss_valid.append(epoch_loss)
                nn_preds = model(val_domain)
                func_preds = func_approximation(val_domain, nn_preds)
                relative_err_valid.append(
                    relative_error(true_analytical_solution_valid, func_preds)
                    .detach()
                    .numpy()
                )
                abs_err_valid.append(
                    MaxAbsoluteError(true_analytical_solution_valid, func_preds)
                    .detach()
                    .numpy()
                )
            #     writer.add_scalar("Loss validation: ", epoch_loss, epoch)

            print("{} Loss: {:.4f}".format(phase, epoch_loss), flush=True)
    # writer.close()
    plot_metrics(mse_loss_train, "MSE Loss on train set")
    plot_metrics(mse_loss_valid, "MSE Loss on validation set")
    plot_metrics(abs_err_train, "Absolute error on train")
    plot_metrics(abs_err_valid, "Absolute error on test")
    plot_metrics(relative_err_train, "Relative error on train %")
    plot_metrics(relative_err_valid, "Relative error on test %")

    model.eval()
    return model
