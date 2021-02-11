import matplotlib.pyplot as plt

def plot_function(domain, function, text):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(text)
    ax.set_xlabel("X")
    ax.set_ylabel("Function")
    ax.plot(domain.detach().numpy(),function.detach().numpy())
    plt.show()


def plot_metrics(metrics_data, text):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(text)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.plot(metrics_data)
    plt.show()