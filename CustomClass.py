import torch.nn as nn

# class CustomClass(nn.Module):
#     def __init__(self, NumberOfInputs,hidden_neurons,NumberOfOutputs):
#         super().__init__()
#         self.Sequence1 = nn.Sequential(
#             nn.Linear(NumberOfInputs,hidden_neurons),
#             nn.Tanh(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Sigmoid(),
#             nn.Linear(hidden_neurons, hidden_neurons)
#         )
#         self.Sequence2 = nn.Sequential(
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Sigmoid(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Tanh(),
#             nn.Linear(hidden_neurons, hidden_neurons)
#         )
#         self.Sequence3 = nn.Sequential(
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Tanh(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Sigmoid(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Tanh(),
#             nn.Linear(hidden_neurons, NumberOfOutputs)
#         )
#     def forward(self,X):
#         X1 = self.Sequence1(X)
#         X2 = self.Sequence2(X1)+X
#         X3 = self.Sequence3(X2)
#         return X3


class CustomClass(nn.Module):
    def __init__(self, NumberOfInputs, hidden_neurons, NumberOfOutputs):
        super().__init__()
        self.Sequence = nn.Sequential(
            nn.Linear(NumberOfInputs, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Sigmoid(),
            nn.Linear(hidden_neurons, NumberOfOutputs),
        )

    def forward(self, X):
        return self.Sequence(X)


# class CustomClass(nn.Module):
#     def __init__(self, NumberOfInputs,hidden_neurons,NumberOfOutputs):
#         super().__init__()
#         self.Sequence = nn.Sequential(
#             nn.Linear(NumberOfInputs,hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons,hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons,NumberOfOutputs)
#         )
#     def forward(self,X):
#         return self.Sequence(X)
