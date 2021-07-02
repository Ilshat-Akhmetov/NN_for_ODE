import torch.nn as nn

# class CustomResNet(nn.Module):
#     def __init__(self, hidden_neurons):
#         super().__init__()
#         self.Sequence = nn.Sequential(
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Tanh(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Tanh(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Tanh()
#         )

# class CustomClass(nn.Module):
#     def __init__(self, NumberOfInputs, hidden_neurons, NumberOfOutputs):
#         super().__init__()
#         self.SeqStart = nn.Sequential(
#             nn.Linear(NumberOfInputs, hidden_neurons),
#             nn.Tanh()
#         )
#         self.RN1 = CustomResNet(hidden_neurons)
#         self.RN2 = CustomResNet(hidden_neurons)
#         self.LinearOutput = nn.Linear(NumberOfInputs, NumberOfOutputs)
#     def forward(self,X):
#         X = self.SeqStart(X)
#         X1 = self.RN1(X) + X
#         X2 = self.RN2(X1) + X1
#         X3 = self.LinearOutput(X2) + X2
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
            nn.Tanh(),
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
