import torch
from CustomClass import CustomClass
import numpy as np
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# equation_rest = equation_rest.to(device)

num_epochs = 30
batch_size = 1

HiddenNeurons = 200
number_of_inputs = 1
number_of_outputs = 1

model = CustomClass(number_of_inputs, HiddenNeurons, number_of_outputs)

# test_size = 0.3  # 30 % of data belongs to the test domain
# model.to(device)

number_of_points = 20

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=1)

# utility calculating function's derivative



