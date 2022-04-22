import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple, deque


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 100
batch_size = 64
learning_rate = 1e-3
gamma = 0.99
class Q_network(nn.Module):
    def __init__(self):
        super(Q_network, self).__init__()
        self.fc1 = nn.Linear(8,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,4)
        
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
#loss_fn   = nn.MSELoss()


def give_action(model, state):
    state = torch.tensor(state, device=device, dtype=torch.float32).reshape(1,-1)
    return model(state).max(1)[1].view(1,1).item()
    
    
def train_step(model,transitions):  

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    loss_fn   = nn.SmoothL1Loss()
    
    samples = random.sample(transitions, batch_size)
    batch   = Transition(*zip(*samples))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
                                 
    non_final_next_states = torch.tensor([s for s in batch.next_state
                                                if s is not None], device=device, dtype=torch.float32)
    
    state_batch = torch.tensor(batch.state, dtype=torch.float32, device=device)

    action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).reshape(-1,1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
   
    #nextS_batch  = torch.tensor(batch.next_state, dtype=torch.float32, device=device)

    qValue  = model(state_batch).gather(1, action_batch)
    qValue1 = torch.zeros(batch_size, device=device)
    qValue1[non_final_mask] = model(non_final_next_states).max(1)[0].detach()

    
    expectedV = gamma*qValue1 + reward_batch
    
    loss = loss_fn(qValue, expectedV.unsqueeze(1))
    
    #Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1,1)
    optimizer.step()
    return expectedV.unsqueeze(1) - qValue
        
    
