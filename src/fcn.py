import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BatchNorm1d as batchnorm
import numpy as np
torch.manual_seed(77)

class FCN(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, hiddens, means=np.array([]), stds=np.array([])):
        super(FCN, self).__init__()
        assert len(hiddens) > 0
        self.fc1 = nn.Linear(in_feat_dim, hiddens[0])
        self.fc2 = nn.Linear(hiddens[0], hiddens[1])

        self.means = Variable(torch.from_numpy(means), requires_grad=False).type(torch.FloatTensor)
        self.stds = Variable(torch.from_numpy(stds), requires_grad=False).type(torch.FloatTensor)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.means = self.means.cuda()
            self.stds = self.stds.cuda()

        self.out = nn.Linear(hiddens[-1], out_feat_dim)
        self.apply(weight_init) 

    def forward(self, x):
        if len(self.means) > 0 and len(self.stds) > 0:
            x = (x - self.means)/self.stds
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.out(x)
        
        return out

def weight_init(m): 
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)