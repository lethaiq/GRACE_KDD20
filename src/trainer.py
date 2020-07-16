from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

        
class Trainer:
    def __init__(self, 
                 model, 
                 lrate = 0.01, 
                 lr_reduce_rate=1.0):

        self.use_cuda = torch.cuda.is_available()
        self.model = model
        self.lr_reduce_rate = lr_reduce_rate
        if self.use_cuda:
            self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lrate)
        self.criterion = nn.CrossEntropyLoss()


    def train(self, 
              train_dataset, 
              val_dataset=None, 
              patience=0, 
              num_epochs=10, 
              batch_size=256):

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        non_improve_count = 0
        best_val_loss = np.inf
        
        print("Training...")
        for epoch in range(num_epochs):
            acc = []
            running_loss = []
            self.model.train()
            for i, (X, y) in enumerate(train_loader):

                X = Variable(X).type(torch.FloatTensor)
                y = Variable(y).type(torch.LongTensor)

                if self.use_cuda:
                    X = X.cuda()
                    y = y.cuda()

                # forward pass
                out = self.model(X)
                loss = self.criterion(out, y)

                # backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # metrics 
                running_loss.append(loss.item())
                pred = torch.max(out, 1)[1].data.cpu().numpy()
                acc.append(accuracy_score(pred, y.data.cpu().numpy()))

                running_loss = []
                acc = []

            if val_dataset != None:
                loss, acc, auc, _ = self.validate(val_dataset, batch_size=batch_size)
                if epoch % 10 == 0:
                    print('Val loss: {:.4f} Val acc: {:.4f}'.format(loss, acc))

                if best_val_loss - loss > 0.001:
                    best_val_loss = loss
                    non_improve_count = 0
                else:
                    non_improve_count += 1

                if non_improve_count == patience:
                    break

                if non_improve_count == int(patience/2):
                    # print("reducing LR")
                    for g in self.optimizer.param_groups:
                        g['lr'] = g['lr']*self.lr_reduce_rate


    def predict_proba(self, sample, return_proba=False):
        x = Variable(torch.from_numpy(sample)).type(torch.FloatTensor)
        if self.use_cuda:
            x = x.cuda()
        self.model.eval()
        f = self.model.forward(Variable(x, requires_grad=True)).data.cpu().numpy().flatten()
        return f


    def predict(self, sample):
        x = Variable(torch.from_numpy(sample)).type(torch.FloatTensor)
        if self.use_cuda:
            x = x.cuda()

        self.model.eval()
        f = self.model.forward(Variable(x, requires_grad=True)).data.cpu().numpy().flatten()
        I = f.argsort()[::-1]
        fk_hat = I[0]
        return fk_hat


    def validate(self, test_dataset, batch_size=100):
        self.model.eval()
        with torch.no_grad():
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            preds = []
            running_loss = []
            for i, (X, y) in enumerate(test_loader):
                X = Variable(X).type(torch.FloatTensor)
                y = Variable(y).type(torch.LongTensor)

                if self.use_cuda:
                    X = X.cuda()
                    y = y.cuda()

                out = self.model(X)
                _, pred = torch.max(out.cpu().data, 1)
                loss = self.criterion(out, y)
                preds.append(pred.cpu().data)
                running_loss.append(loss.cpu().data)

            pred = np.concatenate(preds)
            acc = accuracy_score(test_dataset.y, pred)
            f1 = f1_score(test_dataset.y, pred, average="weighted")
            return np.mean(running_loss), acc, f1, pred
