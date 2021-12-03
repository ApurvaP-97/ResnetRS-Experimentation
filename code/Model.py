### YOUR CODE HERE

import torch
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import math
from typing import Iterable
import copy
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from Network import ResnetRS


"""This script defines the training, validation and testing process.
"""

class MyModel(object):
    
    def __init__(self):
        resnetRs = ResnetRS()
        self.configs = resnetRs._get_cfg(name='resnetrs50')
        self.network = resnetRs.create_model(self.configs['block'], self.configs['layers']).cuda()
        
    '''
    def model_setup(self):
        pass

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        pass

    def evaluate(self, x, y):
        pass

    def predict_prob(self, x):
        pass
    '''

### END CODE HERE

    def save_checkpoint(self, filename='./checkpoint.pth'):
        """Save checkpoint"""
        torch.save(self.network.state_dict(), filename)
    
    
    def get_state_dict(filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError("checkpoint file does not exist.")
    
        key = 'state_dict'
        checkpoint = torch.load(filename, map_location='cpu')
        if isinstance(checkpoint, dict):
            if key in checkpoint.keys():
                state_dict = checkpoint[key]
        else:
            state_dict = checkpoint
        return state_dict
    
    
    def load_checkpoint(self, filename, strict=True):
        state_dict = self.get_state_dict(filename)
        self.network.load_state_dict(state_dict, strict=strict)
    
      
    
    def trainacc(self, train_loader):
        ans=0
        for i, (ip, tgt) in enumerate(train_loader):
    
            output = self.network(ip.cuda())
    
            pred_exp = np.exp(torch.Tensor.cpu(output).detach())
            for i in range(len(tgt)):
                index=int(torch.argmax(pred_exp[i]))
                if(index==tgt[i]):
                    ans=ans+1
        print(ans/50000)
        return(ans/50000)
    
    
    def valacc(self, val_loader):
        ans=0
        for i, (ip, tgt) in enumerate(val_loader):
    
            output = self.network(ip.cuda())
    
            pred_exp = np.exp(torch.Tensor.cpu(output).detach())
            for i in range(len(tgt)):
                index=int(torch.argmax(pred_exp[i]))
                if(index==tgt[i]):
                    ans=ans+1
        print(ans/5000)
        return ans/5000
    
    
    
    def testacc(self, test_loader):
        ans=0
        for i, (ip, tgt) in enumerate(test_loader):
    
            output = self.network(ip.cuda())
    
            pred_exp = np.exp(torch.Tensor.cpu(output).detach())
            for i in range(len(tgt)):
                index=int(torch.argmax(pred_exp[i]))
                if(index==tgt[i]):
                    ans=ans+1
        print(ans/10000)
        return ans/10000
    
    def evaluate(self, test_loader):
        ans=0
        count=0
        self.network.training=False
        for i, (ip, tgt) in enumerate(test_loader):
    
            output = self.network(ip.cuda())
    
            pred_exp = np.exp(torch.Tensor.cpu(output).detach())
            for i in range(len(tgt)):
                count=count+1
                index=int(torch.argmax(pred_exp[i]))
                if(index==tgt[i]):
                    ans=ans+1
        self.network.training=True
        print(ans/count)
        return (ans/count)
    
    def pred_prob(self,data_loader):
        pred_res=[]
        self.network.training=False
        for j,image in enumerate(data_loader):
            a=[]
            for i in image:
                temp=np.transpose(torch.reshape(i, (32,32,3)).numpy(),[2,0,1])
                a.append(temp)
            output= self.network(torch.tensor(a).cuda().float())
            pred = np.exp(torch.Tensor.cpu(output).detach())
            for i in range(len(pred)):
                norm = pred[i]/torch.sum(pred[i])
                index=int(torch.argmax(norm))
                pred_res.append(norm.detach().numpy())
        self.network.training=True
        return pred_res
    
    
    '''Refer to Report Section 4.2 and 5.2 : CosineAnnealingWarmRestarts, LabelSmoothing criterion is implemented here'''
    def train(self, train_loader,test_loader, max_epoch,learning_rate,weight_decay,momentum):
        optimizer = torch.optim.SGD(self.network.parameters(), lr=learning_rate,weight_decay=weight_decay, momentum=momentum)
        #scheduler = MultiStepLR(optimizer, milestones=[40, 80, 120], gamma=0.2)
        it = len(train_loader)
        
        lrcos = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,50)
        criterion = LabelSmoothing() 
        ema = EMA(self.network.parameters(), decay_rate=0.995, num_updates=0)        
        for epoch in range(max_epoch):
            start_time = time.time()
            print(epoch)
            for i, (ip, tgt) in enumerate(train_loader):
                output = self.network(ip.cuda())
                loss = criterion(output, torch.tensor(tgt,dtype=torch.long).cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.update(self.network.parameters())
            print('LR reset of scheduler')
            lrcos.step(epoch + i / it)
            print('\nLearning rate: %0.9f' % lrcos.get_lr()[0])
            #scheduler.step() -- Uncomment if using MultistepLR  
            duration = time.time() - start_time    
            print('Epoch numer {:d} Loss value {:.6f} Time taken {:.3f} seconds.'.format(epoch, loss, duration))
            #Since coslr warm reset occurs after every 50 epochs, we have to save the models when the learning rate is the lowest.
            if epoch%10==9:
                  
                torch.save(self.network.state_dict(), 'model-%d.ckpt'%(epoch))
                self.trainacc(train_loader)
                self.evaluate(test_loader)
                
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        ll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        ll_loss = ll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * ll_loss + self.smoothing * smooth_loss
        return loss.mean()

class EMA:
    """
    Implementation of Exponential Moving Averages(EMA) for model parameters.

    Args:
        parameters: model parameters.
                    Iterable(torch.nn.Parameter)
        decay_rate: decay rate for moving average.
                    Value is between 0. and 1.0
        num_updates: Number of updates to adjust decay_rate.
    Reference for implementation : https://github.com/rwightman/pytorch-image-models
    """
    def __init__(self, parameters: Iterable[torch.nn.Parameter],
                 decay_rate: float,
                 num_updates: int = None) -> None:
        assert 0.0 <= decay_rate <= 1.0, \
               "Decay rate should be in range [0, 1]"
        parameters = list(parameters)
        self.decay_rate = decay_rate
        self.num_updates = num_updates
        self.shadow_params = [p.clone().detach() for p in parameters
                              if p.requires_grad]
        self.saved_params = parameters

    def update(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Update the EMA of parametes with current decay rate.
        """
        self.num_updates += 1
        if self.num_updates is not None:
            decay_rate = min(self.decay_rate,
                             (1+self.num_updates)/(10+self.num_updates))
        else:
            decay_rate = self.decay_rate

        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for shadow, param in zip(self.shadow_params, parameters):
                tmp = shadow - param
                tmp.mul_(1-decay_rate)
                shadow.sub_(tmp)

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Save the model parameters.
        """
        parameters = list(parameters)
        self.saved_params = [p.clone().detach()
                             for p in parameters
                             if p.requires_grad]

    def copy(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy the EMA parmeters to model parameters.
        """
        parameters = list(parameters)
        if len(parameters) != len(self.shadow_params):
            raise ValueError(
                "Number of parameters passed and number of shadow "
                "parameters does not match."
            )
        for param, shadow in zip(parameters, self.shadow_params):
            if param.requires_grad:
                param.data.copy_(shadow.data)

    def copy_back(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy the saved parameters to model parameters.
        """
        parameters = list(parameters)
        if self.saved_params is None:
            raise ValueError(
                "No saved parameters found."
            )

        if len(parameters) != len(self.saved_params):
            raise ValueError(
                "Number of parameters does not match with "
                "number of saved parameters."
            )

        for saved_param, param in zip(self.saved_params, parameters):
            if param.requires_grad:
                param.data.copy_(saved_param.data)

    def to(self, device='cpu', dtype=None) -> None:
        self.shadow_params = [p.to(device=device, dtype=dtype) for p
                              in self.shadow_params]
        if self.saved_params is not None:
            self.saved_params = [p.to(device=device, dtype=dtype) for p
                                 in self.saved_params]

    def state_dict(self) -> dict:
        return {
            "decay_rate": self.decay_rate,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "saved_params": self.saved_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = copy.deepcopy(state_dict)
        self.decay_rate = state_dict['decay_rate']
        assert 0.0 <= self.decay_rate <= 1.0, \
               "Decay rate should be in range [0, 1]"

        self.num_updates = state_dict['num_updates']
        assert self.num_updates is None or isinstance(self.num_updates, int),\
            "num updates should either be None or int."

        def validate_params(params1, params2=None):
            assert isinstance(params1, list), "Parameters must be list."
            for param in params1:
                assert isinstance(param, torch.Tensor), \
                    "Each parameter much be torch tensor."

            if params2 is not None:
                if len(params1) != len(params2):
                    raise ValueError("Parameter length mismatch.")

        self.shadow_params = state_dict['shadow_params']
        validate_params(self.shadow_params)

        self.saved_params = state_dict['saved_params']
        validate_params(self.saved_params, self.shadow_params)