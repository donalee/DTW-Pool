import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f
from softdtw_cuda import SoftDTW

class FCN(nn.Module):
    def __init__(self, num_classes, num_segments, input_size, hidden_sizes=[128, 256, 128], kernel_sizes=[9, 5, 3], cost_type='cosine', pooling_op='avg', gamma=1.0):
        super(FCN, self).__init__()

        self.num_classes = num_classes
        self.num_segments = num_segments
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.kernel_sizes = kernel_sizes
        self.pooling_op = pooling_op
        self.softdtw = SoftDTW(use_cuda=True, gamma=gamma, cost_type=cost_type, normalize=False)
       
        self._build_model(num_classes, num_segments, input_size, hidden_sizes, kernel_sizes)
        self._init_model()

    def _build_model(self, num_classes, num_segments, input_size, hidden_sizes, kernel_sizes):

        self.conv1 = nn.Conv1d(in_channels=input_size,
                                 out_channels=hidden_sizes[0],
                                 kernel_size=kernel_sizes[0])

        self.conv2 = nn.Conv1d(in_channels=hidden_sizes[0],
                                 out_channels=hidden_sizes[1],
                                 kernel_size=kernel_sizes[1])
        
        self.conv3 = nn.Conv1d(in_channels=hidden_sizes[1],
                                 out_channels=hidden_sizes[2],
                                 kernel_size=kernel_sizes[2])
        
        self.norm1 = nn.BatchNorm1d(num_features=hidden_sizes[0])
        self.norm2 = nn.BatchNorm1d(num_features=hidden_sizes[1])
        self.norm3 = nn.BatchNorm1d(num_features=hidden_sizes[2])

        self.fc = nn.Linear(hidden_sizes[2]*num_segments, num_classes)
        self.protos = nn.Parameter(torch.zeros(hidden_sizes[2], num_segments), requires_grad=True)
     
    def _init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
    
    def init_protos(self, data_loader):
        for itr, batch in enumerate(data_loader):
            data = batch['data'].cuda()
            h = self.get_htensor(data)
            self.protos.data += self.stpool(h, 'avg').mean(dim=0)
            
        self.protos.data /= len(data_loader)
    
    def get_htensor(self, x):
        h = x
        h = f.pad(h, (int(self.kernel_sizes[0]/2), int(self.kernel_sizes[0]/2)), "constant", 0)
        h = f.relu(self.norm1(self.conv1(h)))
        
        h = f.pad(h, (int(self.kernel_sizes[1]/2), int(self.kernel_sizes[1]/2)), "constant", 0)
        h = f.relu(self.norm2(self.conv2(h)))
        
        h = f.pad(h, (int(self.kernel_sizes[2]/2), int(self.kernel_sizes[2]/2)), "constant", 0)
        h = f.relu(self.norm3(self.conv3(h)))
        return h
    
    # global temporal pooling
    def gtpool(self, h, op):
        if op == 'avg':
            return torch.mean(h, dim=2)
        if op == 'sum':
            return torch.sum(h, dim=2)
        elif op == 'max':
            return torch.max(h, dim=2)[0]

    # static temporal pooling
    def stpool(self, h, op):
        segment_sizes = [int(h.shape[2]/self.num_segments)] * self.num_segments
        segment_sizes[-1] += h.shape[2] - sum(segment_sizes)
       
        hs = torch.split(h, segment_sizes, dim=2)
        if op == 'avg':
            hs = [h_.mean(dim=2, keepdim=True) for h_ in hs]
        elif op == 'sum':
            hs = [h_.sum(dim=2, keepdim=True) for h_ in hs]
        elif op == 'max':
            hs = [h_.max(dim=2)[0].unsqueeze(dim=2) for h_ in hs]
        hs = torch.cat(hs, dim=2)
        return hs
    
    # dynamic temporal pooling
    def dtpool(self, h, op):
        A = self.softdtw.align(self.protos.repeat(h.shape[0], 1, 1), h)

        if op == 'avg':
            A /= A.sum(dim=2, keepdim=True)
            h = torch.bmm(h, A.transpose(1, 2))
        elif op == 'sum':
            h = h.unsqueeze(dim=2) * A.unsqueeze(dim=1)
            h = h.sum(dim=3)
        elif op == 'max':
            h = h.unsqueeze(dim=2) * A.unsqueeze(dim=1)
            h = h.max(dim=3)[0]
        return h

    def compute_gradcam(self, x, labels):
        
        def hook_func(grad):
            self.h_grad = grad

        h, logits = self.forward(x)
        h.register_hook(hook_func)

        self.zero_grad()
        scores = torch.gather(logits, 1, labels.unsqueeze(dim=1))
        scores.mean().backward()
        gradcam = (h * self.h_grad).sum(dim=1, keepdim=True)

        # min-max normalization
        gradcam_min = torch.min(gradcam, dim=2, keepdim=True)[0]
        gradcam_max = torch.max(gradcam, dim=2, keepdim=True)[0]
        gradcam = (gradcam - gradcam_min) / (gradcam_max - gradcam_min) 

        A = self.softdtw.align(self.protos.unsqueeze(dim=0).repeat(h.shape[0], 1, 1), h).sum(dim=2)
        
        return gradcam, A

    def forward(self, x):
        h = self.get_htensor(x)
        op = self.pooling_op
        out = self.fc(self.dtpool(h, op).reshape(h.shape[0], -1))
        
        return h, out

    def compute_aligncost(self, h):
        cost = self.softdtw(self.protos.repeat(h.shape[0], 1, 1), h.detach())
        return cost.mean() / h.shape[2]


class ResidualBlock(nn.Module):
    """
    Args:
        hidden_size: input dimension (Channel) of 1d convolution
        output_size: output dimension (Channel) of 1d convolution
        kernel_size: kernel size
    """
    def __init__(self, input_size, output_size, kernel_sizes=[9, 5, 3]):
        super(ResidualBlock, self).__init__()
        self.kernel_sizes = kernel_sizes

        self.conv1 = nn.Conv1d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_sizes[0])
        self.conv2 = nn.Conv1d(in_channels=output_size,
                                out_channels=output_size,
                                kernel_size=kernel_sizes[1])
        self.conv3 = nn.Conv1d(in_channels=output_size,
                                out_channels=output_size,
                                kernel_size=kernel_sizes[2])
        self.conv_skip = nn.Conv1d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=1)
        
        self.norm1 = nn.BatchNorm1d(num_features=output_size)
        self.norm2 = nn.BatchNorm1d(num_features=output_size)
        self.norm3 = nn.BatchNorm1d(num_features=output_size)
        self.norm_skip = nn.BatchNorm1d(num_features=output_size)

    def forward(self, x):
        
        h = x
        h = f.pad(h, (int(self.kernel_sizes[0]/2), int(self.kernel_sizes[0]/2)), "constant", 0)
        h = f.relu(self.norm1(self.conv1(h)))
       
        h = f.pad(h, (int(self.kernel_sizes[1]/2), int(self.kernel_sizes[1]/2)), "constant", 0)
        h = f.relu(self.norm2(self.conv2(h)))
        
        h = f.pad(h, (int(self.kernel_sizes[2]/2), int(self.kernel_sizes[2]/2)), "constant", 0)
        h = self.norm3(self.conv3(h))
        
        s = self.norm_skip(self.conv_skip(x))
        h += s
        h = f.relu(h)

        return h

class ResNet(nn.Module):
    def __init__(self, num_classes, num_segments, input_size, hidden_sizes=[64, 128, 128], kernel_sizes=[9, 5, 3], cost_type='cosine', pooling_op='avg', gamma=1.0):
        super(ResNet, self).__init__()

        self.num_classes = num_classes
        self.num_segments = num_segments 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.kernel_sizes = kernel_sizes
        self.pooling_op = pooling_op
        self.softdtw = SoftDTW(use_cuda=True, gamma=gamma, cost_type=cost_type, normalize=False)
        
        self._build_model(num_classes, num_segments, input_size, hidden_sizes, kernel_sizes)
        self._init_model()

    def _build_model(self, num_classes, num_segments, input_size, hidden_sizes, kernel_sizes):
            
        self.resblock1 = ResidualBlock(input_size=input_size,
                                        output_size=hidden_sizes[0],
                                        kernel_sizes=kernel_sizes)
        
        self.resblock2 = ResidualBlock(input_size=hidden_sizes[0],
                                        output_size=hidden_sizes[1],
                                        kernel_sizes=kernel_sizes)
        
        self.resblock3 = ResidualBlock(input_size=hidden_sizes[1],
                                        output_size=hidden_sizes[2],
                                        kernel_sizes=kernel_sizes)


        self.fc = nn.Linear(hidden_sizes[2]*num_segments, num_classes)
        self.protos = nn.Parameter(torch.zeros(hidden_sizes[2], num_segments), requires_grad=True)
     
    def _init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
    
    def init_protos(self, data_loader):
        for itr, batch in enumerate(data_loader):
            data = batch['data'].cuda()
            h = self.get_htensor(data)
            self.protos.data += self.stpool(h, 'avg').mean(dim=0)
            
        self.protos.data /= len(data_loader)
    
    def get_htensor(self, x):
        h = self.resblock1(x)
        h = self.resblock2(h)
        h = self.resblock3(h)
        return h
    
    # global temporal pooling
    def gtpool(self, h, op):
        if op == 'avg':
            return torch.mean(h, dim=2)
        if op == 'sum':
            return torch.sum(h, dim=2)
        elif op == 'max':
            return torch.max(h, dim=2)[0]

    # static temporal pooling
    def stpool(self, h, op):
        segment_sizes = [int(h.shape[2]/self.num_segments)] * self.num_segments
        segment_sizes[-1] += h.shape[2] - sum(segment_sizes)

        hs = torch.split(h, segment_sizes, dim=2)
        if op == 'avg':
            hs = [h_.mean(dim=2, keepdim=True) for h_ in hs]
        if op == 'sum':
            hs = [h_.sum(dim=2, keepdim=True) for h_ in hs]
        elif op == 'max':    
            hs = [h_.max(dim=2)[0].unsqueeze(dim=2) for h_ in hs]
        hs = torch.cat(hs, dim=2)
        return hs

    # dynamic temporal pooling
    def dtpool(self, h, op):
        A = self.softdtw.align(self.protos.repeat(h.shape[0], 1, 1), h)
        
        if op == 'avg':
            A /= A.sum(dim=2, keepdim=True)
            h = torch.bmm(h, A.transpose(1, 2))
        elif op == 'sum':
            h = h.unsqueeze(dim=2) * A.unsqueeze(dim=1)
            h = h.sum(dim=3)
        elif op == 'max':
            h = h.unsqueeze(dim=2) * A.unsqueeze(dim=1)
            h = h.max(dim=3)[0]
        return h
    
    def compute_gradcam(self, x, labels):
        
        def hook_func(grad):
            self.h_grad = grad

        h, logits = self.forward(x)
        h.register_hook(hook_func)

        self.zero_grad()
        scores = torch.gather(logits, 1, labels.unsqueeze(dim=1))
        scores.mean().backward()
        gradcam = (h * self.h_grad).sum(dim=1, keepdim=True)

        # min-max normalization
        gradcam_min = torch.min(gradcam, dim=2, keepdim=True)[0]
        gradcam_max = torch.max(gradcam, dim=2, keepdim=True)[0]
        gradcam = (gradcam - gradcam_min) / (gradcam_max - gradcam_min) 

        A = self.softdtw.align(self.protos.unsqueeze(dim=0).repeat(h.shape[0], 1, 1), h).sum(dim=2)

        return gradcam, A
    
    def forward(self, x):
        h = self.get_htensor(x) 
        op = self.pooling_op
        out = self.fc(self.dtpool(h, op).reshape(h.shape[0], -1))
        
        return h, out
    
    def compute_aligncost(self, h):
        cost = self.softdtw(self.protos.repeat(h.shape[0], 1, 1), h.detach())
        return cost.mean() / h.shape[2]

