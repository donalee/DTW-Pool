import argparse
import os
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from model import FCN, ResNet
from timeseries import TimeSeriesWithLabels

def train(args, train_dataset, valid_dataset, test_dataset):

    batch_size = int(min(len(train_dataset)/10, args.batch_size))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if args.model == 'fcn':
        model = FCN(num_classes=train_dataset.num_classes, 
                    num_segments=args.n_segments, 
                    input_size=train_dataset.input_size, 
                    cost_type=args.cost_type, 
                    pooling_op=args.pooling_op, 
                    gamma=args.gamma)
    elif args.model == 'resnet':
        model = ResNet(num_classes=train_dataset.num_classes, 
                    num_segments=args.n_segments, 
                    input_size=train_dataset.input_size, 
                    cost_type=args.cost_type, 
                    pooling_op=args.pooling_op, 
                    gamma=args.gamma)
    model.cuda()

    ce = torch.nn.CrossEntropyLoss() 
    optim_h = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optim_p = torch.optim.Adam([model.protos], lr=args.learning_rate, weight_decay=args.weight_decay)
    
    loss_results, acc_results = [], []

    # Start Training
    model.init_protos(train_loader)
    for epoch in range(args.n_epochs):
        model.train()
        total_step = len(train_loader)
        total, total_ce_loss, total_dtw_loss = 0, 0, 0
       
        for itr, batch in enumerate(train_loader):
            data, labels = batch['data'].cuda(), batch['labels'].cuda()
            h, logits = model(data)

            ce_loss = ce(logits, labels)
            optim_h.zero_grad()
            ce_loss.backward(retain_graph=True)
            optim_h.step()

            dtw_loss = model.compute_aligncost(h)
            optim_p.zero_grad()
            dtw_loss.backward(retain_graph=True)
            optim_p.step()

            with torch.no_grad():
                total_ce_loss += ce_loss.item() * data.size(0)
                total_dtw_loss += dtw_loss.item() * data.size(0)
                total += data.size(0)
            
        train_loss = total_ce_loss / total
       
        with torch.no_grad():
            model.eval()
            correct, test_total = 0, 0
            for itr, batch in enumerate(test_loader):
                data, labels = batch['data'].cuda(), batch['labels'].cuda()
                _, logits = model(data)
                _, predicted = torch.max(logits, 1)
                test_total += data.size(0)
                correct += (predicted == labels).sum().item()
        
        print('\tEpoch [{:3d}/{:3d}], Train Loss: {:.4f}, {:.4f}, Test Accuracy: {:.4f}'
            .format(epoch+1, args.n_epochs, total_ce_loss/total, total_dtw_loss/total, correct/test_total))
        
        loss_results.append(total_ce_loss/total)
        acc_results.append(correct/test_total)
    
    print('The Best Test Accuracy: {:.4f}'.format(acc_results[loss_results.index(min(loss_results))]))

def main(args):

    train_dataset = TimeSeriesWithLabels(args.dataset, args.vartype, 'TRAIN') 
    valid_dataset = TimeSeriesWithLabels(args.dataset, args.vartype, 'TRAIN') 
    test_dataset = TimeSeriesWithLabels(args.dataset, args.vartype, 'TEST') 
    
    train(args, train_dataset, valid_dataset, test_dataset)

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuidx', default=0, type=int, help='gpu index')
    parser.add_argument('--dataset', default='GunPoint', type=str, help='target dataset')
    parser.add_argument('--vartype', default='univar', type=str, help='univar | multivar')
    parser.add_argument('--model', default='fcn', type=str, help='fcn | resnet')
    
    parser.add_argument('--cost_type', default='cosine', type=str, help='cosine | dotprod | euclidean')
    parser.add_argument('--pooling_op', default='max', type=str, help='avg | sum | max')
    
    parser.add_argument('--n_segments', default=4, type=int, help='# of segments')
    parser.add_argument('--gamma', default=1.0, type=float, help='smoothing for differentiable DTW')

    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--n_epochs', default=300, type=int, help='# of training epochs')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate') # 0.001 in original paper
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='l2 regularization')
    args = parser.parse_args()

    print(args)

    # GPU setting
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)
    
    # Random seed initialization
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    main(args=args)
