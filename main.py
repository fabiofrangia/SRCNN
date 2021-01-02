import argparse
import os
import torch
from torch import nn
from tqdm import tqdm
from models import SRCNN
import torch.backends.cudnn as cudnn 
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from dataset import BasicDataset
from utils import AverageMeter
from torchvision import transforms, utils

TRAIN_DIR = 'Data/input/'
MASK_DIR = 'Data/output/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    train_dataset = BasicDataset(TRAIN_DIR, MASK_DIR,transform= transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])
                                                                            ]))
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last = True)

    
    for epoch in range(args.epochs):
        model.train()

        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.epochs - 1))

            for data in train_dataloader:

                inputs = data['image']
                mask = data['mask']
                inputs = inputs.to(device)
                mask = mask.to(device)

    

    