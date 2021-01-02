import warnings
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
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


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

        try:
            os.makedirs('Output/' + str(epoch))
        except Exception as e:
            print(e)

        model.train()

        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.epochs - 1))

            for iteration, data in enumerate(train_dataloader):

                inputs = data['image']
                mask = data['mask']
                inputs = inputs.to(device)
                mask = mask.to(device)

                preds = model(inputs)

                loss = criterion(preds, mask)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                input = np.array(inputs.detach().cpu()[0]).transpose((1,2,0))
                preds = np.array(preds.detach().cpu()[0]).transpose((1,2,0))
                mask = np.array(mask.detach().cpu()[0]).transpose((1,2,0))

                input = input/np.max(input)
                input = np.clip(input, 0, 1)

                preds = preds/np.max(preds)
                preds = np.clip(preds, 0, 1)

                mask = mask/np.max(mask)
                mask = np.clip(mask, 0, 1)

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,15))
                ax1.imshow(input.astype(float))
                ax2.imshow(preds.astype(float))
                ax3.imshow(mask.astype(float))
                plt.savefig('Output/{}/{}.png'.format(epoch, iteration))

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

    

    