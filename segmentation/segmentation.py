import os, sys, logging
import time
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import torch, monai
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from copy import deepcopy

# A class to apply random transforms on images and masks
# Transforms include affine transformation and normalization
class CustomTransform(object):
    def __init__(
            self,size=256,
            angle=10,
            translate=0.1,
            scale=0.1,
            shear=10,
            b_factor=0.3,
            c_factor=0.3,
            mean=0.5,
            std=0.5,
        ):
        self.size = size
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.b_factor = b_factor
        self.c_factor = c_factor
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, mask = data

        size = self.size
        angle = 2 * self.angle * random.random() - self.angle # angle in [-self.angle,self.angle]
        translate = (self.translate * random.random(), self.translate * random.random())
        scale = 2 * self.scale * random.random() + 1 - self.scale # scale in [1-self.scale,1+self.scale]
        shear = 2 * self.shear * random.random() - self.shear # shear in [-self.shear,self.shear]
        b_factor = 2 * self.b_factor * random.random() + 1 - self.b_factor # b_factor in [1-self.b_factor,1+self.b_factor]
        c_factor = 2 * self.c_factor * random.random() + 1 - self.c_factor # c_factor in [1-self.c_factor,1+self.c_factor]
        mean = self.mean
        std = self.std

        image = transforms.functional.resize(image, size, transforms.InterpolationMode.BILINEAR)
        image = transforms.functional.affine(image, angle, translate, scale, shear)
        image = transforms.functional.adjust_brightness(image, b_factor)
        image = transforms.functional.adjust_contrast(image, c_factor)
        image = image.float() / 255
        image = transforms.functional.normalize(image, [mean], [std])

        mask = transforms.functional.resize(mask, size, transforms.InterpolationMode.NEAREST)
        mask = transforms.functional.affine(mask, angle, translate, scale, shear)
        mask = mask.float() / 85 # {0,85,170,255} -> {0,1,2,3}; we have 4 classes
        mask = mask.long()

        return image, mask

class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, device='cpu'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.frames = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        image = read_image(self.image_dir + self.frames[index], mode=ImageReadMode.GRAY)
        mask = read_image(self.mask_dir + self.masks[index], mode=ImageReadMode.GRAY)

        if self.transform:
            image, mask = self.transform((image, mask))

        return image, mask


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="This is our segmentation script.")

    '''
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    '''

    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="The path to load the model from."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result-seg/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"
        ),
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size for the training dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=300)

    parser.add_argument(
        "--valid_batch_size", type=int, default=8, help="Batch size for the validation dataloader."
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate to use.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    
    parser.add_argument(
        "--train_frames",
        type=str,
        default='./data-echo-pat/train-frames-efn/',
        help=(
            "Train frames directory."
        ),
    )

    parser.add_argument(
        "--train_masks",
        type=str,
        default='./data-echo-pat/train-masks-efn/',
        help=(
            "Train masks directory."
        ),
    )

    parser.add_argument(
        "--valid_frames",
        type=str,
        default='./data-echo-pat/valid-frames-efn/',
        help=(
            "Valid frames directory."
        ),
    )

    parser.add_argument(
        "--valid_masks",
        type=str,
        default='./data-echo-pat/valid-masks-efn/',
        help=(
            "Valid masks directory."
        ),
    )

    '''
    parser.add_argument(
        "--save_image_epochs",
        type=int,
        default=15,
        help=(
            "Dummy save image epochs!"
        ),
    )
    '''

    parser.add_argument(
        "--val_interval",
        type=int,
        default=5,
        help=(
            "validation period!"
        ),
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
        help=(
            "Number of semantic classes!"
        ),
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    start_time = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # finding mean and std
    # ???
    
    train_trans = CustomTransform()
    valid_trans = CustomTransform(angle=0,translate=0,scale=0,shear=0,b_factor=0,c_factor=0,hflip=0)

    train_ds = SegDataset(args.train_frames, args.train_masks, transform=train_trans)
    valid_ds = SegDataset(args.valid_frames, args.valid_masks, transform=valid_trans)

    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, pin_memory=torch.cuda.is_available())
    valid_dl = DataLoader(valid_ds, batch_size=args.valid_batch_size, pin_memory=torch.cuda.is_available())

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=args.num_classes, # number of labels including background
        channels=(2, 4, 8, 16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2, 2, 2, 2),
        num_res_units=0,
        act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}),
    ).to(device)

    
    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))
        print('Model loaded from {}'.format(args.load_model_path))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.num_train_epochs,eta_min=0.0001)

    logs = {
        'epoch': [],
        'loss': [],
        'dice_mean': [],
        'dice_std': [],
        'dice_metric': [],
        'val_loss': [],
        # test_loss, test_dice_mean
    }

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    with open(args.output_dir + 'args.pkl','wb') as f:
        pickle.dump(args, f)

    # -------------------------------------------------
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    #loss_func = monai.losses.DiceLoss(sigmoid=True, weight=[0.01,1.8,1,1.3])
    # run-5: [0.01,0.23,0.38,0.38]; run-6: [0.01,0.11,0.44,0.44]; run-7: [0.01,0.11,0.40,0.48]; 
    # run-8,9: [0.01,0.11,0.35,0.53], run-10,11,12: [0.05,0.25,0.35,0.35]
    loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.05,0.25,0.35,0.35]).to(device)) # W
    dice_metric = DiceMetric(include_background=False,reduction="mean")

    iter = args.num_train_epochs
    for ep in range(iter):
        print("-" * 10)
        print(f"epoch {ep + 1}/{iter}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch in train_dl:
            step += 1

            xb, yb = batch[0].to(device), batch[1].to(device)
            yb = F.one_hot(yb.long(), args.num_classes).squeeze().permute(0, 3, 1, 2)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_func(y_pred, yb.float())
            loss.backward()

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_dl.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        epoch_loss /= step
        print(f"epoch {ep + 1} average loss: {epoch_loss:.4f}")

        if (ep + 1) % val_interval == 0:
            logs['loss'].append(epoch_loss)
            logs['epoch'].append(ep + 1)
            model.eval()
            with torch.no_grad():
                step = 0
                dice_scores = []
                for batch in valid_dl:
                    step += 1

                    xb, yb = batch[0].to(device), batch[1].to(device)
                    yb = F.one_hot(yb.long(), args.num_classes).squeeze().permute(0, 3, 1, 2)

                    y_pred = model(xb)
                    loss = loss_func(y_pred, yb.float())
                    epoch_loss += loss.item()

                    # may need to change this part
                    #y_pred = torch.cat([torch.unsqueeze(post_trans(i),dim=0) for i in decollate_batch(y_pred)])
                    val_outputs = F.one_hot(torch.argmax(y_pred,dim=1), num_classes=args.num_classes).permute(0,3,1,2)
                    metric = dice_metric(val_outputs, yb)
                    dice_scores.append(metric)

                epoch_loss /= step
                logs['val_loss'].append(epoch_loss)

                dice_scores = torch.cat(dice_scores, dim=0)
                s = tuple(torch.std(dice_scores,dim=0).cpu().numpy())
                m = tuple(torch.mean(dice_scores,dim=0).cpu().numpy())
                logs['dice_mean'].append(m)
                logs['dice_std'].append(s)
                epoch_metric = torch.mean(dice_scores).item()
                logs['dice_metric'].append(epoch_metric)

                if epoch_metric > best_metric:
                    best_metric = epoch_metric
                    best_metric_epoch = ep + 1
                    torch.save(model.state_dict(), args.output_dir + f"best_metric_model_segmentation2d.pth")
                    print("saved new best metric model")

                print("current epoch: {} current dice loss: {:.4f} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            ep + 1, epoch_loss, epoch_metric, best_metric, best_metric_epoch))
                print(f"class mean dice: {m} class std dice: {s}")

                with open(args.output_dir + "logs.pkl", 'wb') as f:
                    pickle.dump(logs, f)
    end_time = time.time()
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    print("Elapsed time: {:.2f} mins.".format((end_time-start_time)/60))
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
