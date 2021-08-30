import argparse
import json
import multiprocessing
import os
from importlib import import_module

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from torchvision import models

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from loss import get_criterion
import settings
import logger

def train(helper):
    args = helper.args
    device = helper.device
    is_cuda = helper.device == torch.device('cuda')

    Dataset = getattr(import_module("dataset"), args.dataset)
    dataset = Dataset(
        data_dir=args.data_dir,
        mean=(0.56019358, 0.52410121, 0.501457),
        std=(0.23318603, 0.24300033, 0.24567522)
    )
    num_classes = dataset.num_classes

    Transform = getattr(import_module("transform"), args.transform)
    transform = Transform(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    train_set, val_set = dataset.split_dataset(val_size=0.2)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=is_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=is_cuda,
        drop_last=True,
    )

    Model = getattr(import_module("model"), args.model)
    model = Model(num_classes=num_classes, freeze=args.freeze).to(device)
    model = torch.nn.DataParallel(model)

    criterion = get_criterion(args.criterion)
    Optimizer = getattr(import_module('torch.optim'), args.optimizer)
    optimizer = Optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    save_dir = helper.get_save_dir(dump=args.dump)
    writer = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, f'{args.model_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    best_f1 = 0
    for epoch in range(1, args.epochs + 1):
        loss_value = 0
        matches = 0
        accumulated_f1 = 0
        iter_count = 0

        for idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outs = model(imgs)
            preds = torch.argmax(outs, dim=1)
            loss = criterion(outs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).float().mean().item()
            accumulated_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            iter_count += 1
            
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.log_interval
                train_f1 = accumulated_f1 / iter_count
                current_lr = logger.get_lr(optimizer)
                print(
                    f'Epoch: {epoch:0{len(str(args.epochs))}d}/{args.epochs} '
                    f'[{idx + 1:0{len(str(len(train_loader)))}d}/{len(train_loader)}]\n'
                    f'training accuracy: {train_acc:>3.2%}\ttraining loss: {train_loss:>4.4f}\ttraining f1: {train_f1:>4.4f}\tlearning rate: {current_lr}\n'
                )
                writer.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                writer.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                writer.add_scalar("Train/f1", train_f1, epoch * len(train_loader) + idx)   

                loss_value = 0
                matches = 0

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss_items = []
            val_acc_items = []
            val_f1_items = []
        
            figure = None
            for val_batch in tqdm(val_loader, colour='GREEN'):
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).float().sum().item()
                f1_item = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                val_f1_items.append(f1_item)

                if figure is None:
                    imgs = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    imgs = Dataset.denormalize_image(imgs, dataset.mean, dataset.std)
                    figure = logger.grid_image(
                        imgs=imgs, labels=labels, preds=preds,
                        n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            val_f1 = np.average(val_f1_items) 
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:3.2f}%! saving the best model..")
                torch.save(model, os.path.join(save_dir, f'{args.model_name}acc.pt'))
                best_val_acc = val_acc
            if val_f1 > best_f1:
                print(f"New best model for f1 : {val_f1:3.2f}%! saving the best model..")
                torch.save(model, os.path.join(save_dir, f'{args.model_name}f1.pt'))
                best_f1 = val_f1
            # torch.save(model.module.state_dict(), os.path.join(save_dir, 'last.pt'))
            print(
                f'Validation:\n'
                f'accuracy: {val_acc:>3.2%}\tloss: {val_loss:>4.2f}\tf1: {val_f1:>4.2f}\n'
                f'best acc : {best_val_acc:>3.2%}\tbest loss: {best_val_loss:>4.2f}\n'
            )
            writer.add_scalar("Val/loss", val_loss, epoch)
            writer.add_scalar("Val/accuracy", val_acc, epoch)
            writer.add_scalar("Val/f1", val_f1, epoch)
            writer.add_figure("results", figure, epoch)
        model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset transform type (default: MaskBaseDataset)')
    parser.add_argument('--transform', type=str, default='BaseTransform', help='data transform type (default: BaseTransform)')
    parser.add_argument("--resize", nargs="+", type=list, default=(128, 96), help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000, help='input batch size for validation (default: 1000)')
    parser.add_argument('--model', type=str, default='ResNet18Pretrained', help='model type (default: ResNet18Pretrained)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', type=str, default='exp', help='model to save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--mode', type=str, default='all', help='select mask, age, gender, all')
    parser.add_argument('--model_name', type=str, default='best', help='custom model name')
    parser.add_argument('--freeze', nargs='+', default =[], help='layers to freeze (default: [])')
    parser.add_argument('--dump', type=bool, default=False, help="choose dump or not to save model")
    args = parser.parse_args()
    print(args)

    helper = settings.SettingsHelper(
        args=args,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    train(helper=helper)
