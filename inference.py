import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MaskBaseDataset, TestDataset
from tqdm import tqdm

def load_model(model_dir, device, model_name):
    model_path = os.path.join(model_dir, model_name)
    model = torch.load(model_path, map_location=device)

    return model 


@torch.no_grad()
def inference(data_dir, model_dir, output_dir):
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    model = load_model(model_dir, device, args.model_name).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=is_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for images in tqdm(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


@torch.no_grad()
def inference_with_ensemble(data_dir, model_dir, output_dir):
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    age_model = load_model(model_dir, device, 'agef1.pt').to(device)
    gender_model = load_model(model_dir, device, 'genderf1.pt').to(device)
    mask_model  = load_model(model_dir, device, 'maskf1.pt').to(device)
    age_model.eval()
    gender_model.eval()
    mask_model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=is_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for images in tqdm(loader):

            images = images.to(device)
            pred = mask_model(images)
            pred_mask = pred.argmax(dim=-1)
            
            pred = age_model(images)
            pred_age = pred.argmax(dim=-1)
            
            pred = gender_model(images)
            pred_gender = pred.argmax(dim=-1)

            result = pred_mask * 6 + pred_gender * 3 + pred_age
            preds.extend(result.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'{args.name}_output.csv'), index=False)
    print(f'Inference Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--model_name', type=str, default='best.pt')

    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--mode', type=str, default='all', help='choose all or ensemble')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'all':
        inference(data_dir=args.data_dir, model_dir=os.path.join(args.model_dir, args.name), output_dir=args.output_dir)
    elif args.mode == 'ensemble':
        inference_with_ensemble(data_dir=args.data_dir, model_dir=os.path.join(args.model_dir, args.name), output_dir=args.output_dir)
