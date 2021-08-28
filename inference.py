import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MaskBaseDataset, TestDataset


def load_model(model_dir, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(num_classes=num_classes)

    model_path = os.path.join(model_dir, 'best.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model #hi


@torch.no_grad()
def inference(data_dir, model_dir, output_dir):
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes
    model = load_model(model_dir, num_classes, device).to(device)
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
        for images in loader:
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='ResNet18PretrainedL12Frozen', help='model type (default: ResNet18PretrainedL12Frozen)')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    inference(data_dir=args.data_dir, model_dir=args.model_dir, output_dir=args.output_dir)
