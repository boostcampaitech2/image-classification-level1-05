import random

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset import *


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(imgs, labels, preds, n=16, shuffle=False):
    batch_size = imgs.shape[0]
    assert n <= batch_size

    choices = random.choices(
        range(batch_size), k=n) if shuffle else list(range(n))
    # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    figure = plt.figure(figsize=(12, 18 + 2))
    # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = labels[choice].item()
        pred = preds[choice].item()
        image = imgs[choice]

        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = '\n'.join([
            f'{task} - gt: {gt_label}, pred: {pred_label}'
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

def save_confusion_matrix(num_classes, labels, preds, save_path):
    confusion = confusion_matrix(y_true=labels, y_pred=preds, normalize='true')
    df = pd.DataFrame(
        confusion,
        index=list(range(num_classes)),
        columns=list(range(num_classes))
    )
    df = df.fillna(0)

    plt.figure(figsize=(10, 9))
    plt.tight_layout()
    plt.suptitle('Confusion Matrix')
    sns.heatmap(
        df, cmap=sns.color_palette("Blues"),
        annot=True, fmt='.2f',
        linewidth=0.1, square=True
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(save_path)
