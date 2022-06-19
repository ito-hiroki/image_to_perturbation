import argparse
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append("../lib/")
from resnet import resnet20
from vgg import vgg16_bn


def one_epoch(model, data_loader):
    model.eval()

    data_num = 0
    correct_num = 0

    for images, targets in tqdm(data_loader):
        images, targets = images.to(device), targets.to(device)
        data_num += len(targets)

        with torch.no_grad():
            logits = model(images)

        prediction = torch.argmax(logits, dim=1)
        correct_num += (prediction == targets).sum().item()

    return correct_num / data_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    # Constants
    DATA_PATH = "../data/"
    TEST_BATCH_SIZE = 256
    CHECKPOINT_FOLDER = "../model/classifier/"
    NUM_WORKER = 4
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    save_name = f"classifier_{args.model}_{args.dataset}.pth"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Reproducibility
    torch.manual_seed(100)
    random.seed(200)
    np.random.seed(300)
    cudnn.deterministic = True
    cudnn.benchmark = False

    fc_size = None
    if args.dataset == "cifar10":
        num_classes = 10
        fc_size = 512

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        test_ds = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=False, download=True, transform=transform_test)
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True)
    elif args.dataset == "cifar100":
        num_classes = 100
        fc_size = 512

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        test_ds = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=False, download=True, transform=transform_test)
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True)
    else:
        raise ValueError(f"dataset argment is invalid. {args.dataset}")

    if args.model == "resnet20":
        model = resnet20(in_channels=3, num_classes=num_classes).to(device)
    elif args.model == "vgg16":
        model = vgg16_bn(in_channels=3, num_classes=num_classes, fc_size=fc_size).to(device)
    else:
        raise ValueError(f"model argment is invalid. {args.model}")
    model.load_state_dict(torch.load(CHECKPOINT_FOLDER + save_name, map_location=device))

    accuracy = one_epoch(model, test_dl)
    print(f"test accuracy: {accuracy:.3%}")
