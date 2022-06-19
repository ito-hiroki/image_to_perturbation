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
from unet import UNet


def worker_init_fn(worker_id):
    random.seed(worker_id)


def one_epoch(model, classifier, data_loader, mean, std):
    model.eval()
    classifier.eval()

    data_num = 0
    correct_num = 0

    for images, targets in tqdm(data_loader):
        images, targets = images.to(device), targets.to(device)
        data_num += len(targets)

        with torch.no_grad():
            logits = model(images)
            logits = torch.sigmoid(logits)
            logits = (logits - mean) / std
            logits = classifier(logits)

        prediction = torch.argmax(logits, dim=1)
        correct_num += (prediction == targets).sum().item()

    return correct_num / data_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--p_ratio", required=True, type=float)
    args = parser.parse_args()

    # Constants
    DATA_PATH = "../data/"
    TEST_BATCH_SIZE = 256
    CHECKPOINT_FOLDER = "../model/transform/"
    BASELINE_FOLDER = "../model/classifier/"
    NUM_WORKER = 4
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_name = f"transform_{args.model}_{str(args.p_ratio).replace('.', '-')}_{args.classifier}_{args.dataset}.pth"
    baseline_name = f"classifier_{args.classifier}_{args.dataset}.pth"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean = torch.FloatTensor(mean).view(3, 1, 1).to(device)
    std = torch.FloatTensor(std).view(3, 1, 1).to(device)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Reproducibility
    torch.manual_seed(100)
    random.seed(200)
    np.random.seed(300)
    cudnn.deterministic = True
    cudnn.benchmark = False

    if args.dataset == "cifar10":
        num_classes = 10

        test_ds = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=False, transform=transform_test)
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True)
    elif args.dataset == "cifar100":
        num_classes = 100

        test_ds = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=False, transform=transform_test)
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True)
    else:
        raise ValueError(f"dataset argment is invalid. {args.dataset}")

    if args.model == "unet":
        model = UNet(img_ch=3, output_ch=3).to(device)
    else:
        raise ValueError(f"model argment is invalid. {args.model}")
    model.load_state_dict(torch.load(CHECKPOINT_FOLDER + save_name))
    for param in model.parameters():
        param.requires_grad = False
    model.to(device).eval()

    if args.classifier == "resnet20":
        classifier = resnet20(in_channels=3, num_classes=num_classes).to(device)
    elif args.classifier == "vgg16":
        classifier = vgg16_bn(in_channels=3, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"classifier argment is invalid. {args.classifier}")
    classifier.load_state_dict(torch.load(BASELINE_FOLDER + baseline_name))
    for param in classifier.parameters():
        param.requires_grad = False
    classifier.to(device).eval()

    accuracy = one_epoch(model, classifier, test_dl, mean, std)
    print(f"valid accuracy: {accuracy:.3%}")
