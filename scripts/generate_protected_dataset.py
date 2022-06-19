import argparse
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append("../lib/")
from unet import UNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--p_ratio", required=True, type=float)
    parser.add_argument("--original", action="store_true")
    args = parser.parse_args()

    # Constants
    DATA_PATH = "../data/"
    os.environ["TORCH_HOME"] = DATA_PATH  # pretrained model

    CHECKPOINT_FOLDER = "../model/transform/"
    NUM_WORKER = 0
    TEST_BATCH_SIZE = 512
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    # pretrained model folder
    os.environ["TORCH_HOME"] = DATA_PATH
    checkpoint_name = f"transform_{args.model}_{str(args.p_ratio).replace('.', '-')}_{args.classifier}_{args.dataset}.pth"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    mean = torch.FloatTensor(mean).view(3, 1, 1).to(device)
    std = torch.FloatTensor(std).view(3, 1, 1).to(device)

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    if args.dataset == "cifar10":
        num_classes = 10

        train_ds = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=True, transform=transform_test
        )
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=TEST_BATCH_SIZE, num_workers=0, pin_memory=True
        )

        test_ds = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=False, transform=transform_test
        )
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=TEST_BATCH_SIZE, num_workers=0, pin_memory=True
        )
    elif args.dataset == "cifar100":
        num_classes = 100

        train_ds = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=False, transform=transform_test
        )
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=TEST_BATCH_SIZE, num_workers=0, pin_memory=True
        )

        test_ds = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=False, transform=transform_test
        )
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=TEST_BATCH_SIZE, num_workers=0, pin_memory=True
        )
    else:
        raise ValueError(f"dataset argment is invalid. {args.dataset}")

    if args.model == "unet":
        model = UNet(img_ch=3, output_ch=3).to(device)
    else:
        raise ValueError(f"model argment is invalid. {args.model}")
    model.load_state_dict(torch.load(CHECKPOINT_FOLDER + checkpoint_name))
    for param in model.parameters():
        param.requires_grad = False
    model.to(device).eval()

    to_pil = torchvision.transforms.ToPILImage()

    model.eval()

    SAVE_TRANS = f"../dataset/protect/trans/{args.dataset}/{args.classifier}/{str(args.p_ratio).replace('.', '-')}/"
    SAVE_ORIGINAL = f"../dataset/origin/"

    for dl, data_type in zip([train_dl, test_dl], ["train", "test"]):
        idx = 0
        os.makedirs(SAVE_TRANS + data_type, exist_ok=True)
        os.makedirs(SAVE_ORIGINAL + data_type, exist_ok=True)

        for images, targets in tqdm(dl):
            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                protect_image = model(images)
                protect_image = torch.sigmoid(protect_image)

            for i in range(len(images)):
                tra = to_pil(protect_image[i].cpu())
                tra.save(SAVE_TRANS + data_type + f"/{idx:0>6}.png")
                if args.original:
                    img = to_pil(images[i].cpu())
                    img.save(SAVE_ORIGINAL + data_type + f"/{idx:0>6}.png")

                idx += 1
