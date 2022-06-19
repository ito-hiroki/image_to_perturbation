import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append("../lib/")
from resnet import resnet20
from vgg import vgg16_bn
from unet import UNet
from loss import TransformLoss


def worker_init_fn(worker_id):
    random.seed(worker_id)


def one_epoch(model, data_loader, criterion, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    losses = 0
    data_num = 0
    correct_num = 0
    iter_num = 0
    losses_adv = 0
    losses_mse = 0

    for images, targets in tqdm(data_loader):
        images, targets = images.to(device), targets.to(device)
        data_num += len(targets)
        iter_num += 1

        if optimizer:
            logits = model(images)
            loss, cor_num, loss_ce, loss_mse = criterion(images, logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss, cor_num, loss_ce, loss_mse = criterion(images, logits, targets)

        losses += loss.item()
        losses_adv += loss_ce.item()
        losses_mse += loss_mse.item()
        correct_num += cor_num

    return losses / iter_num, correct_num / data_num, losses_adv / iter_num, losses_mse / iter_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--p_ratio", required=True, type=float)
    args = parser.parse_args()

    # Constants
    DATA_PATH = "../data/"
    os.environ['TORCH_HOME'] = DATA_PATH  # pretrained model
    TRAIN_BATCH_SIZE = 128
    VALID_BATCH_SIZE = 256
    EPOCH_NUM = 200
    CHECKPOINT_FOLDER = "../model/transform/"
    CLASSIFIER_FOLDER = "../model/classifier/"
    NUM_WORKER = 4
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    # pretrained model folder
    os.environ['TORCH_HOME'] = DATA_PATH
    save_name = f"transform_{args.model}_{str(args.p_ratio).replace('.', '-')}_{args.classifier}_{args.dataset}.pth"
    classifier_name = f"classifier_{args.classifier}_{args.dataset}.pth"

    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    # Reproducibility
    torch.manual_seed(100)
    random.seed(200)
    np.random.seed(300)
    cudnn.deterministic = True
    cudnn.benchmark = False

    if args.dataset == "cifar10":
        num_classes = 10
        fc_size = 512

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])

        transform_valid = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_ds = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=True, download=True, transform=transform_train)
        valid_ds = torchvision.datasets.CIFAR10(
            root=DATA_PATH, train=True, download=True, transform=transform_valid)

        train_idx, valid_idx = train_test_split(np.arange(len(train_ds.targets)),
                                                test_size=0.1,
                                                shuffle=True,
                                                stratify=train_ds.targets)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True,
            worker_init_fn=worker_init_fn, sampler=train_sampler)
        valid_dl = torch.utils.data.DataLoader(
            valid_ds, batch_size=VALID_BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True,
            worker_init_fn=worker_init_fn, sampler=valid_sampler)
    elif args.dataset == "cifar100":
        num_classes = 100
        fc_size = 512

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])

        transform_valid = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_ds = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=True, download=True, transform=transform_train)
        valid_ds = torchvision.datasets.CIFAR100(
            root=DATA_PATH, train=True, download=True, transform=transform_valid)

        train_idx, valid_idx = train_test_split(np.arange(len(train_ds.targets)),
                                                test_size=0.05,
                                                shuffle=True,
                                                stratify=train_ds.targets)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True,
            worker_init_fn=worker_init_fn, sampler=train_sampler)
        valid_dl = torch.utils.data.DataLoader(
            valid_ds, batch_size=VALID_BATCH_SIZE, num_workers=NUM_WORKER, pin_memory=True,
            worker_init_fn=worker_init_fn, sampler=valid_sampler)
    else:
        raise ValueError(f"dataset argment is invalid. {args.dataset}")

    if args.model == "unet":
        model = UNet(img_ch=3, output_ch=3).to(device)
    else:
        raise ValueError(f"model argment is invalid. {args.model}")

    if args.classifier == "resnet20":
        classifier = resnet20(in_channels=3, num_classes=num_classes).to(device)
    elif args.classifier == "vgg16":
        classifier = vgg16_bn(in_channels=3, num_classes=num_classes, fc_size=fc_size).to(device)
    else:
        raise ValueError(f"classifier argment is invalid. {args.classifier}")
    classifier.load_state_dict(torch.load(CLASSIFIER_FOLDER + classifier_name, map_location=device))
    for param in classifier.parameters():
        param.requires_grad = False
    classifier.to(device).eval()

    criterion = TransformLoss(classifier=classifier, perceptual_ratio=args.p_ratio, device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    optim_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    valid_loss_list = []
    train_loss_list = []
    valid_accuracy_list = []
    train_accuracy_list = []
    best_valid_loss = None
    best_valid_accuracy = None

    for epoch in range(EPOCH_NUM):
        print(f"EPOCH: {epoch}, p_ratio: {args.p_ratio}")
        # train
        loss, accuracy, loss_ce, loss_mse = one_epoch(model, train_dl, criterion, optimizer=optimizer)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        print(f"train loss: {loss:.3}, accuracy: {accuracy:.3%}")
        print(f"train loss_ce: {loss_ce:.5}, loss_mse: {loss_mse:.5}")

        # valid
        loss, accuracy, loss_ce, loss_mse = one_epoch(model, valid_dl, criterion)
        valid_loss_list.append(loss)
        valid_accuracy_list.append(accuracy)
        print(f"valid loss: {loss:.3}, accuracy: {accuracy:.3%}")
        print(f"valid loss_ce: {loss_ce:.5}, loss_mse: {loss_mse:.5}")

        # step scheduler
        optim_scheduler.step()

        if epoch == 0 or best_valid_loss >= loss:
            print(f"------------------> Update model! {save_name} <------------------")
            best_valid_loss = loss
            best_valid_accuracy = accuracy
            torch.save(model.state_dict(), CHECKPOINT_FOLDER + save_name)
