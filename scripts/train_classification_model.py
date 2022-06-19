import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append("../lib/")
from resnet import resnet20
from vgg import vgg16_bn


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

    for images, targets in tqdm(data_loader):
        images, targets = images.to(device), targets.to(device)
        data_num += len(targets)
        iter_num += 1

        if optimizer:
            logits = model(images)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, targets)

        losses += loss.item()

        prediction = torch.argmax(logits, dim=1)
        correct_num += (prediction == targets).sum().item()

    return losses / iter_num, correct_num / data_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    # Constants
    DATA_PATH = "../data/"
    TRAIN_BATCH_SIZE = 128
    VALID_BATCH_SIZE = 256
    EPOCH_NUM = 200
    CHECKPOINT_FOLDER = "../model/classifier/"
    NUM_WORKER = 4
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    save_name = f"classifier_{args.model}_{args.dataset}.pth"

    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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
            normalize,
        ])

        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
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
            normalize,
        ])

        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
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

    if args.model == "resnet20":
        model = resnet20(in_channels=3, num_classes=num_classes).to(device)
    elif args.model == "vgg16":
        model = vgg16_bn(in_channels=3, num_classes=num_classes, fc_size=fc_size).to(device)
    else:
        raise ValueError(f"model argment is invalid. {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    optim_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    train_loss_list = []
    train_accuracy_list = []
    valid_loss_list = []
    valid_accuracy_list = []
    best_valid_loss = None
    best_valid_accuracy = None

    for epoch in range(EPOCH_NUM):
        print("EPOCH: {}".format(epoch))
        # train
        loss, accuracy = one_epoch(model, train_dl, criterion, optimizer=optimizer)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        print("train loss: {:.3}, accuracy: {:.3%}".format(loss, accuracy))

        # valid
        loss, accuracy = one_epoch(model, valid_dl, criterion)
        valid_loss_list.append(loss)
        valid_accuracy_list.append(accuracy)
        print("valid loss: {:.3}, accuracy: {:.3%}".format(loss, accuracy))

        # step scheduler
        optim_scheduler.step()

        if epoch == 0 or best_valid_loss >= loss:
            print("-------------------------------> Update model! <-------------------------------")
            best_valid_loss = loss
            best_valid_accuracy = accuracy
            torch.save(model.state_dict(), CHECKPOINT_FOLDER + save_name)

    print("best valid loss: {:.3}, accuracy: {:.3%}".format(best_valid_loss, best_valid_accuracy))
