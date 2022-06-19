import torch
import torch.nn as nn
import torchvision


class TransformLoss(nn.Module):
    def __init__(self, classifier, perceptual_ratio,
                 device="cuda:0", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(TransformLoss, self).__init__()
        self.classifier = classifier
        self.perceptual_ratio = perceptual_ratio

        self.relu2 = torchvision.models.vgg16(pretrained=True).features[:9]
        for param in self.relu2.parameters():
            param.requires_grad = False
        self.relu2.to(device).eval()

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

        self.mean = torch.FloatTensor(mean).view(3, 1, 1).to(device)
        self.std = torch.FloatTensor(std).view(3, 1, 1).to(device)

    def forward(self, inputs, logits, targets):
        logits = self.sigmoid(logits)

        logits = (logits - self.mean) / self.std
        inputs = (inputs - self.mean) / self.std

        outputs = self.classifier(logits)
        loss_ce = self.ce_loss(outputs, targets)

        loss_mse = self.mse_loss(self.relu2(logits), self.relu2(inputs))

        pred = torch.argmax(outputs, dim=1)
        corrects = (pred == targets).sum().item()

        loss = loss_ce - loss_mse * self.perceptual_ratio

        return loss, corrects, loss_ce, loss_mse
