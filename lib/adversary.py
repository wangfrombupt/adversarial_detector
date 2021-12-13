import torch
import torch.nn as nn
from models.resnet import LabelSmoothingCrossEntropy


class PGD(nn.Module):
    def __init__(self, classifier, detector, label_smooth, sigma, mode='iter', eps=8/255, alpha=2/255, steps=20, random_start=True):
        super(PGD, self).__init__()
        assert mode in ['combine', 'iter']
        if mode == 'combine': assert sigma is not None
        self.mode = mode
        self.classifier = classifier
        self.detector = detector
        self.label_smooth = label_smooth
        self.sigma = sigma
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = next(classifier.parameters()).device

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss_clf = LabelSmoothingCrossEntropy(eps=self.label_smooth).to(self.device)
        loss_det = nn.CrossEntropyLoss().to(self.device)
        
        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs_clf, features = self.classifier.feature_list(adv_images)
            cost_clf = loss_clf(outputs_clf, labels)

            outputs_det = self.detector(features)
            cost_det = loss_det(outputs_det, torch.ones(outputs_det.size(0)).long().to(self.device))

            if self.mode == 'iter':
                if i % 2 == 0:
                    loss = cost_clf
                else:
                    loss = cost_det
            elif self.mode == 'combine':
                loss = (1 - self.sigma) * cost_clf + self.sigma * cost_det

            loss.backward(retain_graph=True)

            adv_images = adv_images.detach() + self.alpha * adv_images.grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images