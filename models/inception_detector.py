import torch
import torch.nn as nn
import torch.nn.functional as F


class Text_CNN_Inception(torch.nn.Module):
    # Text-CNN Detector For Inception
    def __init__(self):
        super(Text_CNN_Detector_Inception, self).__init__()
        self.cp1 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp2 = nn.Sequential(
            nn.Conv2d(72, 192, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp3 = nn.Sequential(
            nn.Conv2d(192, 512, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )

        filter_sizes = [1, 2, 3, 4]
        num_filters = 100
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (K, 512), bias=True) for K in filter_sizes])
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, 200)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        out0 = self.cp1(x[0])
        out0 = self.cp2(out0)
        out0 = self.cp3(out0)
        out0 = F.avg_pool2d(out0, 2)
        out0 = out0.view(out0.size(0), 1, -1)

        out1 = self.cp2(x[1])
        out1 = self.cp3(out1)
        out1 = F.avg_pool2d(out1, 4)
        out1 = out1.view(out1.size(0), 1, -1)

        out2 = self.cp3(x[2])
        out2 = F.avg_pool2d(out2, 4)
        out2 = out2.view(out2.size(0), 1, -1)

        out3 = self.cp4(x[3])
        out3 = out3.view(out3.size(0), 1, -1)
        out3 = self.linear1(out3)

        out4 = x[4]
        out4 = out4.view(out4.size(0), 1, -1)

        txt = torch.cat((out0, out1, out2, out3, out4), 1)
        txt = torch.unsqueeze(txt, 1)

        out = [F.relu(conv(txt)).squeeze(3) for conv in self.convs]
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1)
        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        logit = self.fc2(out)

        return logit