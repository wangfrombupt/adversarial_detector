import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 16, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv2d(input_channels, 12, kernel_size=1),
            BasicConv2d(12, 16, kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 16, kernel_size=1),
            BasicConv2d(16, 24, kernel_size=3, padding=1),
            BasicConv2d(24, 24, kernel_size=3, padding=1)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, pool_features, kernel_size=3, padding=1)
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(x)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(x)

        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = BasicConv2d(input_channels, 96, kernel_size=3, stride=2)

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 16, kernel_size=1),
            BasicConv2d(16, 24, kernel_size=3, padding=1),
            BasicConv2d(24, 24, kernel_size=3, stride=2)
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        #x - > 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 3x3 -> 3x3(downsample)
        branch3x3stack = self.branch3x3stack(x)

        #x -> avgpool(downsample)
        branchpool = self.branchpool(x)

        #"""We can use two parallel stride 2 blocks: P and C. P is a pooling
        #layer (either average or maximum pooling) the activation, both of
        #them are stride 2 the filter banks of which are concatenated as in
        #figure 10."""
        outputs = [branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, input_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 48, kernel_size=1)

        c7 = channels_7x7

        #In theory, we could go even further and argue that one can replace any n × n
        #convolution by a 1 × n convolution followed by a n × 1 convolution and the
        #computational cost saving increases dramatically as n grows (see figure 6).
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 48, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch7x7stack = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 48, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 48, kernel_size=1),
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1layer 1*7 and 7*1 (same)
        branch7x7 = self.branch7x7(x)

        #x-> 2layer 1*7 and 7*1(same)
        branch7x7stack = self.branch7x7stack(x)

        #x-> avgpool (same)
        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionD(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels,48, kernel_size=1),
            BasicConv2d(48, 80, kernel_size=3, stride=2)
        )

        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 48, kernel_size=1),
            BasicConv2d(48, 48, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(48, 48, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(48, 48, kernel_size=3, stride=2)
        )

        self.branchpool = nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        #x -> 1x1 -> 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 1x1 -> 1x7 -> 7x1 -> 3x3 (downsample)
        branch7x7 = self.branch7x7(x)

        #x -> avgpool (downsample)
        branchpool = self.branchpool(x)

        outputs = [branch3x3, branch7x7, branchpool]

        return torch.cat(outputs, 1)


#same
class InceptionE(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 80, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(input_channels, 96, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(96, 96, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(96, 96, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3stack_1 = BasicConv2d(input_channels, 112, kernel_size=1)
        self.branch3x3stack_2 = BasicConv2d(112, 96, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConv2d(96,96, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConv2d(96, 96, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 48, kernel_size=1)
        )

    def forward(self, x):

        #x -> 1x1 (same)
        branch1x1 = self.branch1x1(x)

        # x -> 1x1 -> 3x1
        # x -> 1x1 -> 1x3
        # concatenate(3x1, 1x3)
        #"""7. Inception modules with expanded the filter bank outputs.
        #This architecture is used on the coarsest (8 × 8) grids to promote
        #high dimensional representations, as suggested by principle
        #2 of Section 2."""
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        # x -> 1x1 -> 3x3 -> 1x3
        # x -> 1x1 -> 3x3 -> 3x1
        #concatenate(1x3, 3x1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)

        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionV3(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.Conv_module=nn.Sequential(
        BasicConv2d(3, 8, kernel_size=3, padding=1),
        BasicConv2d(8, 8, kernel_size=3, padding=1),
        BasicConv2d(8, 16, kernel_size=3, padding=1),
        BasicConv2d(16, 20, kernel_size=1),
        BasicConv2d(20, 48, kernel_size=3))

        #naive inception module
        self.A_module=nn.Sequential(
        InceptionA(48, pool_features=8),
        InceptionA(64, pool_features=16),
        InceptionA(72, pool_features=16))

        #downsample
        '''
        self.B_module=nn.Sequential(
        InceptionB(72))
        '''
        # self.B_module=nn.Sequential(
        # InceptionB(72))
        # self.C_module = nn.Sequential(
        # InceptionC(192, channels_7x7=32),
        # InceptionC(192, channels_7x7=40),
        # InceptionC(192, channels_7x7=40),
        # InceptionC(192, channels_7x7=48))

        self.BC_module = nn.Sequential(
        InceptionB(72),
        InceptionC(192, channels_7x7=32),
        InceptionC(192, channels_7x7=40),
        InceptionC(192, channels_7x7=40),
        InceptionC(192, channels_7x7=48))

        
        '''
        #downsample
        self.D_module=nn.Sequential(
        InceptionD(192))
        '''
        # self.D_module = nn.Sequential(
        # InceptionD(192))
        # self.E_module = nn.Sequential(
        # InceptionE(320),
        # InceptionE(512))

        self.DE_module = nn.Sequential(
        InceptionD(192),
        InceptionE(320),
        InceptionE(512))

        #6*6 feature size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(512, num_classes)

    # def forward(self, x):

    #     x = self.Conv_module(x)
    #     x = self.A_module(x)

    #     x = self.B_module(x)
    #     x = self.C_module(x)

    #     x = self.D_module(x)
    #     x = self.E_module(x)

    #     x = self.avgpool(x)
    #     x = self.dropout(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.linear(x)
    #     return x

    # def feature_list(self, x):
    #     out_list = []  
    #     x = self.Conv_module(x)
    #     out_list.append(x)

    #     x = self.A_module(x)
    #     out_list.append(x)

    #     x = self.B_module(x)
    #     out_list.append(x)

    #     x = self.C_module(x)
    #     out_list.append(x)

    #     x = self.D_module(x)
    #     out_list.append(x)

    #     x = self.E_module(x)
    #     out_list.append(x)

    #     x = self.avgpool(x)
    #     x = self.dropout(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.linear(x)
    #     # print(out_list)
    #     return x, out_list

    def forward(self, x):

        x = self.Conv_module(x)
        x = self.A_module(x)

        x = self.BC_module(x)

        x = self.DE_module(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    def feature_list(self, x):
        out_list = []  
        x = self.Conv_module(x)
        out_list.append(x)

        x = self.A_module(x)
        out_list.append(x)

        x = self.BC_module(x)
        out_list.append(x)

        x = self.DE_module(x)
        out_list.append(x)

        x = self.avgpool(x)
        out_list.append(x)

        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # print(out_list)
        return x,out_list

def inceptionv3():
    return InceptionV3()
