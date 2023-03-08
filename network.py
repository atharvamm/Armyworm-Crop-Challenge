import torch.nn as nn

# Later add dropout and batchnorm
# Define weight initialization

class Network(nn.Module):
    def __init__(self,):
        super(Network,self).__init__()
        self.cnn1  = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=4,padding=1,stride=1)
        # self.pool1 = nn.AdaptiveAvgPool2d()
        self.cnn2  = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=2,padding=1,stride=1)
        # self.pool1 = nn.AdaptiveAvgPool2d()
        self.cnn3  = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2,padding=1,stride=1)
        self.cnn4  = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,padding=1,stride=1)
        self.pool_adap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.lin1 = nn.Linear(in_features=256,out_features = 256)
        self.lin2 = nn.Linear(in_features=256,out_features = 512)
        self.lin3 = nn.Linear(in_features=512,out_features = 2)
        self.activation1 = nn.GELU()
        self.activation2 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self,x):

        x = self.cnn1(x)
        x = self.activation1(x)
        x = self.cnn2(x)
        x = self.activation1(x)
        x = self.cnn3(x)
        x = self.activation1(x)
        x = self.cnn4(x)
        x = self.activation1(x)
        x = self.pool_adap(x)
        x = self.flatten(x)

        x = self.lin1(x)
        x = self.activation1(x)
        x = self.lin2(x)
        x = self.activation1(x)
        x = self.lin3(x)
        
        return x
    

        
