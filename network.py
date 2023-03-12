import torch.nn as nn

# Later add dropout and batchnorm
# Define weight initialization

class Network(nn.Module):
    def __init__(self,):

        super(Network,self).__init__()
        self.cnn1  = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,padding=0,stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.cnn2  = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2,stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=2)
        self.cnn3  = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1,stride=1)
        self.cnn4  = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1,stride=1)
        self.cnn5  = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1,stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.cnn6 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.pool_adap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.lin1 = nn.Linear(in_features=896,out_features = 512)
        self.lin2 = nn.Linear(in_features=512,out_features = 64)
        self.lin3 = nn.Linear(in_features=64,out_features = 2)
        self.activation1 = nn.GELU()
        self.activation2 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=0.25)



    def forward(self,x):

        x = self.cnn1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.cnn2(x)
        x = self.activation1(x)
        x = self.pool2(x)
        x = self.cnn3(x)
        x = self.activation1(x)
        x = self.cnn4(x)
        x = self.activation1(x)
        x = self.cnn5(x)
        x = self.activation1(x)
        x = self.pool3(x)
        x = self.cnn6(x)
        x = self.activation1(x)
        x = self.pool_adap(x)
        x = self.flatten(x)

        x = self.lin1(x)
        x = self.dropout(x)
        x = self.activation1(x)
        x = self.lin2(x)
        x = self.dropout(x)
        x = self.activation1(x)
        x = self.lin3(x)
        
        return x
    

        
