
import torch
import torch.nn as nn



class STFTANet(nn.Module):
    def __init__(self, num_classes=6):
        super(STFTANet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(9,1), stride=(7,1), padding=(1,0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (3,1), stride = (2,1)),
            nn.Dropout(0.5))

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(7,1), stride=(5,1), padding=(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(5,1), stride=(3,1), padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5))
        
                
        self.fc1= nn.Sequential(
            #nn.Dropout(0.5),
            #nn.Linear(1280, num_classes))
            nn.Linear(512, num_classes))
        

    def forward(self,  x):
        x1 = self.layer1(x)   
        x2 = self.layer2(x1)    
        x3 = self.layer3(x2)  
        x4 = x3.reshape(x3.size(0), -1)  
        x5 = torch.sigmoid(x4)        
        out = self.fc1(x5)  
      
        return  out  








