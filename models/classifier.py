import torch.nn as nn
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class LinearClassifierAlexNet(nn.Module):
    def __init__(self, in_dim, n_hid=200, n_label=3):
        super(LinearClassifierAlexNet, self).__init__()
        
        self.classifier = nn.Sequential()
        self.classifier.add_module('Flatten', Flatten())
        #self.classifier.add_module('Dropout', nn.Dropout(p=0.3))
        self.classifier.add_module('LinearClassifier', nn.Linear(in_dim, n_hid))
        #self.classifier.add_module('ReLU', nn.ReLU(inplace=True))
        
        #self.classifier.add_module('Dropout', nn.Dropout(p=0.3))
        self.classifier.add_module('LinearClassifier2', nn.Linear(n_hid, n_label))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)

class LinearRegressor_mmse(nn.Module):
    def __init__(self, in_dim, n_hid=200):
        super(LinearRegressor_mmse, self).__init__()
        
        self.regressor = nn.Sequential()
        self.regressor.add_module('Flatten', Flatten())
        self.regressor.add_module('LinearClassifier', nn.Linear(in_dim, n_hid))
        self.regressor.add_module('ReLU', nn.ReLU(inplace=True))
        self.regressor.add_module('Dropout', nn.Dropout(p=0.3))
        self.regressor.add_module('LinearClassifier2', nn.Linear(n_hid, 1))
        self.initilize()

    def initilize(self):
        for n,m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)
            if 'LinearClassifier2' in n:
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)


    def forward(self, x):
        return self.regressor(x)

class AdversarialClassifier(nn.Module):
    def __init__(self, in_dim, nhid, out_dim):
        super(AdversarialClassifier, self).__init__()
        self.nhid = nhid
        self.l2norm = Normalize(2)
        layers = [
            nn.Linear(in_dim, out_dim),
            #nn.ReLU(),
            #nn.Linear(nhid, nhid),
            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.3),
            #nn.Linear(nhid, out_dim)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.l2norm(self.main(x))

class Deconv2D(nn.Module):
    def __init__(self, in_dim):
        super(Deconv2D, self).__init__()
        self.linear = nn.Linear(in_dim,144)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, 3, stride=2),  # b, 16, 25, 25
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, 7, stride=2),  # b, 32, 55, 55
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 7, stride=2),  # b, 1, 115, 115
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 1, 7, stride=1),  # b, 1, 121, 121
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1,1,12,12)
        x = self.decoder(x).squeeze(1)

        return x

if __name__ == '__main__':

    import torch
    
    model = Deconv2D(128)
    data = torch.rand(3, 128)#.cuda()
    out = model(data)
    print(out.shape)
