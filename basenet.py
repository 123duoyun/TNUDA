import math as math
from torch import nn

class ResClassifier(nn.Module):
    def __init__(self, class_num, in_f,extract=False, training=False, dropout_p=0.5):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_f, in_f // 2),
            nn.BatchNorm1d(in_f // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
            )

        self.fc11 = nn.Linear(in_f // 2, class_num)
        self.extract = extract
        self.training = training
        self.dropout_p = dropout_p

    def forward(self, x):
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))
        logit = self.fc11(fc1_emb)
        if self.training:
            return logit, fc1_emb

        if self.extract:
            return fc1_emb, logit
        return logit