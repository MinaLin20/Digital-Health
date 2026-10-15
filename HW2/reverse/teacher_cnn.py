"""Teacher CNN used by HW2 Part D (public). Load with:
    m = TeacherCNN(); m.load_state_dict(torch.load("cnn.pt")); m.eval()
Input: (N, 1, 28, 28) float tensor, pixel scale x/255 in [0, 1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16, 2)

    def features(self, x):
        """Last conv feature map after ReLU, before GAP: (N, 16, 14, 14)."""
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        return F.relu(self.conv2(x))

    def forward(self, x):
        a = self.features(x)
        return self.fc(a.mean(dim=(2, 3)))
