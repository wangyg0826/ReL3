# -*- coding: utf-8 -*-
"""L3_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wC_K7ak3yJT2rqjM5Ld79SiTo8j47paz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VisualSubNet(nn.Module):
  def __init__(self):
    super(VisualSubNet, self).__init__()
    self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    self.bn1_1 = nn.BatchNorm2d(64)
    self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn1_2 = nn.BatchNorm2d(64)
    self.pool1 = nn.MaxPool2d(2)
    
    self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn2_1 = nn.BatchNorm2d(128)
    self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn2_2 = nn.BatchNorm2d(128)
    self.pool2 = nn.MaxPool2d(2)

    self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.bn3_1 = nn.BatchNorm2d(256)
    self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.bn3_2 = nn.BatchNorm2d(256)
    self.pool3 = nn.MaxPool2d(2)

    self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.bn4_1 = nn.BatchNorm2d(512)
    self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn4_2 = nn.BatchNorm2d(512)
    self.pool4 = nn.MaxPool2d(28)
    
  def visual_feature(self, image):#(b, c=3, h=224, w=224)
    x = F.relu(self.bn1_1(self.conv1_1(image)))
    x = F.relu(self.bn1_2(self.conv1_2(x)))
    x = self.pool1(x)

    x = F.relu(self.bn2_1(self.conv2_1(x)))
    x = F.relu(self.bn2_2(self.conv2_2(x)))
    x = self.pool2(x)

    x = F.relu(self.bn3_1(self.conv3_1(x)))
    x = F.relu(self.bn3_2(self.conv3_2(x)))
    x = self.pool3(x)
    
    x = F.relu(self.bn4_1(self.conv4_1(x)))
    x = F.relu(self.bn4_2(self.conv4_2(x)))
    return x
    
    
  def forward(self, image):#(b, c=3, h=224, w=224)
    x = self.visual_feature(image)
    x = self.pool4(x)

    return x

class AudioSubNet(nn.Module):
  def __init__(self):
    super(AudioSubNet, self).__init__()
    self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    self.bn1_1 = nn.BatchNorm2d(64)
    self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn1_2 = nn.BatchNorm2d(64)
    self.pool1 = nn.MaxPool2d(2)
    
    self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn2_1 = nn.BatchNorm2d(128)
    self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn2_2 = nn.BatchNorm2d(128)
    self.pool2 = nn.MaxPool2d(2)

    self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.bn3_1 = nn.BatchNorm2d(256)
    self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.bn3_2 = nn.BatchNorm2d(256)
    self.pool3 = nn.MaxPool2d(2)

    self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.bn4_1 = nn.BatchNorm2d(512)
    self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn4_2 = nn.BatchNorm2d(512)
    self.pool4 = nn.MaxPool2d(kernel_size=(32, 24))
  
  def audio_feature(self, audio):
    x = F.relu(self.bn1_1(self.conv1_1(audio)))
    x = F.relu(self.bn1_2(self.conv1_2(x)))
    x = self.pool1(x)

    x = F.relu(self.bn2_1(self.conv2_1(x)))
    x = F.relu(self.bn2_2(self.conv2_2(x)))
    x = self.pool2(x)

    x = F.relu(self.bn3_1(self.conv3_1(x)))
    x = F.relu(self.bn3_2(self.conv3_2(x)))
    x = self.pool3(x)
    
    x = F.relu(self.bn4_1(self.conv4_1(x)))
    x = self.bn4_2(self.conv4_2(x))
    return x

  def forward(self, audio):#(b, c=3, h=224, w=224)
    x = self.audio_feature(audio)
    x = F.relu(x)
    x = self.pool4(x)

    return x

class FusionNet(nn.Module):
  def __init__(self):
    super(FusionNet, self).__init__()
    self.fc1 = nn.Linear(1024, 128)
    self.fc2 = nn.Linear(128, 2)

  def forward(self, image_feat, audio_feat):#(b, 512, 1, 1)
    x = torch.cat([image_feat, audio_feat], dim=1).view(-1, 1024)#(b, 1024)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)#(b, 2)
    return x

class L3Net(nn.Module):
  def __init__(self):
    super(L3Net, self).__init__()
    self.visualsub = VisualSubNet()
    self.audiosub = AudioSubNet()
    self.fusion = FusionNet()
  
  def forward(self, image, audio):#image:(b, 3, 224, 224); audio:(b, 1, 257, 199)
    return self.fusion(self.visualsub(image), self.audiosub(audio))#(b, 2)
    
  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

class AudioFeature(nn.Module):
  def __init__(self, audioSubNet):
    super(AudioFeature, self).__init__()
    self.audioSubNet = audioSubNet
    self.audioSubNet.eval()
    self.pool = nn.MaxPool2d(kernel_size=(8, 8))
    self.eval()

  def forward(self, audio):
    with torch.no_grad():
      feature = self.audioSubNet.audio_feature(audio)
      feature = self.pool(feature)
      feature = feature.view(feature.shape[0], -1)
      feature_mean = torch.mean(feature, dim=1, keepdim=True)
      feature_std = torch.std(feature, dim=1, keepdim=True)
      feature.sub_(feature_mean).div_(feature_std)
      #print(torch.mean(feature, dim=1))
      #print(torch.std(feature, dim=1))
      return feature

class VisualFeature(nn.Module):
  def __init__(self, visualsub, out_dim, in_dim=8192):
    super(VisualFeature, self).__init__()
    self.visualsub = visualsub
    self.visualsub.eval()
    self.fc = nn.Linear(in_dim, out_dim)
    self.mp = nn.MaxPool2d(kernel_size=3, stride=3)

  def forward(self, image):
    with torch.no_grad():
      feature = self.visualsub.visual_feature(image)
      feature = self.mp(feature)
    feature = feature.view(feature.shape[0], -1)
    feature = self.fc(feature)
    return feature

class AudioEncoder(nn.Module):
  def __init__(self):
    super(AudioEncoder, self).__init__()
    self.conv1 = nn.Conv1d(1, 32, 64, stride=2, padding=32)
    self.pool1 = nn.MaxPool1d(8, stride=8, padding=0, return_indices=True)
    self.conv2 = nn.Conv1d(32, 64, 32, stride=2, padding=16)
    self.pool2 = nn.MaxPool1d(8, stride=8, padding=0, return_indices=True)
    self.conv3 = nn.Conv1d(64, 128, 16, stride=2, padding=8)
    self.pool3 = nn.MaxPool1d(8, stride=8, padding=0, return_indices=True)
    self.conv4 = nn.Conv1d(128, 256, 8, stride=2, padding=4)

  def forward(self, x):
    x = self.conv1(x)
    size1 = x.size()
    x, idx1 = self.pool1(x)
    x = self.conv2(x)
    size2 = x.size()
    x, idx2 = self.pool2(x)
    x = self.conv3(x)
    size3 = x.size()
    x, idx3 = self.pool3(x)
    x = self.conv4(x)
    #print(x.shape)
    return x, idx1, idx2, idx3, size1, size2, size3

class AudioDecoder(nn.Module):
  def __init__(self):
    super(AudioDecoder, self).__init__()
    self.deconv4 = nn.ConvTranspose1d(256, 128, 8, stride=2, padding=4, output_padding=1)
    self.unpool3 = nn.MaxUnpool1d(8, stride=8, padding=0)
    self.deconv3 = nn.ConvTranspose1d(128, 64, 16, stride=2, padding=8, output_padding=1)
    self.unpool2 = nn.MaxUnpool1d(8, stride=8, padding=0)
    self.deconv2 = nn.ConvTranspose1d(64, 32, 32, stride=2, padding=16)
    self.unpool1 = nn.MaxUnpool1d(8, stride=8, padding=0)
    self.deconv1 = nn.ConvTranspose1d(32, 1, 64, stride=2, padding=32)

  def forward(self, x, idx1, idx2, idx3, size1, size2, size3):
    x = self.deconv4(x)
    x = self.unpool3(x, idx3, output_size=size3)
    x = self.deconv3(x)
    x = self.unpool2(x, idx2, output_size=size2)
    x = self.deconv2(x)
    x = self.unpool1(x, idx1, output_size=size1)
    x = self.deconv1(x)
    return x


class AudioAutoencoder(nn.Module):
  def __init__(self):
    super(AudioAutoencoder, self).__init__()
    self.encoder = AudioEncoder()
    self.decoder = AudioDecoder()

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x, idx1, idx2, idx3, size1, size2, size3 = self.encoder(x)
    x = self.decoder(x, idx1, idx2, idx3, size1, size2, size3)
    return x

class AudioEncLinear(nn.Module):
  def __init__(self, encoder, n_feature, n_class):
    super(AudioEncLinear, self).__init__()
    self.encoder = encoder
    #self.encoder.eval()
    self.fc = nn.Linear(n_feature, n_class)

  def forward(self, x):
    self.encoder.eval()
    with torch.no_grad():
      feature, _, _, _, _, _, _ = self.encoder(x)
      feature = feature.view(feature.shape[0], -1)
    out = self.fc(feature.detach())
    return out

  def _initialize_weights(self):
    nn.init.normal_(self.fc.weight, 0, 0.01)
    nn.init.constant_(self.fc.bias, 0)

class SVM(nn.Module):
  def __init__(self, n_feature, n_class):
    super(SVM, self).__init__()
    self.fc = nn.Linear(n_feature, n_class)

  def _initialize_weights(self):
    nn.init.normal_(self.fc.weight, 0, 0.01)
    nn.init.constant_(self.fc.bias, 0)

  def forward(self, x):
    return self.fc(x)
  
  def predict(self, x):
    with torch.no_grad():
      pred = self.forward(x)
      pred_label = pred.argmax(dim=1)
      return pred_label

#######Dimension Check########
# image = torch.rand(2, 3, 224, 224)
# audio = torch.rand(2, 1, 257, 199)
# print(image.shape, audio.shape)

# model = L3Net()
# pred = model(image, audio)
# print(pred.shape)
