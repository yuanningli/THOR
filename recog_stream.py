import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from PIL import Image


class RecogNet(object):
    def __init__(self):
        # import pretrained model and remove the soft-max layer
        self.model = models.vgg19(pretrained=True)
        new_classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        self.model.classifier = new_classifier

    def feat_extract(self, frame):
        # normalize the input image
        normalize = transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
        )
        preprocess = transforms.Compose([
           transforms.Scale(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize
        ])
        image = Image.fromarray(frame)
        img_tensor = preprocess(image)
        # extract features
        img_tensor.unsqueeze_(0)
        return self.model(Variable(img_tensor))


