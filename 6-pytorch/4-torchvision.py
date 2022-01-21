from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision import datasets, models, transforms

from torchvision.models.resnet import resnet18

model = resnet18()

print(model)