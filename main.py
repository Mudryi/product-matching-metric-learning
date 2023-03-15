from torchvision import transforms

from src.config import *
from src.dataloaders import triplet_dataloaders
from src.model_training import fine_tune_ViT_triplet
from src.VIT_model.VIT_model import load_pretrained_VIT_model_triplet

transform = {'train': transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(50),
                                          transforms.ToTensor()]),

             'eval': transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()])
             }

dataloaders = triplet_dataloaders(batch_size=batch_size, num_workers=num_workers, transform=transform)

encoder = load_pretrained_VIT_model_triplet()
encoder = fine_tune_ViT_triplet(encoder,
                                dataloaders,
                                n_epochs=n_epochs,
                                lr=learning_rate,
                                batch_size=batch_size)
