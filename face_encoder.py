from typing import Any
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import GPUtil
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
# Ignore DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
sys.path.append('./')
sys.path.append('../')
class FaceEncoder:
    def __init__(self, pretrained='vggface2', 
                 device = 0,
                 
                 ) -> None:
        ##### setup and checking phase #####  
        print('setup and checking phase for face_encoder.py')
        self.device = None
        if device == -1:
            self.device = torch.device('cpu')
            print('Using CPU')
        else:
            if torch.cuda.is_available():
                # check how many GPUs are available
                if device<0 or device>=torch.cuda.device_count():
                    print('Invalid device! Using CPU')
                    self.device = torch.device('cpu')
                # check how many memory is available
                elif self.get_size_of_free_memory(device) < 1000:
                    print('Not enough memory! Using CPU')
                    self.device = torch.device('cpu')
                else:
                    print('Using GPU, device:', device)
                    self.device = torch.device(f'cuda:{device}')        
            else:
                print('CUDA not available! Using CPU')
                self.device = torch.device('cpu')  
        print(self.device)
        self.encoder = InceptionResnetV1(pretrained=pretrained, device=self.device).eval()
        self.encoder = self.encoder.to(self.device)
        self.new_image_size = (182, 182)
        self.batch_size = 16
        self.transform = transforms.Compose([
            transforms.Resize(self.new_image_size),
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    def __call__(self, input_image) -> Any:
        return self.encode(self.read_image_from_capture_thread(input_image))
    
    def get_size_of_free_memory(self, device) -> int:
        gpus = GPUtil.getGPUs()
        gpu = gpus[device]     
        return gpu.memoryFree
    
    def encode(self, face):
        return self.encoder(face).cpu().detach().numpy()
    
    def read_image_from_capture_thread(self, cv2_image_list) -> torch.Tensor:
        image_list = []
        for image in cv2_image_list:
            image = to_pil(image)
            image = self.transform(image)
            image_list.append(image)
            del image
        image_tensor = torch.stack(image_list)  # Create a tensor from the list of images
        del image_list
        return image_tensor.to(self.device)
        
    