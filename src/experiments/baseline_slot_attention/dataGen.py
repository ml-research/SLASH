import torch
import torchvision

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import datasets as datasets


def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    '''
    Returns and iterable dataset with specified batchsize and shuffling.
    '''
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers
)


def get_encoding_shapeworld(color, shape, shade, size):
    
    if color == 'red':
        col_enc = [1,0,0,0,0,0,0,0]
    elif color == 'blue':
        col_enc = [0,1,0,0,0,0,0,0]
    elif color == 'green':
        col_enc = [0,0,1,0,0,0,0,0]
    elif color == 'gray':
        col_enc = [0,0,0,1,0,0,0,0]
    elif color == 'brown':
        col_enc = [0,0,0,0,1,0,0,0]
    elif color == 'magenta':
        col_enc = [0,0,0,0,0,1,0,0]
    elif color == 'cyan':
        col_enc = [0,0,0,0,0,0,1,0]
    elif color == 'yellow':
        col_enc = [0,0,0,0,0,0,0,1]

    if shape == 'circle':
        shape_enc = [1,0,0]
    elif shape == 'triangle':
        shape_enc = [0,1,0]
    elif shape == 'square':
        shape_enc = [0,0,1]    
   
    if shade == 'bright':
        shade_enc = [1,0]
    elif shade =='dark':
        shade_enc = [0,1]

             
    if size == 'small':
        size_enc = [1,0]
    elif size == 'big':
        size_enc = [0,1]
    
    return np.array([1]+ col_enc + shape_enc + shade_enc + size_enc)
    
    
class SHAPEWORLD4(Dataset):
    def __init__(self, root, mode, learn_concept='default', bg_encoded=True):
        
        datasets.maybe_download_shapeworld4()

        self.root = root
        self.mode = mode
        assert os.path.exists(root), 'Path {} does not exist'.format(root)

        #dictionary of the form {'image_idx':'img_path'}
        self.img_paths = {}
        
        
        for file in os.scandir(os.path.join(root, 'images', mode)):
            img_path = file.path
            
            img_path_idx =   img_path.split("/")
            img_path_idx = img_path_idx[-1]
            img_path_idx = img_path_idx[:-4][6:]
            try:
                img_path_idx =  int(img_path_idx)
                self.img_paths[img_path_idx] = img_path
            except:
                print("path:",img_path_idx)
                

        
        count = 0
        
        #target maps of the form {'target:idx': observation string} or {'target:idx': obj encoding}
        self.obj_map = {}
                
        with open(os.path.join(root, 'labels', mode,"world_model.json")) as f:
            worlds = json.load(f)
            
            
            
            #iterate over all objects
            for world in worlds:
                num_objects = 0
                target_obs = ""
                obj_enc = []
                for entity in world['entities']:
                    
                    color = entity['color']['name']
                    shape = entity['shape']['name']
                    
                    shade_val = entity['color']['shade']
                    if shade_val == 0.0:
                        shade = 'bright'
                    else:
                        shade = 'dark'
                    
                    size_val = entity['shape']['size']['x']
                    if size_val == 0.075:
                        size = 'small'
                    elif size_val == 0.15:
                        size = 'big'
                    
                    name = 'o' + str(num_objects+1)
                    obj_enc.append(get_encoding_shapeworld(color, shape, shade, size))
                    num_objects += 1
                    
                #bg encodings
                for i in range(num_objects, 4):
                    name = 'o' + str(num_objects+1)
                    obj_enc.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
                    num_objects += 1

                self.obj_map[count] = torch.Tensor(obj_enc)
                count+=1

    def __getitem__(self, index):
        
        #get the image
        img_path = self.img_paths[index]
        img = io.imread(img_path)[:, :, :3]
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.CenterCrop(250),
            #transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        img = transform(img)
        img = (img - 0.5) * 2.0  # Rescale to [-1, 1].

        return img, self.obj_map[index]#, mask
        
    def __len__(self):
        return len(self.img_paths)

def get_encoding_clevr(size, material, shape, color ):
    
    #size (small, large, bg)
    if size == "small":
        size_enc = [1,0]
    elif size == "large":
        size_enc = [0,1]
    
    #material (rubber, metal, bg)
    if material == "rubber":
        material_enc = [1,0]
    elif material == "metal":
        material_enc = [0,1]
    
    #shape (cube, sphere, cylinder, bg)
    if shape == "cube":
        shape_enc = [1,0,0]
    elif shape == "sphere":
        shape_enc = [0,1,0]
    elif shape == "cylinder":
        shape_enc = [0,0,1]

    
    #color (gray, red, blue, green, brown, purple, cyan, yellow, bg)
    if color == "gray":
        color_enc = [1,0,0,0,0,0,0,0]
    elif color == "red":
        color_enc = [0,1,0,0,0,0,0,0]
    elif color == "blue":
        color_enc = [0,0,1,0,0,0,0,0]
    elif color == "green":
        color_enc = [0,0,0,1,0,0,0,0]
    elif color == "brown":
        color_enc = [0,0,0,0,1,0,0,0]
    elif color == "purple":
        color_enc = [0,0,0,0,0,1,0,0]
    elif color == "cyan":
        color_enc = [0,0,0,0,0,0,1,0]
    elif color == "yellow":
        color_enc = [0,0,0,0,0,0,0,1]

        
    return np.array([1] + size_enc + material_enc + shape_enc + color_enc )


class CLEVR(Dataset):
    def __init__(self, root, mode, img_paths=None, files_names=None, obj_num=None):
        self.root = root  # The root folder of the dataset
        self.mode = mode  # The mode of 'train' or 'val'
        self.files_names = files_names # The list of the files names with correct nuber of objects
        if obj_num is not None:
            self.obj_num = obj_num  # The upper limit of number of objects 
        else:
            self.obj_num = 10

        assert os.path.exists(root), 'Path {} does not exist'.format(root)

        #list of sorted image paths
        self.img_paths = []
        if img_paths:
            self.img_paths = img_paths
        else:                    
            #open directory and save all image paths
            for file in os.scandir(os.path.join(root, 'images', mode)):
                img_path = file.path
                if '.png' in img_path:
                    self.img_paths.append(img_path)

        self.img_paths.sort()
        self.img_paths = np.array(self.img_paths, dtype=str)

        count = 0
        
        #target maps of the form {'target:idx': query string} or {'target:idx': obj encoding}
        #self.obj_map = {}

        
        count = 0        
        #We have up to 10 objects in the image, load the json file
        with open(os.path.join(root, 'scenes','CLEVR_'+ mode+"_scenes.json")) as f:
            data = json.load(f)
            
            self.obj_map = np.empty((len(data['scenes']),10,16), dtype=np.float32)
            
            #iterate over each scene and create the query string and obj encoding
            print("parsing scences")
            for scene in data['scenes']:
                obj_encoding_list = []

                if self.files_names:
                    if any(scene['image_filename'] in file_name for file_name in files_names):                    
                        num_objects = 0
                        for idx, obj in enumerate(scene['objects']):
                            obj_encoding_list.append(get_encoding_clevr(obj['size'], obj['material'], obj['shape'], obj['color']))
                            num_objects = idx+1 #store the num of objects 
                        #fill in background objects
                        for idx in range(num_objects, self.obj_num):
                            obj_encoding_list.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                        self.obj_map[count] = np.array(obj_encoding_list)
                        count += 1
                else:
                    num_objects=0
                    for idx, obj in enumerate(scene['objects']):
                        obj_encoding_list.append(get_encoding_clevr(obj['size'], obj['material'], obj['shape'], obj['color']))
                        num_objects = idx+1 #store the num of objects 
                    #fill in background objects
                    for idx in range(num_objects, 10):
                        obj_encoding_list.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                    self.obj_map[scene['image_index']] = np.array(obj_encoding_list, dtype=np.float32)
                
            print("done")
        if self.files_names:
            print(f'Correctly found images {count} out of {len(files_names)}')

    def __getitem__(self, index):
        
        #get the image
        img_path = self.img_paths[index]
        img = io.imread(img_path)[:, :, :3]
        img = Image.fromarray(img).resize((128,128)) #using transforms to resize gets us a shrared-memory leak :(

        transform = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.CenterCrop((29, 221,64, 256)), #why do we need to crop?
            #transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        img = transform(img)
        img = (img - 0.5) * 2.0  # Rescale to [-1, 1].

        return img, self.obj_map[index]#, mask
        
    def __len__(self):
        return self.img_paths.shape[0]