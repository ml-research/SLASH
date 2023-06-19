import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# from torch.utils.data import Dataset
# from torchvision import transforms
from skimage import io
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json

from tqdm import tqdm

    
def object_encoding(size, material, shape, color ):
    
    #size (small, large, bg)
    if size == "small":
        size_enc = [1,0,0]
    elif size == "large":
        size_enc = [0,1,0]
    elif size == "bg":
        size_enc = [0,0,1]
    
    #material (rubber, metal, bg)
    if material == "rubber":
        material_enc = [1,0,0]
    elif material == "metal":
        material_enc = [0,1,0]
    elif material == "bg":
        material_enc = [0,0,1]
    
    #shape (cube, sphere, cylinder, bg)
    if shape == "cube":
        shape_enc = [1,0,0,0]
    elif shape == "sphere":
        shape_enc = [0,1,0,0]
    elif shape == "cylinder":
        shape_enc = [0,0,1,0]
    elif shape == "bg":
        shape_enc = [0,0,0,1]
    
    #color (gray, red, blue, green, brown, purple, cyan, yellow, bg)
    #color (gray, red, blue, green, brown, purple, cyan, yellow, bg)
    #color  1      2     3     4     5     6        7     8      9
    if color == "gray":
        color_enc = [1,0,0,0,0,0,0,0,0]
    elif color == "red":
        color_enc = [0,1,0,0,0,0,0,0,0]
    elif color == "blue":
        color_enc = [0,0,1,0,0,0,0,0,0]
    elif color == "green":
        color_enc = [0,0,0,1,0,0,0,0,0]
    elif color == "brown":
        color_enc = [0,0,0,0,1,0,0,0,0]
    elif color == "purple":
        color_enc = [0,0,0,0,0,1,0,0,0]
    elif color == "cyan":
        color_enc = [0,0,0,0,0,0,1,0,0]
    elif color == "yellow":
        color_enc = [0,0,0,0,0,0,0,1,0]
    elif color == "bg":
        color_enc = [0,0,0,0,0,0,0,0,1]
        
    return size_enc + material_enc + shape_enc + color_enc +[1]
       


    
    
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
        count = 0
        
        #target maps of the form {'target:idx': query string} or {'target:idx': obj encoding}
        self.query_map = {}
        self.obj_map = {}
        
        count = 0        
        #We have up to 10 objects in the image, load the json file
        with open(os.path.join(root, 'scenes','CLEVR_'+ mode+"_scenes.json")) as f:
            data = json.load(f)
            
            #iterate over each scene and create the query string and obj encoding
            print("parsing scences")
            for scene in tqdm(data['scenes']):
                target_query = ""
                obj_encoding_list = []

                if self.files_names:
                    if any(scene['image_filename'] in file_name for file_name in files_names):                    
                        num_objects = 0
                        for idx, obj in enumerate(scene['objects']):
                            target_query += " :- not object(o{}, {}, {}, {}, {}).".format(idx+1, obj['size'], obj['material'], obj['shape'], obj['color'])
                            obj_encoding_list.append(object_encoding(obj['size'], obj['material'], obj['shape'], obj['color']))
                            num_objects = idx+1 #store the num of objects 
                        #fill in background objects
                        for idx in range(num_objects, self.obj_num):
                            target_query += " :- not object(o{}, bg, bg, bg, bg).".format(idx+1)
                            obj_encoding_list.append([0,0,1, 0,0,1, 0,0,0,1, 0,0,0,0,0,0,0,0,1, 1])
                        self.query_map[count] = target_query
                        self.obj_map[count] = np.array(obj_encoding_list)
                        count += 1
                else:
                    num_objects=0
                    for idx, obj in enumerate(scene['objects']):
                        target_query += " :- not object(o{}, {}, {}, {}, {}).".format(idx+1, obj['size'], obj['material'], obj['shape'], obj['color'])
                        obj_encoding_list.append(object_encoding(obj['size'], obj['material'], obj['shape'], obj['color']))
                        num_objects = idx+1 #store the num of objects 
                    #fill in background objects
                    for idx in range(num_objects, 10):
                        target_query += " :- not object(o{}, bg, bg, bg, bg).".format(idx+1)
                        obj_encoding_list.append([0,0,1, 0,0,1, 0,0,0,1, 0,0,0,0,0,0,0,0,1, 1])
                    self.query_map[scene['image_index']] = target_query
                    self.obj_map[scene['image_index']] = np.array(obj_encoding_list)
                
            print("done")
        if self.files_names:
            print(f'Correctly found images {count} out of {len(files_names)}')
                    
        
        #print(np.array(list(self.obj_map.values()))[0:20])
    def __getitem__(self, index):
        #get the image
        img_path = self.img_paths[index]
        img = io.imread(img_path)[:, :, :3]
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.CenterCrop((29, 221,64, 256)), #why do we need to crop?
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        img = transform(img)
        img = (img - 0.5) * 2.0  # Rescale to [-1, 1].

        return {'im':img}, self.query_map[index] ,self.obj_map[index]


                
    def __len__(self):
        return len(self.img_paths)

