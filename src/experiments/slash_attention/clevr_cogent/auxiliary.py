import json
import os
from pathlib import Path


def get_files_names_and_paths(root:str='./data/CLEVR_v1.0/', mode:str='val', obj_num:int=4):
    data_file = Path(os.path.join(root, 'scenes','CLEVR_'+ mode+"_scenes.json"))
    data_file.parent.mkdir(parents=True, exist_ok=True)
    if data_file.exists():
        print('File exists. Parsing file...')
    else:
        print(f'The JSON file {data_file} does not exist!')
        quit()
    img_paths = []
    files_names = []
    with open(data_file, 'r') as json_file:
        json_data = json.load(json_file)

        for scene in json_data['scenes']:
            if len(scene['objects']) <= obj_num:
                img_paths.append(Path(os.path.join(root,'images/'+mode+'/'+scene['image_filename'])))
                files_names.append(scene['image_filename'])

    print("...done ")
    return img_paths, files_names


def get_slash_program(obj_num:int=4):
    program = ''
    if obj_num == 10:
        program ='''
    slot(s1).
    slot(s2).
    slot(s3).
    slot(s4).
    slot(s5).
    slot(s6).
    slot(s7).
    slot(s8).
    slot(s9).
    slot(s10).

    name(o1).
    name(o2).
    name(o3).
    name(o4).
    name(o5).
    name(o6).
    name(o7).
    name(o8).
    name(o9).
    name(o10).
        '''
    elif obj_num ==4:
        program ='''
    slot(s1).
    slot(s2).
    slot(s3).
    slot(s4).

    name(o1).
    name(o2).
    name(o3).
    name(o4).
        '''
    elif obj_num ==6:
        program ='''
    slot(s1).
    slot(s2).
    slot(s3).
    slot(s4).
    slot(s5).
    slot(s6).

    name(o1).
    name(o2).
    name(o3).
    name(o4).
    name(o5).
    name(o6).
        '''
    else:
        print(f'The number of objects {obj_num} is wrong!')
        quit()
    program +='''        
    %assign each name a slot
    %{slot_name_comb(N,X): slot(X)}=1 :- name(N). %problem we have dublicated slots

    %remove each model which has multiple slots asigned to the same name
    %:- slot_name_comb(N1,X1), slot_name_comb(N2,X2), X1 == X2, N1 != N2. 

    %build the object ontop of the slot assignment
    object(N, S, M, P, C) :- size(0, +X, -S), material(0, +X, -M), shape(0, +X, -P), color(0, +X, -C), slot(X), name(N), slot_name_comb(N,X).

    %define the SPNs
    npp(size(1,X),[small, large, bg]) :- slot(X).
    npp(material(1,X),[rubber, metal, bg]) :- slot(X).
    npp(shape(1,X),[cube, sphere, cylinder, bg]) :- slot(X).
    npp(color(1,X),[gray, red, blue, green, brown, purple, cyan, yellow, bg]) :- slot(X).

    '''
    return program