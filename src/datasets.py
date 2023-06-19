import os
import tempfile
import  urllib.request
import shutil

from zipfile import ZipFile
import gzip
import utils

def maybe_download(directory, url_base, filename, suffix='.zip'):
    '''
    Downloads the specified dataset and extracts it

    @param directory:
    @param url_base: URL where to find the file
    @param filename: name of the file to be downloaded
    @param suffix: suffix of the file

    :returns: true if nothing went wrong downloading 
    '''

    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        return False

    if not os.path.isdir(directory):
        utils.mkdir_p(directory)

    url = url_base  +filename
    
    _, zipped_filepath = tempfile.mkstemp(suffix=suffix)
        
    print('Downloading {} to {}'.format(url, zipped_filepath))
    
    urllib.request.urlretrieve(url, zipped_filepath)
    print('{} Bytes'.format(os.path.getsize(zipped_filepath)))
    
    print('Move to {}'.format(filepath))
    shutil.move(zipped_filepath, filepath)
    return True


def extract_dataset(directory, filepath, filepath_extracted):
    if not os.path.isdir(filepath_extracted):
        print('unzip ',filepath, " to", filepath_extracted)
        with ZipFile(filepath, 'r') as zipObj:
           # Extract all the contents of zip file in current directory
           zipObj.extractall(directory)
            
            
def maybe_download_shapeworld4():
    '''
    Downloads the shapeworld4 dataset if it is not downloaded yet
    '''
        
    directory = "../../data/"
    file_name= "shapeworld4.zip"
    maybe_download(directory, "https://hessenbox.tu-darmstadt.de/dl/fiEE3hftM4n1gBGn4HJLKUkU/", file_name)
    
    filepath = os.path.join(directory, file_name)
    filepath_extracted = os.path.join(directory,"shapeworld4")
    
    extract_dataset(directory, filepath, filepath_extracted)

    
def maybe_download_shapeworld_cogent():
    '''
    Downloads the shapeworld4 cogent dataset if it is not downloaded yet
    '''

    directory = "../../data/"
    file_name= "shapeworld_cogent.zip"
    maybe_download(directory, "https://hessenbox.tu-darmstadt.de/dl/fi3CDjPRsYgAvotHcC8GPaWj/", file_name)
    
    filepath = os.path.join(directory, file_name)
    filepath_extracted = os.path.join(directory,"shapeworld_cogent")
    
    extract_dataset(directory, filepath, filepath_extracted)
