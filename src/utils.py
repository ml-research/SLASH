import numpy as np
import os
import torch
import errno
from PIL import Image
import torch 
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import random


def mkdir_p(path):
    """Linux mkdir -p"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def one_hot(x, K, dtype=torch.float):
    """One hot encoding"""
    with torch.no_grad():
        ind = torch.zeros(x.shape + (K,), dtype=dtype, device=x.device)
        ind.scatter_(-1, x.unsqueeze(-1), 1)
        return ind


def save_image_stack(samples, num_rows, num_columns, filename, margin=5, margin_gray_val=1., frame=0, frame_gray_val=0.0):
    """Save image stack in a tiled image"""

    # for gray scale, convert to rgb
    if len(samples.shape) == 3:
        samples = np.stack((samples,) * 3, -1)

    height = samples.shape[1]
    width = samples.shape[2]

    samples -= samples.min()
    samples /= samples.max()

    img = margin_gray_val * np.ones((height*num_rows + (num_rows-1)*margin, width*num_columns + (num_columns-1)*margin, 3))
    for h in range(num_rows):
        for w in range(num_columns):
            img[h*(height+margin):h*(height+margin)+height, w*(width+margin):w*(width+margin)+width, :] = samples[h*num_columns + w, :]

    framed_img = frame_gray_val * np.ones((img.shape[0] + 2*frame, img.shape[1] + 2*frame, 3))
    framed_img[frame:(frame+img.shape[0]), frame:(frame+img.shape[1]), :] = img

    img = Image.fromarray(np.round(framed_img * 255.).astype(np.uint8))

    img.save(filename)


def sample_matrix_categorical(p):
    """Sample many Categorical distributions represented as rows in a matrix."""
    with torch.no_grad():
        cp = torch.cumsum(p[:, 0:-1], -1)
        rand = torch.rand((cp.shape[0], 1), device=cp.device)
        rand_idx = torch.sum(rand > cp, -1).long()
        return rand_idx


def set_manual_seed(seed:int=1):
    """Set the seed for the PRNGs."""
    os.environ['PYTHONASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.benchmark = True
    

def time_delta_now(t_start: float, simple_format=False, ret_sec=False) -> str:
    """
                    
    Convert a timestamp into a human readable timestring.
    Parameters
    ----------
    t_start : float
        The timestamp describing the begin of any event.
    Returns
    -------
    Human readable timestring.
    """
    a = t_start
    b = time.time()  # current epoch time
    c = b - a  # seconds
    days = round(c // 86400)
    hours = round(c // 3600 % 24)
    minutes = round(c // 60 % 60)
    seconds = round(c % 60)
    millisecs = round(c % 1 * 1000)
    if simple_format:
        return f"{hours}h:{minutes}m:{seconds}s"

    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds", c

def time_delta(c: float, simple_format=False,) -> str:
    c# seconds
    days = round(c // 86400)
    hours = round(c // 3600 % 24)
    minutes = round(c // 60 % 60)
    seconds = round(c % 60)
    millisecs = round(c % 1 * 1000)
    if simple_format:
        return f"{hours}h:{minutes}m:{seconds}s"
    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds", c


def export_results(test_accuracy_list, train_accuracy_list,
                   export_path, export_suffix, 
                   confusion_matrix , exp_dict):
    
    #set matplotlib styles
    plt.style.use(['science','grid'])
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,
            "legend.fontsize": 22
        }
    )

    
    fig, axs = plt.subplots(1, 1 , figsize=(10,21))
    # fig.suptitle(exp_dict['exp_name'], fontsize=16)
    #axs[0]
    axs.plot(test_accuracy_list[:,1],test_accuracy_list[:,0], label='test accuracy')
    #axs[0]
    axs.plot(train_accuracy_list[:,1],train_accuracy_list[:,0], label='train accuracy')
    #axs[0].
    axs.legend(loc="lower right")
    #axs[0].
    axs.set(xlabel='epochs', ylabel='accuracy')
    #axs[0].
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))


    #ax.[0, 1].set_xticklabels([0,1,2,3,4,5,6,7,8,9])
    #ax.[0, 1].set_yticklabels([0,1,2,3,4,5,6,7,8,9])

    #axs[0, 1].set(xlabel='target', ylabel='prediction')
    #axs[1] = sns.heatmap(confusion_matrix, linewidths=2, cmap="viridis")


    
    if exp_dict['structure'] == 'poon-domingos':
        text = "trainable parameters = {}, lr = {}, batchsize= {}, time_per_epoch= {},\n structure= {}, pd_num_pieces = {}".format(
            exp_dict['num_trainable_params'], exp_dict['lr'],
            exp_dict['bs'], exp_dict['train_time'],            
            exp_dict['structure'], exp_dict['pd_num_pieces'] )    
        
    else:
        text = "trainable parameters = {}, lr = {}, batchsize= {}, time_per_epoch= {},\n structure= {}, num_repetitions = {} , depth = {}".format(
            exp_dict['num_trainable_params'], exp_dict['lr'],
            exp_dict['bs'],exp_dict['train_time'],
            exp_dict['structure'], exp_dict['num_repetitions'], exp_dict['depth'])
        
    #plt.gcf().text(
    #0.5,
    #0.02,
    #text,
    #ha="center",
    #fontsize=12,
    #linespacing=1.5,
    #bbox=dict(
    #    facecolor="grey", alpha=0.2, edgecolor="black", boxstyle="round,pad=1"
    #))
    
    fig.savefig(export_path, format="svg")
    
    plt.show()
    
    
    #Tensorboard outputs
    writer = SummaryWriter("../../results", filename_suffix=export_suffix)

    for train_acc_elem, test_acc_elem in zip(train_accuracy_list, test_accuracy_list):
        writer.add_scalar('Accuracy/train', train_acc_elem[0], train_acc_elem[1])
        writer.add_scalar('Accuracy/test', test_acc_elem[0], test_acc_elem[1])

        
