print("start importing...")

import time
import sys
import argparse
import datetime

sys.path.append('../../')
sys.path.append('../../SLASH/')
sys.path.append('../../EinsumNetworks/src/')


#torch, numpy, ...
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import torchvision

import numpy as np

#own modules
from dataGen import MNIST_Addition
from einsum_wrapper import EiNet
from network_nn import Net_nn

#import slash
from slash import SLASH

import utils
from utils import set_manual_seed
from pathlib import Path
from rtpt import RTPT

print("...done")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate of model"
    )
    parser.add_argument(
        "--network-type",
        choices=["nn","pc"],
        help="The type of external to be used e.g. neural net or probabilistic circuit",
    )
    parser.add_argument(
        "--pc-structure",
        choices=["poon-domingos","binary-trees"],
        help="The type of external to be used e.g. neural net or probabilistic circuit",
    )

    parser.add_argument(
        "--method",
        choices=["exact","top_k","same"],
        help="How many images should be used in the addition",
    )
    parser.add_argument(
        "--k", type=int, default=0, help="Maximum number of stable model to be used"
    )
    parser.add_argument(
        "--images-per-addition",
        choices=["2","3","4","6"],
        help="How many images should be used in the addition",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=6, help="Number of threads for data loader"
    )

    parser.add_argument(
        "--p-num", type=int, default=8, help="Number of processes to devide the batch for parallel processing"
    )

    parser.add_argument("--credentials", type=str, help="Credentials for rtpt")


    args = parser.parse_args()

    if args.network_type == 'pc':
        args.use_pc = True
    else:
        args.use_pc = False

    return args


def slash_mnist_addition():

    args = get_args()
    print(args)

    
    # Set the seeds for PRNG
    set_manual_seed(args.seed)

    # Create RTPT object
    rtpt = RTPT(name_initials=args.credentials, experiment_name='SLASH MNIST pick-k', max_iterations=args.epochs)

    # Start the RTPT tracking
    rtpt.start()
    
    i_num = int(args.images_per_addition)
    if i_num == 2:
        program = '''
        img(i1). img(i2).
        addition(A,B,N):- digit(0,+A,-N1), digit(0,+B,-N2), N=N1+N2, A!=B.
        npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
        '''

    elif i_num == 3:
        program = '''
        img(i1). img(i2). img(i3).
        addition(A,B,C,N):- digit(0,+A,-N1), digit(0,+B,-N2), digit(0,+C,-N3), N=N1+N2+N3, A!=B, A!=C, B!=C.
        npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
        '''
    
    elif i_num == 4:
        program = '''
        img(i1). img(i2). img(i3). img(i4).
        addition(A,B,C,D,N):- digit(0,+A,-N1), digit(0,+B,-N2), digit(0,+C,-N3), digit(0,+D,-N4), N=N1+N2+N3+N4, A != B, A!=C, A!=D, B!= C, B!= D, C!=D.
        npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
        '''

    elif i_num == 6:
        program = '''
        img(i1). img(i2). img(i3). img(i4). img(i5). img(i6).
        addition(i1,i2,i3,i4,i5,i6,N):- digit(0,+A,-N1), digit(0,+B,-N2), digit(0,+C,-N3), digit(0,+D,-N4), digit(0,+E,-N5), digit(0,+F,-N6), N=N1+N2+N3+N4+N5+N6, A != B, A!=C, A!=D, A!=E, A!=F, B!= C, B!= D, B!=E, B!=F, C!=D, C!=E, C!=F, D!=E, D!=F, E!=F.
        npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
        '''
    
    exp_name= str(args.method)+"/" +args.network_type+"_i"+str(i_num)+"_k"+ str(args.k)

    saveModelPath = 'data/'+exp_name+'/slash_digit_addition_models_seed'+str(args.seed)+'.pt'
    Path("data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)

    
    #use neural net or probabilisitc circuit
    if args.network_type == 'pc':
    
        #setup new SLASH program given the network parameters
        if args.pc_structure == 'binary-trees':
            m = EiNet(structure = 'binary-trees',
                      depth = 3,
                      num_repetitions = 20,
                      use_em = False,
                      num_var = 784,
                      class_count = 10,
                      learn_prior = True)
        elif args.pc_structure == 'poon-domingos': 
            m = EiNet(structure = 'poon-domingos',
                      pd_num_pieces = [4,7,28],
                      use_em = False,
                      num_var = 784,
                      class_count = 10,
                      pd_width = 28,
                      pd_height = 28,
                      learn_prior = True)
        else:
            print("pc structure learner unknown")

    else:
        m = Net_nn()    

    
    #trainable paramas
    num_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in m.parameters())
    print("training with {} trainable params and {} params in total".format(num_trainable_params,num_params))
            
        
    #create the SLASH Program
    nnMapping = {'digit': m}
    optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=args.lr, eps=1e-7)}
    SLASHobj = SLASH(program, nnMapping, optimizers)
    SLASHobj.grad_comp_device ='cpu' #set gradient computation to cpu


    #metric lists
    train_accuracy_list = []
    test_accuracy_list = []
    confusion_matrix_list = []
    loss_list = []
    startTime = time.time()

    forward_time_list = []
    asp_time_list = []
    backward_time_list = []
    sm_per_batch_list = [] 
    train_test_times = []
    
    #load data
    #if we are using spns we need to flatten the data(Tensor has form [bs, 784])
    if args.use_pc: 
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, )), transforms.Lambda(lambda x: torch.flatten(x))])
    #if not we can keep the dimensions(Tensor has form [bs,28,28])
    else: 
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))]) 
    data_path = 'data/labels/train_data_s'+str(i_num)+'.txt'

    mnist_addition_dataset = MNIST_Addition(torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform), data_path, i_num, args.use_pc)
    train_dataset_loader = torch.utils.data.DataLoader(mnist_addition_dataset, shuffle=True,batch_size=args.batch_size,pin_memory=True, num_workers=8)
    
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, transform=transform), batch_size=100, shuffle=True)
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, transform=transform), batch_size=100, shuffle=True)

    
    # Evaluate the performanve directly after initialisation
    time_test = time.time()
    test_acc, _, confusion_matrix = SLASHobj.testNetwork('digit', test_loader, ret_confusion=True)
    train_acc, _ = SLASHobj.testNetwork('digit', train_loader)
    confusion_matrix_list.append(confusion_matrix)
    train_accuracy_list.append([train_acc,0])
    test_accuracy_list.append([test_acc, 0])
    timestamp_test = utils.time_delta_now(time_test, simple_format=True)
    timestamp_total = utils.time_delta_now(startTime, simple_format=True)

    train_test_times.append([0.0, timestamp_test, timestamp_total])

    # Save and print statistics
    print('Train Acc: {:0.2f}%, Test Acc: {:0.2f}%'.format(train_acc, test_acc))
    print('--- train time:  ---', 0)
    print('--- test time:  ---' , timestamp_test)
    print('--- total time from beginning:  ---', timestamp_total)

    # Export results and networks
    print('Storing the trained model into {}'.format(saveModelPath))
    torch.save({"addition_net": m.state_dict(),
                "test_accuracy_list": test_accuracy_list,
                "train_accuracy_list":train_accuracy_list,
                "confusion_matrix_list":confusion_matrix_list,
                "num_params": num_trainable_params,
                "args":args,
                "exp_name":exp_name,
                "train_test_times": train_test_times,
                "program":program}, saveModelPath)
    
    start_e= 0

    # Train and evaluate the performance
    for e in range(start_e, args.epochs):
        print('Epoch {}...'.format(e+1))

        #one epoch of training
        time_train= time.time()        
        loss, forward_time, asp_time, backward_time, sm_per_batch, model_computation_time, gradient_computation_time = SLASHobj.learn(dataset_loader = train_dataset_loader,
                       epoch=e, method=args.method, p_num=args.p_num, k_num = args.k , same_threshold=0.99)
        timestamp_train = utils.time_delta_now(time_train, simple_format=True)

        #store detailed timesteps per batch
        forward_time_list.append(forward_time)
        asp_time_list.append(asp_time)
        backward_time_list.append(backward_time)
        sm_per_batch_list.append(sm_per_batch)
        

        time_test = time.time()
        test_acc, _, confusion_matrix = SLASHobj.testNetwork('digit', test_loader, ret_confusion=True)
        confusion_matrix_list.append(confusion_matrix)
        train_acc, _ = SLASHobj.testNetwork('digit', train_loader)        
        train_accuracy_list.append([train_acc,e])
        test_accuracy_list.append([test_acc, e])
        timestamp_test = utils.time_delta_now(time_test, simple_format=True)
        timestamp_total = utils.time_delta_now(startTime, simple_format=True)
        loss_list.append(loss)
        train_test_times.append([timestamp_train, timestamp_test, timestamp_total])

        # Save and print statistics
        print('Train Acc: {:0.2f}%, Test Acc: {:0.2f}%'.format(train_acc, test_acc))
        print('--- train time:  ---', timestamp_train)
        print('--- test time:  ---' , timestamp_test)
        print('--- total time from beginning:  ---', timestamp_total)
        
        # Export results and networks
        print('Storing the trained model into {}'.format(saveModelPath))
        torch.save({"addition_net": m.state_dict(),
                    "resume": {
                        "optimizer_digit":optimizers['digit'].state_dict(),
                        "epoch":e
                            },
                    "test_accuracy_list": test_accuracy_list,
                    "train_accuracy_list":train_accuracy_list,
                    "confusion_matrix_list":confusion_matrix_list,
                    "num_params": num_trainable_params,
                    "args":args,
                    "exp_name":exp_name,
                    "train_test_times": train_test_times,
                    "forward_time_list":forward_time_list,
                    "asp_time_list":asp_time_list,
                    "backward_time_list":backward_time_list,
                    "sm_per_batch_list":sm_per_batch_list,
                    "loss": loss_list,
                    "program":program}, saveModelPath)
        
        # Update the RTPT
        rtpt.step(subtitle=f"accuracy={test_acc:2.2f}")




if __name__ == "__main__":
    slash_mnist_addition()