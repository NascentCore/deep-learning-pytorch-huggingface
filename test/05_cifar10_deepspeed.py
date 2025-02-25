import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed

# define Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='gloo',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_dataset(mode="cpu", batch_size=64, num_workers=2, local_rank=0):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainsampler = None
    trainloader = None

    # 1. cpu or cuda or DataParallel
    if mode == "cpu" or mode == "gpu" or mode == "DataParallel":
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 2. DistributedDataParallel
    elif mode == "DistributedDataParallel":
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, rank=local_rank)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=trainsampler)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainset, testset, trainloader, testloader, trainsampler

# show some images
def sample_imshow(trainloader, batch_size=64, save_dir='./sample'):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'sample_grid.png')
    sample_grid = torchvision.utils.make_grid(images, nrow=int(batch_size ** 0.5), normalize=True, pad_value=1)
    torchvision.utils.save_image(sample_grid, filename)
    print(">>> save image to {}".format(filename))
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def train(log_step_interval=100, save_step_interval=1000, eval_step_interval=200, save_path="ckpt", mode="cpu", epochs=10, batch_size=64, resume=""):
    save_path = f"{save_path}_{mode}"

    if mode == "cpu":
        device = torch.device("cpu")
    elif mode == "gpu" or mode == "DataParallel":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif mode == "DistributedDataParallel":
        n_gpus = 2
        torch.distributed.init_process_group("nccl", world_size=n_gpus)
        local_rank = torch.distributed.get_rank()
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
    elif mode == "deepspeed":
        ####### 1. init deepspeed #######
        args = get_args()
        deepspeed.init_distributed(dist_backend=args.backend, dist_init_required=True)
        args.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

    print(">>> use device : {}".format(device))

    # 1. load dataset
    trainset, testset, trainloader, _, _ = get_dataset(mode=mode, batch_size=batch_size)

# 2. define model
    model = Net()
    if args.local_rank == 0:
        print(">>> model param sum : {}".format(sum(p.numel() for p in model.parameters())))
    
    # resume
    if resume != "":
        # load resume pth -> use [cpu/cuda/cuda:index]
        checkpoint = torch.load(resume, map_location=torch.device("cuda:0"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']

    ####### 2. model && data move to deepspeed #######
    parameters = [p for p in model.parameters() if p.requires_grad]
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters, training_data=trainset)
    fp16 = model_engine.fp16_enabled()
    print(f'fp16={fp16}')       

    # 3. define loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 4. train
    epoch_index = 0
    start_epoch = 0
    num_batches = len(trainloader)
    for epoch_index in range(start_epoch, start_epoch+epochs+1):
        ema_loss = 0.0
        for batch_index, data in enumerate(trainloader):
            step = num_batches*(epoch_index) + batch_index + 1
            ####### 3. data move to model_engine.local_rank #######
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)
            if fp16:
                inputs = inputs.half()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            ####### 4. use model_engine.backward and model_engine.step #######
            model_engine.backward(loss)
            model_engine.step()
            ema_loss = 0.9*ema_loss + 0.1*loss

            # log loss
            if step % log_step_interval == 0 and args.local_rank == 0:
                print(">>> epoch: {:5}, step: {:7}, loss: {:2.4f}".format(epoch_index, step, ema_loss.item()))
            
            # 5. save model pth
            if step % save_step_interval == 0 and args.local_rank == 0:
                os.makedirs(save_path, exist_ok=True)
                save_file = os.path.join(save_path, f"cifar_net_step_{step}.pth")
                torch.save({'epoch': epoch_index,
                            ####### 5. use model.state_dict() #######
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, save_file)
                print(">>> save pth to {}".format(save_file))
    if args.local_rank == 0:
        print('>>> Finished Training')

if __name__ == '__main__':
    ####### 6. batch_size = batch_size_origin * gpu #######
    batch_size = 4 * 2
    train(log_step_interval=1000, save_step_interval=10000, eval_step_interval=200, mode="deepspeed", epochs=100, batch_size=batch_size)

# export CUDA_VISIBLE_DEVICES=0,1
# deepspeed cifar10_deepspeed.py --deepspeed_config ds_config.json
