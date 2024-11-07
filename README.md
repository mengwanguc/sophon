# SOPHON

**S**electively **O**ffloading **P**reprocessing with **H**ybrid **O**perations **N**ear-storage

## Preparations

Currently I use *2 RTX-6000 nodes* on Chameleon, which ensures heterogeous CPU type on both nodes. However, we only use the GPU on the first node.

You should run the following commands on both nodes.

1. Chameleon image

We used Ubuntu 20 and CUDA 11.

2. Set up ssh
```
ssh-keygen -t rsa -b 4096
```

```
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
cat ~/.ssh/id_rsa.pub
```

Copy and paste into: https://github.com/settings/keys

Configure your username and email.
```
git config --global user.name "FIRST_NAME LAST_NAME"
git config --global user.email "MY_NAME@example.com"
```

for example:

```
git config --global user.name "Meng Wang"
git config --global user.email "mengwanguc@gmail.com"
```



3. clone this repo to local

```
git clone git@github.com:mengwanguc/sophon.git
```

4. Install conda

```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
```
After installation, log out and log in bash again.

5. Install packages required for builing pytorch

**Note: the commands below assumes you have cuda 11.2 installed on your machine. If you have other cuda versions, please use the magma-cuda\* that matches your CUDA version from** https://anaconda.org/pytorch/repo.

For example, here our cuda version is 11.2 (check it by running nvidia-smi), that's why the command is `conda install -y -c pytorch magma-cuda112`

```
conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# CUDA only: Add LAPACK support for the GPU if needed
conda install -y -c pytorch magma-cuda112  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
```

6. Install CuDNN:

Install libcudnn from apt

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install libcudnn8=8.1.1.33-1+cuda11.2
```

Download headers
```
cd ~
mkdir cudnn
cd cudnn
pip install gdown
gdown https://drive.google.com/uc?id=1IJGIH7Axqd8E5Czox_xRDCLOyvP--ej-
tar -xvf cudnn-11.2-linux-x64-v8.1.1.33.tar
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```



7. Download our custom pytorch and build it

```
cd ~
git clone git@github.com:mengwanguc/pytorch-sophon.git
cd pytorch-sophon
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

8. Install torchvision

```
cd ~
git clone git@github.com:mengwanguc/torchvision-sophon.git
cd torchvision-sophon
python setup.py install
cd ~
```

## Download datasets

We now have data on *both compute node and storage node*. 

When we read data from remote storage node, we:
- Scan the local data folder to get the list of filepaths 
- Shuffle and generate samples of filepaths
- Send the requests to the remote storage node with sample's filepaths

Therefore, we require both nodes to have the same directory structure for the data.

This certainly can be improved. But currently we just use this method for quick evaluations.

### OpenImages


We download 40K images from OpenImages. This requires using boto3:

```
pip install boto3
```

```
cd ~
mkdir data
cp -r ~/sophon/data/openimages -r ~/data/
cd data/openimages/
# this creates a sample subset of openimages dataset for fast evaluation
python downloader.py ids_40000.txt --download_folder=./40k/train/class0/ --num_processes=16
# we don't run validation. But for the data loader to run, we create a dummy validation folder
mkdir -p ~/data/openimages/40k/val/class0
touch ~/data/openimages/40k/val/class0/empty.jpg
```

## Setup network

### Install grpc
```
pip install pympler
pip install grpcio==1.62.1 grpcio-tools==1.62.1
```

If you have updated the proto, then run this command to re-generate the grpc scripts:

```
cd ~/sophon
python -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/data_transfer.proto
```

### Allow network access between compute node and storage node

On storage node, run:

```
# this insert the rule to line 4, which is above the reject rule
sudo iptables -I INPUT 4 -p tcp --dport 50051 -j ACCEPT
sudo iptables -I OUTPUT 4 -p tcp --sport 50051 -j ACCEPT
```

list all rules for INPUT:

```
sudo iptables -L INPUT --line-numbers
```


## Run the application

### Run storage server

On the storage node, run:

```
python offload_server.py --grpc-workers=32 --prep-workers=12
```

Here `grpc-workers` is the number of threads used for gRPC server's messaging.

`prep-workers` is the number of threads used for performing offloaded preprocessing. 

Each grpc worker sends the preprocessing work to the prep worker.

### Run DL job

On the compue node, run:

```
python main.py  ~/data/openimages/40k/ -a alexnet -j 32 --dataset-mode=remote_nooff --epochs=1 --host=110.140.82.239 --remote-cpus=5 --net-bw=430 --profile-folder='profiles/openimages/'
```

Replace `10.140.82.229` with the IP address of your storage machine. You can get it by running `ifconfig` or `ip a`.

Replace `remote_nooff` with the mode that you want to test. Supported modes:

- remote_nooff
- remote_alloff
- remote_piloff
- remote smartoff

Replace `remote-cpus=5` with the number of CPUs used on remote node for preprocessing

Replace `430` with the network bandwidth. Here I use 430 because it's what's observed from iftop when setting throughput cap to 500Mbps. (see below.)



### Profiling

The profiled data is in `profiles/openimages/`.

If you want to create new profile, you should use the `--write-profile-folder` argument to specify where to write the profiled data.

To run profiling:

```

```

### Limit network bandwidth

Install wondershaper on storage node:

```
cd ~
git clone https://github.com/magnific0/wondershaper.git
cd wondershaper
sudo make install
```

Limit bandwidth to 500Mbps:
(You may need to replace `eno1` with the network interface on your machine. To check which interface to use, run `ifconfig` or `ip a` and pick the network interface responding to the IP address. )

```
sudo wondershaper -a eno1 -d 500000 -u 500000
```


Clear the limit

```
sudo wondershaper -c -a eno1
```

### Monitor the network traffic and throughput

Install `iftop`:

```
sudo apt-get install iftop  # Debian/Ubuntu
```

Then run:

```
sudo iftop -i eno1
```

When `wondershaper` sets the limit to 500000, you will monitor that the real network throughput is capped to ~435Mbps. I don't know why, but this is the same for all experiments. So I think it's fair enough.

At the bottom of the monitor, it records the cumulative data traffic. To start from 0, you just need to stop the iftop (`q` or `ctrl+c`) and restart it. 



## Reproduce our results

To reproduce our results, you can run the following commands:


### With ample CPUs:

On the server side, run:

```
python offload_server.py --grpc-workers=32 --prep-workers=20
```

On the client side:

For all off:
```
python main.py  ~/data/openimages/40k/ -a alexnet -j 32  --dataset-mode=remote_alloff --epochs=1 --host=192.5.87.114'
```

For no off:
```
python main.py  ~/data/openimages/40k/ -a alexnet -j 32  --dataset-mode=remote_nooff --epochs=1 --host=192.5.87.114'
```

For resize off:
```
python main.py  ~/data/openimages/40k/ -a alexnet -j 32  --dataset-mode=remote_piloff --epochs=1 --host=192.5.87.114'
```

For SOPHON:

First run profiling for one epoch (which uses no-off):
```
python main.py  ~/data/openimages/40k/ -a alexnet -j 32  --dataset-mode=remote_sophon --epochs=1 --host=192.5.87.114 --remote-cpus=20 --net-bw=430 --profile-folder='profiles/openimages/' --profiling
```

Then run the epoch with SOPHON offloading:
```
python main.py  ~/data/openimages/40k/ -a alexnet -j 32  --dataset-mode=remote_sophon --epochs=1 --host=192.5.87.114 --remote-cpus=20 --net-bw=430 --profile-folder='profiles/openimages/'
```

### With limited CPUs:

On the server side, run:

```
# This limits the server to 5 CPUs for preprocessing:
python offload_server.py --grpc-workers=32 --prep-workers=5
```

On the client side:

For all off:
```
python main.py  ~/data/openimages/40k/ -a alexnet -j 32  --dataset-mode=remote_alloff --epochs=1 --host=192.5.87.114'
```

For no off:
```
python main.py  ~/data/openimages/40k/ -a alexnet -j 32  --dataset-mode=remote_nooff --epochs=1 --host=192.5.87.114'
```

For resize off:
```
python main.py  ~/data/openimages/40k/ -a alexnet -j 32  --dataset-mode=remote_piloff --epochs=1 --host=192.5.87.114'
```

For SOPHON:

First run profiling for one epoch (which uses no-off):
```
python main.py  ~/data/openimages/40k/ -a alexnet -j 32  --dataset-mode=remote_sophon --epochs=1 --host=192.5.87.114 --remote-cpus=20 --net-bw=430 --profile-folder='profiles/openimages/' --profiling
```

Then run the epoch with SOPHON offloading:
```
python main.py  ~/data/openimages/40k/ -a alexnet -j 32  --dataset-mode=remote_sophon --epochs=1 --host=192.5.87.114 --remote-cpus=5 --net-bw=430 --profile-folder='profiles/openimages/'
```