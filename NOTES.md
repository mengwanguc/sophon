```
pip install pympler
pip install grpcio
pip install grpcio-tools
```

```
python -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/data_transfer.proto
```


```
python main.py  ~/data/test-accuracy/imagenette2 -j 16
```

```
python main.py  ~/data/test-accuracy/imagenette2 -j 4 --dataset-mode=remote_alloff --epochs=1
python main.py  ~/data/openimages/40k/ -j 32 --dataset-mode=remote_nooff --epochs=1 --host=10.52.2.143
```

```
taskset -c 1-4 python 
offload_server.py
```

### Set up dataset

Need to run this on both server and client:

```
cd ~
mkdir data
cd data
mkdir test-accuracy
cd test-accuracy
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -zxvf imagenette2.tgz
```


#### OpenImages


```

```

```
python downloader.py ids.txt --download_folder=./images --num_processes=16
python downloader.py ids_40000.txt --download_folder=./40k/train/class0/ --num_processes=16
```


### Limit bandwidth with wondershaper

Install wondershaper

```
cd ~
git clone https://github.com/magnific0/wondershaper.git
cd wondershaper
sudo make install
```

```
sudo wondershaper -a eno1 -d 500000 -u 500000
```

```
sudo wondershaper -c -a eno1
```


### Limit localhost network bandwidth with tc (not working)

```
# Create a root qdisc (queueing discipline) with handle 1:
sudo tc qdisc add dev lo root handle 1: htb default 11

# Create a class under the root qdisc with limited bandwidth (e.g., 1mbit):
sudo tc class add dev lo parent 1: classid 1:1 htb rate 1mbit

# Attach a filter to the class, directing traffic to port 50051 through it:
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 50051 0xffff flowid 1:1

```


On two separate nodes:
```
sudo tc qdisc add dev ens2f0 root handle 1: htb

sudo tc class add dev ens2f0 parent 1: classid 1:1 htb rate 10Mbps

sudo tc filter add dev ens2f0 protocol ip parent 1:0 prio 1 u32 match ip dport 50051 0xffff flowid 1:1

```


```
sudo tc qdisc add dev ens2f0 root handle 1: htb

sudo tc class add dev ens2f0 parent 1: classid 1:1 htb rate 10Mbps

sudo tc filter add dev ens2f0 protocol ip parent 1: prio 1 u32 match ip src 0.0.0.0/0 flowid 1:1
sudo tc filter add dev ens2f0 protocol ip parent 1: prio 1 u32 match ip dst 0.0.0.0/0 flowid 1:1


```





#### delete the existing qdisc

```
sudo tc qdisc del dev lo root
```

```
sudo tc qdisc del dev ens2f0 root
```


#### test using iperf

```
sudo apt update
sudo apt install iperf3
```

server
```
iperf3 -s -p 50051
```

client
```
iperf3 -c 127.0.0.1 -p 50051
```


#### iptable access


```
# this insert the rule to line 8, which is above the reject rule
sudo iptables -I INPUT 4 -p tcp --dport 50051 -j ACCEPT
# sudo iptables -A OUTPUT -p tcp --sport 50051 -j ACCEPT
```


list all rules for INPUT:

```
sudo iptables -L INPUT --line-numbers
```


#### disable all rules
```
sudo iptables -P INPUT ACCEPT
sudo iptables -P FORWARD ACCEPT
sudo iptables -P OUTPUT ACCEPT
sudo iptables -F
```

```
sudo iptables -L
```


### monitor network bandwidth

```
sudo apt-get install iftop  # Debian/Ubuntu
```

Then run:

```
sudo iftop -i ens2f0
```



nload:

```
sudo apt-get install nload  # Debian/Ubuntu
```
```
sudo nload ens2f0

```

