# Kubernetes deployed on multiple ubuntu nodes

This document describes how to deploy kubernetes on multiple ubuntu nodes, including 1 master node and 3 minion nodes, and people uses this approach can scale to **any number of minion nodes** by changing some settings with ease. Although there exists saltstack based ubuntu k8s installation ,  it may be tedious and hard for a guy that knows little about saltstack but want to build a really distributed k8s cluster. This approach is inspired by [k8s deploy on a single node](http://docs.k8s.io/getting-started-guides/ubuntu_single_node.md).

[Cloud team from ZJU](https://github.com/ZJU-SEL) will keep updating this work.

### **Prerequisites：**
*1 The minion nodes have installed docker version 1.2+* 

*2 All machines can communicate with each orther, no need to connect Internet (should use private docker registry in this case)*

*3 These guide is tested OK on Ubuntu 14.04 LTS 64bit server, but it should also work on most Ubuntu versions*

*4 Dependences of this guide: etcd-2.0.0, flannel-0.2.0, k8s-0.12.0, but it may work with higher versions*


### **Main Steps**
#### I. Make *kubernetes* , *etcd* and *flanneld* binaries

On your laptop, copy `cluster/ubuntu-cluster` directory to your workspace.

The `build.sh` will download and build all the needed binaries into `./binaries`.

You can customize your etcd version or K8s version in the build.sh by changing  variable `ETCD_V` and `K8S_V` in build.sh, default etcd version is 2.0.0 and K8s version is 0.12.0.


```
$ cd cluster/ubuntu-cluster
$ sudo ./build.sh
```

Please copy all the files in `./binaries` into `/opt/bin` of every machine you want to run as Kubernetes cluster node.


Alternatively, if your Kubernetes nodes have access to Internet, you can copy `cluster/ubuntu-cluster` directory to every node and run:
```
# in every node
$ cd cluster/ubuntu-cluster
$ sudo ./build.sh
$ sudo cp ./binaries/* /opt/bin
```


> We used flannel here because we want to use overlay network, but please remember it is not the only choice, and it is also not a k8s' necessary dependence. Actually you can just build up k8s cluster natively, or use flannel, Open vSwitch or any other SDN tool you like, we just choose flannel here as a example.

#### II. Configue and install every components upstart script
An example cluster is listed as below:

| IP Address|Role |      
|---------|------|
|10.10.103.223| minion|
|10.10.103.224| minion|
|10.10.103.162| minion|
|10.10.103.250| master|

First of all, make sure `cluster/ubuntu-cluster` exists on this node，and run `configue.sh`.

On master( infra1 10.10.103.250 ) node:

```
# in cluster/ubuntu-cluster
$ sudo ./configure.sh
Welcome to use this script to configure k8s setup

Please enter all your cluster node ips, MASTER node comes first
And separated with blank space like "<ip_1> <ip2> <ip3>": 10.10.103.250 10.10.103.223 10.10.103.224 10.10.103.162

This machine acts as
  both MASTER and MINION:      1
  only MASTER:                 2
  only MINION:                 3
Please choose a role > 2

IP address of this machine > 10.10.103.250

Configure Success
```

On every minion ( e.g.  10.10.103.224 ) node:


```
# in cluster/ubuntu-cluster
$ sudo ./configure.sh 
Welcome to use this script to configure k8s setup

Please enter all your cluster node ips, MASTER node comes first
And separated with blank space like "<ip_1> <ip2> <ip3>": 10.10.103.250 10.10.103.223 10.10.103.224 10.10.103.162

This machine acts as
  both MASTER and MINION:      1
  only MASTER:                 2
  only MINION:                 3
Please choose a role > 3

IP address of this machine > 10.10.103.224

Configure Success
```

If you want a node acts as **both running the master and minion**, please choose option 1.

#### III. Start all components
1. On the master node:
  
	`$ sudo service etcd start`
	
	Then on every minion node:
	
	`$ sudo service etcd start`
	
	> The kubernetes commands will be started automatically after etcd
  
2. On any node:
	
	`$ /opt/bin/etcdctl mk /coreos.com/network/config '{"Network":"10.0.0.0/16"}'`

	Note the `10.0.0.0/16` is a virtual network address. It has nothing to do with master and minions IP addresses assigned by the cloud provider. In other words even if your master and minions use address from another network (e.g. 172.16.0x) you can still use `10.0.0.0/16` for your virtual network.
	
	> You can use the below command on another node to confirm if the network setting is correct.
	
	> `$ /opt/bin/etcdctl get /coreos.com/network/config`
	
	> If you got `{"Network":"10.0.0.0/16"}`, then etcd cluster is working well.
	> If not , please check` /var/log/upstart/etcd.log` to resolve etcd problem before going forward.
	> Finally, use `ifconfig` to see if there is a new network interface named `flannel0` coming up.
  
  
3. On every minion node
	
	Make sure you have `brctl` installed on every minion, otherwise please run `sudo apt-get install bridge-utils`
	
	`$ sudo ./reconfigureDocker.sh`
	
	This will make the docker daemon aware of flannel network.
 

**All done !**

#### IV. Validation
You can use kubectl command to see if the newly created k8s is working correctly. 

For example , `$ kubectl get minions` to see if you get all your minion nodes comming up. 

Also you can run kubernetes [guest-example](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/guestbook) to build a redis backend cluster on the k8s．

#### V. Trouble Shooting

Generally, what of this guide did is quite simple: 

1. Build and copy binaries and configuration files to proper dirctories on every node

2. Configure `etcd` using IPs based on input from user 

3. Create and start flannel network

So, whenver you have problem, do not blame Kubernetes, **check etcd configuration first** 

Please try:

1. Check `/var/log/upstart/etcd.log` for suspicisous etcd log 

2. Check `/etc/default/etcd`, as we do not have much input validation, a right config should be like:
	```
	ETCD_OPTS="-name infra1 -initial-advertise-peer-urls <http://ip_of_this_node:2380> -listen-peer-urls <http://ip_of_this_node:2380> -initial-cluster-token etcd-cluster-1 -initial-cluster infra1=<http://ip_of_this_node:2380>,infra2=<http://ip_of_another_node:2380>,infra3=<http://ip_of_another_node:2380> -initial-cluster-state new"
	```

3. Remove `data-dir` of etcd and run `reconfigureDocker.sh`again, the default path of `data-dir` is /infra*.etcd/

4. You can also customize your own settings in `/etc/default/{component_name}` after configured success. 
