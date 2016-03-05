# Kubernetes on CenturyLink Cloud
These scripts handle the creation, deletion and expansion of kubernetes clusters on CenturyLink Cloud. 

You can accomplish all these tasks with a simple single command. And, for those interested in what's under the covers, we used Ansible to perform these tasks and we have made these Ansible playbooks available as well. 

## Find Help
If you run into any problems or want help with anything, we are here to help. Reach out to use via any of the following ways:
- Submit a github issue
- or
- Send an email to kubernetes AT ctl DOT io
- or
- Visit http://info.ctl.io/kubernetes

## Clusters of VMs or Physical Servers, your choice.
- We support Kubernetes clusters on both Virtual Machines or Physical Servers. If you want to use physical servers for the worker nodes (minions), simple use the --minion_type=bareMetal flag.
- For more information on physical servers, visit: [https://www.ctl.io/bare-metal/](https://www.ctl.io/bare-metal/))
- Physical serves are only available in the VA1 and GB3 data centers.
- VMs are available in all 13 of our public cloud locations

## Requirements
The requirements to run this script are:
- A linux host (tested on ubuntu and OSX)
- ansible _version 2.0_ or newer.  If on OSX, try installing with `brew install ansible`.
- python
- pip
- git
- A CenturyLink Cloud account with rights to create new hosts
- An active VPN connection to the centurylink cloud from your linux/ansible host

## Script Installation
After you have all the requirements met, please follow these instructions to install this script.

1) Clone this repository and cd into it.

```
git clone https://github.com/CenturyLinkCloud/adm-kubernetes-on-clc
```

2) Install the CenturyLink Cloud SDK and Ansible Modules

```
sudo pip install -r requirements.txt
```

3) Create the credentials file from the template and use it to set your ENV variables

```
cp ansible/credentials.sh.template ansible/credentials.sh
vi ansible/credentials.sh
source ansible/credentials.sh

```
4) Make sure the computer you are working on has access to the CenturyLink Cloud network. This is done by using a VM inside the CenturyLink Cloud network or having an active VPN connection to the CenturyLink Cloud network. To find out how to configure the VPN connection, [visit here](https://www.ctl.io/knowledge-base/network/how-to-configure-client-vpn/)


### Script Installation Example: Ubuntu 14 Walkthrough
If you use an ubuntu 14, for your convenience we have provided a step by step guide to install the requirements and install the script.

```
  # system
  apt-get update
  apt-get install -y git python python-crypto
  curl -O https://bootstrap.pypa.io/get-pip.py
  python get-pip.py

  # installing this repository
  mkdir -p ~home/k8s-on-clc
  cd ~home/k8s-on-clc
  git clone https://github.com/CenturyLinkCloud/adm-kubernetes-on-clc.git
  cd adm-kubernetes-on-clc/
  pip install -r requirements.txt

  # getting started
  cd ansible
  cp credentials.sh.template credentials.sh; vi credentials.sh
  source credentials.sh
```



## Cluster Creation
To create a new Kubernetes cluster, simply run the kube-up.sh script. A complete list of script options and some examples are listed below.

```
cd ./adm-kubernetes-on-clc
bash kube-up.sh -c="name_of_kubernetes_cluster"
```

It takes about 15 minutes to create the cluster. Once the script completes, it will output some commands that will help you setup kubectl on your machine to point to the new cluster. 


### Cluster Creation: Script Options

```
Usage: kube-up.sh [OPTIONS]
Create servers in the CenturyLinkCloud environment and initialize a Kubernetes cluster
Environment variables CLC_V2_API_USERNAME and CLC_V2_API_PASSWD must be set in
order to access the CenturyLinkCloud API

All options (both short and long form) require arguments, and must include "="
between option name and option value.

     -h (--help)                   display this help and exit
     -c= (--clc_cluster_name=)     set the name of the cluster, as used in CLC group names
     -t= (--minion_type=)          standard -> VM (default), bareMetal -> physical]
     -d= (--datacenter=)           VA1 (default)
     -m= (--minion_count=)         number of kubernetes minion nodes
     -mem= (--vm_memory=)          number of GB ram for each minion
     -cpu= (--vm_cpu=)             number of virtual cps for each minion node
     -phyid= (--server_conf_id=)   physical server configuration id, one of
                                      physical_server_20_core_conf_id
                                      physical_server_12_core_conf_id
                                      physical_server_4_core_conf_id (default)
     -etcd_separate_cluster=yes    create a separate cluster of three etcd nodes,
                                   otherwise run etcd on the master node
```

## Cluster Expansion
To expand an existing Kubernetes cluster, simply run the add-kube-node.sh script. A complete list of script options and some examples are listed below. This script must be run from the same hose that created the cluster (or a host that has the cluster artifact files stored in ~/.clc_kube/$cluster_name). 

```
cd ./adm-kubernetes-on-clc
bash add-kube-node.sh -c="name_of_kubernetes_cluster" -m=2
```

### Cluster Expansion: Script Options

```
Usage: add-kube-node.sh [OPTIONS]
Create servers in the CenturyLinkCloud environment and add to an
existing CLC kubernetes cluster

Environment variables CLC_V2_API_USERNAME and CLC_V2_API_PASSWD must be set in
order to access the CenturyLinkCloud API

     -h (--help)                   display this help and exit
     -c= (--clc_cluster_name=)     set the name of the cluster, as used in CLC group names
     -m= (--minion_count=)         number of kubernetes minion nodes to add
     
```

## Cluster Deletion
There are two ways to delete an existing cluster: 

1) Use our python script: 

```
python delete_cluster.py --cluster=clc_cluster_name --datacenter=DC1

```

2) Use the CenturyLink Cloud UI. To delete a cluster, log into the CenturyLink Cloud control portal and delete the
parent server group that contains the Kubernetes Cluster. We hope to add a
scripted option to do this soon.

## Examples
Create a cluster with name of k8s_1, 1 master node and 3 worker minions (on physical machines), in VA1

```
 bash kube-up.sh --clc_cluster_name=k8s_1 --minion_type=bareMetal --minion_count=3 --datacenter=VA1
```

Create a cluster with name of k8s_2, an ha etcd cluster on 3 VMs and 6 worker minions (on VMs), in VA1

```
 bash kube-up.sh --clc_cluster_name=k8s_2 --minion_type=standard --minion_count=6 --datacenter=VA1 --etcd_separate_cluster=yes
```

Create a cluster with name of k8s_3, 1 master node, and 10 worker minions (on VMs) with higher mem/cpu, in UC1:

```
  bash kube-up.sh --clc_cluster_name=k8s_3 --minion_type=standard --minion_count=10 --datacenter=VA1 -mem=6 -cpu=4
```



## Cluster Features and Architecture
We configue the Kubernetes cluster with the following features:

* KubeDNS: DNS resolution and service discovery
* Heapster/InfluxDB: For metric collection. Needed for Grafana and auto-scaling. 
* Grafana: Kubernetes/Docker metric dashboard
* KubeUI: Simple web interface to view kubernetes state
* Kube Dashboard: New web interface to interact with your cluster

We use the following to create the kubernetes cluster:

* Kubernetes 1.1.7
* Unbuntu 14.04
* Flannel 0.5.4
* Docker 1.9.1-0~trusty
* Etcd 2.2.2

## Optional add-ons

* Logging: We offer an integrated centralized logging ELK platform so that all kubernetes and docker logs get sent to the ELK stack. To install the ELK stack and configure kubernetes to send logs to it, follow this documentation: [log aggregation](log_aggregration.md). Note: We don't install this by default as the footprint isn't trivial. 

## Cluster management

The most widely used tool for managing a kubernetes cluster is the command-line
utility _kubectl_.  If you do not already have a copy of this binary on your
administrative machine, you may run the script _install-kubectl.sh_ which will
download it and install it in _/usr/bin/local_.

The script requires that the environment variable CLC_CLUSTER_NAME be defined

_install_kubectl.sh_ also writes a configuration file which will embed the necessary
authentication certificates for the particular cluster.  The configuration file is
written to the local directory, named *kubectl_${CLC_CLUSTER_NAME}_config*

```
export KUBECONFIG=kubectl_${CLC_CLUSTER_NAME}_config
kubectl version
kubectl cluster-info
```

### Accessing the cluster programmatically

It's possible to use the locally-stored client certificates to access the api server
```
curl \
   --cacert ${CLC_CLUSTER_HOME}/pki/ca.crt  \
   --key ${CLC_CLUSTER_HOME}/pki/kubecfg.key \
   --cert ${CLC_CLUSTER_HOME}/pki/kubecfg.crt  https://${MASTER_IP}:6443
```
But please note, this *does not* work out of the box with the curl binary
distributed with OSX

### Accessing the cluster with a browser

We install two UIs on kubernetes. The orginal KubeUI and the newer kube dashboard. When you create a cluster, the script should output URLs for these interfaces like this:

KubeUI is running at https://${MASTER_IP}:6443/api/v1/proxy/namespaces/kube-system/services/kube-ui
kubernetes-dashboard is running at https://${MASTER_IP}:6443/api/v1/proxy/namespaces/kube-system/services/kubernetes-dashboard

Note on Authentication to the UIs: The cluster is set up to use basic authentication for the user _admin_.  
Hitting the url at https://${MASTER_IP}:6443 will require accepting the self-
signed certificate from the apiserver, and then presenting the admin password
found at _${CLC_CLUSTER_HOME}/kube/admin_password.txt_


### Configuration files

Various configuration files are written into the home directory under
_.clc_kube/${CLC_CLUSTER_NAME}_ in several subdirectories. You can use these files
to access the cluster from machines other than where you created the cluster from. 

* _config/_: ansible variable files containing parameters describing the master and minion hosts
* _hosts/_: hosts files listing access information for the ansible playbooks
* _kube/_: kubectl configuration files, and the basic-authentication password for admin access to the kubernetes api
* _pki/_: public key infrastructure files enabling TLS communication in the cluster
* _ssh/_: ssh keys for root access to the hosts


## _kubectl_ usage examples

There are a great many features of _kubectl_.  Here are a few examples

List existing nodes, pods, services and more, in all namespaces, or in just one
```
kubectl get nodes

kubectl get --all-namespaces services

kubectl get --namespace=kube-system replicationcontrollers

```

The kubernetes api server exposes services on web urls, which are protected by requiring
client certificates.  If you run a kubectl proxy locally, kubectl will provide
the necessary certificates and serve locally over http.
```
kubectl proxy -p 8001
```
and then access urls like http://127.0.0.1:8001/api/v1/proxy/namespaces/kube-system/services/kube-ui/
without the need for client certificates in your browser.


## What Kubernetes features do not work on CenturyLink Cloud
- At this time, there is no support services of the type 'loadbalancer'. We are actively working on this and hope to publish the changes soon.
- At this time, there is no support for persistent storage volumes provided by CenturyLink Cloud. However, customers can bring their own persistent storage offering. We ourselves use Gluster. 

## Ansible Files
If you want more information about our ansible files, please [read this file](ansible/README.md)

## License
The project is licensed under the [Apache License v2.0](http://www.apache.org/licenses/LICENSE-2.0.html).

