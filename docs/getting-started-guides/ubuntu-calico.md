<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
Kubernetes Deployment On Bare-metal Ubuntu Nodes with Calico Networking
------------------------------------------------

## Introduction

This document describes how to deploy Kubernetes on Ubuntu bare metal nodes with Calico Networking plugin. See [projectcalico.org](http://projectcalico.org) for more information on what Calico is, and [the calicoctl github](https://github.com/projectcalico/calico-docker) for more information on the command-line tool, `calicoctl`.

This guide will set up a simple Kubernetes cluster with a master and two nodes. We will start the following processes with systemd:

On the Master:
- `etcd`
- `kube-apiserver`
- `kube-controller-manager`
- `kube-scheduler`
- `calico-node`

On each Node:
- `kube-proxy`
- `kube-kubelet`
- `calico-node`

## Prerequisites

1. This guide uses `systemd` and thus uses Ubuntu 15.04 which supports systemd natively.
2. All machines should have the latest docker stable version installed. At the time of writing, that is Docker 1.7.0.
	- To install docker, follow [these instructions](https://docs.docker.com/installation/ubuntulinux/)
3. All hosts should be able to communicate with each other, as well as the internet, to download the necessary files.
4. This demo assumes that none of the hosts have been configured with any Kubernetes or Calico software yet.

## Setup Master

First, get the sample configurations for this tutorial

```
wget https://github.com/Metaswitch/calico-kubernetes-ubuntu-demo/archive/master.tar.gz
tar -xvf master.tar.gz
```

### Setup environment variables for systemd services on Master

Many of the sample systemd services provided rely on environment variables on a per-node basis. Here we'll edit those environment variables and move them into place.

1.) Copy the network-environment-template from the `master` directory for editing.

```
cp calico-kubernetes-ubuntu-demo-master/master/network-environment-template network-environment
```

2.) Edit `network-environment` to represent your current host's settings.

3.) Move the `network-environment` into `/etc`

```
sudo mv -f network-environment /etc
```

### Install Kubernetes on Master

1.) Build & Install Kubernetes binaries

```
# Get the Kubernetes Source
wget https://github.com/kubernetes/kubernetes/releases/download/v1.0.3/kubernetes.tar.gz

# Untar it
tar -xf kubernetes.tar.gz
tar -xf kubernetes/server/kubernetes-server-linux-amd64.tar.gz
kubernetes/cluster/ubuntu/build.sh

# Add binaries to /usr/bin
sudo cp -f binaries/master/* /usr/bin
sudo cp -f binaries/kubectl /usr/bin
```

2.) Install the sample systemd processes settings for launching kubernetes services

```
sudo cp -f calico-kubernetes-ubuntu-demo-master/master/*.service /etc/systemd
sudo systemctl enable /etc/systemd/etcd.service
sudo systemctl enable /etc/systemd/kube-apiserver.service
sudo systemctl enable /etc/systemd/kube-controller-manager.service
sudo systemctl enable /etc/systemd/kube-scheduler.service
```

3.) Launch the processes.

```
sudo systemctl start etcd.service
sudo systemctl start kube-apiserver.service
sudo systemctl start kube-controller-manager.service
sudo systemctl start kube-scheduler.service
```

### Install Calico on Master

In order to allow the master to route to pods on our nodes, we will launch the calico-node daemon on our master. This will allow it to learn routes over BGP from the other calico-node daemons in the cluster. The docker daemon should already be running before calico is started.

```
# Install the calicoctl binary, which will be used to launch calico
wget https://github.com/projectcalico/calico-docker/releases/download/v0.5.5/calicoctl
chmod +x calicoctl
sudo cp -f calicoctl /usr/bin

# Install and start the calico service
sudo cp -f calico-kubernetes-ubuntu-demo-master/master/calico-node.service /etc/systemd
sudo systemctl enable /etc/systemd/calico-node.service
sudo systemctl start calico-node.service
```

>Note: calico-node may take a few minutes on first boot while it downloads the calico-node docker image.

## Setup Nodes

Perform these steps **once on each node**, ensuring you appropriately set the environment variables on each node

### Setup environment variables for systemd services on the Node

1.) Get the sample configurations for this tutorial

```
wget https://github.com/Metaswitch/calico-kubernetes-ubuntu-demo/archive/master.tar.gz
tar -xvf master.tar.gz
```

2.) Copy the network-environment-template from the `node` directory

```
cp calico-kubernetes-ubuntu-demo-master/node/network-environment-template network-environment
```

3.) Edit `network-environment` to represent your current host's settings.

4.) Move `network-environment` into `/etc`

```
sudo mv -f network-environment /etc
```

### Configure Docker on the Node

#### Create the veth

Instead of using docker's default interface (docker0), we will configure a new one to use desired IP ranges

```
sudo apt-get install -y bridge-utils
sudo brctl addbr cbr0
sudo ifconfig cbr0 up
sudo ifconfig cbr0 <IP>/24
```

> Replace \<IP\>  with the subnet for this host's containers. Example topology:

  Node   |   cbr0 IP
-------- | -------------
node-1  | 192.168.1.1/24
node-2  | 192.168.2.1/24
node-X  | 192.168.X.1/24

#### Start docker on cbr0

The Docker daemon must be started and told to use the already configured cbr0 instead of using the usual docker0, as well as disabling ip-masquerading and modification of the ip-tables.

1.) Edit the ubuntu-15.04 docker.service for systemd at: `/lib/systemd/system/docker.service`

2.) Find the line that reads `ExecStart=/usr/bin/docker -d -H fd://` and append the following flags: `--bridge=cbr0 --iptables=false --ip-masq=false`

3.) Reload systemctl and restart docker.

```
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### Install Calico on the Node

1.) Install Calico

```
# Get the calicoctl binary
wget https://github.com/projectcalico/calico-docker/releases/download/v0.5.5/calicoctl
chmod +x calicoctl
sudo cp -f calicoctl /usr/bin

# Start calico on this node
sudo cp calico-kubernetes-ubuntu-demo-master/node/calico-node.service /etc/systemd
sudo systemctl enable /etc/systemd/calico-node.service
sudo systemctl start calico-node.service
```

>The calico-node service will automatically get the kubernetes-calico plugin binary and install it on the host system.

2.) Use calicoctl to add an IP pool. We must specify the IP and port that the master's etcd is listening on.
**NOTE: This step only needs to be performed once per Kubernetes deployment, as it covers all the node's IP ranges.**

```
ETCD_AUTHORITY=<MASTER_IP>:4001 calicoctl pool add 192.168.0.0/16
```

### Install Kubernetes on the Node

1.) Build & Install Kubernetes binaries

```
# Get the Kubernetes Source
wget https://github.com/kubernetes/kubernetes/releases/download/v1.0.3/kubernetes.tar.gz

# Untar it
tar -xf kubernetes.tar.gz
tar -xf kubernetes/server/kubernetes-server-linux-amd64.tar.gz
kubernetes/cluster/ubuntu/build.sh

# Add binaries to /usr/bin
sudo cp -f binaries/minion/* /usr/bin

# Get the iptables based kube-proxy reccomended for this demo
wget https://github.com/projectcalico/calico-kubernetes/releases/download/v0.1.1/kube-proxy
sudo cp kube-proxy /usr/bin/
sudo chmod +x /usr/bin/kube-proxy
```

2.) Install and launch the sample systemd processes settings for launching Kubernetes services

```
sudo cp calico-kubernetes-ubuntu-demo-master/node/kube-proxy.service /etc/systemd/
sudo cp calico-kubernetes-ubuntu-demo-master/node/kube-kubelet.service /etc/systemd/
sudo systemctl enable /etc/systemd/kube-proxy.service
sudo systemctl enable /etc/systemd/kube-kubelet.service
sudo systemctl start kube-proxy.service
sudo systemctl start kube-kubelet.service
```

>*You may want to consider checking their status after to ensure everything is running*

## Install the DNS Addon

Most Kubernetes deployments will require the DNS addon for service discovery.  For more on DNS service discovery, check [here](../../cluster/addons/dns/).

The config repository for this guide comes with manifest files to start the DNS addon.  To install DNS, do the following on your Master node.

Replace `<MASTER_IP>` in `calico-kubernetes-ubuntu-demo-master/dns/skydns-rc.yaml` with your Master's IP address.  Then, create `skydns-rc.yaml` and `skydns-svc.yaml` using `kubectl create -f <FILE>`.

## Launch other Services With Calico-Kubernetes

At this point, you have a fully functioning cluster running on kubernetes with a master and 2 nodes networked with Calico. You can now follow any of the [standard documentation](../../examples/) to set up other services on your cluster.

## Connectivity to outside the cluster

With this sample configuration, because the containers have private `192.168.0.0/16` IPs, you will need NAT to allow   connectivity between containers and the internet. However, in a full datacenter deployment, NAT is not always necessary, since Calico can peer with the border routers over BGP.

### NAT on the nodes

The simplest method for enabling connectivity from containers to the internet is to use an iptables masquerade rule. This is the standard mechanism [recommended](../../docs/admin/networking.md#google-compute-engine-gce) in the Kubernetes GCE environment.

We need to NAT traffic that has a destination outside of the cluster. Internal traffic includes the master/nodes, and the container IP pools. A suitable masquerade chain would follow the pattern below, replacing the following variables:
- `CONTAINER_SUBNET`: The cluster-wide subnet from which container IPs are chosen.  All cbr0 bridge subnets fall within this range. The above example uses `192.168.0.0/16`.
- `KUBERNETES_HOST_SUBNET`: The subnet from which Kubernetes node / master IP addresses have been chosen.
- `HOST_INTERFACE`: The interface on the Kubernetes node which is used for external connectivity.  The above example uses `eth0`

```
sudo iptables -t nat -N KUBE-OUTBOUND-NAT
sudo iptables -t nat -A KUBE-OUTBOUND-NAT -d <CONTAINER_SUBNET> -o <HOST_INTERFACE> -j RETURN
sudo iptables -t nat -A KUBE-OUTBOUND-NAT -d <KUBERNETES_HOST_SUBNET> -o <HOST_INTERFACE> -j RETURN
sudo iptables -t nat -A KUBE-OUTBOUND-NAT -j MASQUERADE
sudo iptables -t nat -A POSTROUTING -j KUBE-OUTBOUND-NAT
```

This chain should be applied on the master and all nodes. In production, these rules should be persisted, e.g. with `iptables-persistent`.

### NAT at the border router

In a datacenter environment, it is recommended to configure Calico to peer with the border routers over BGP. This means that the container IPs will be routable anywhere in the datacenter, and so NAT is not needed on the nodes (though it may be enabled at the datacenter edge to allow outbound-only internet connectivity).




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/ubuntu-calico.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
