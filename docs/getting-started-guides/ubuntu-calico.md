<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/getting-started-guides/ubuntu-calico.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Bare Metal Kubernetes with Calico Networking
------------------------------------------------

## Introduction

This document describes how to deploy Kubernetes with Calico networking on _bare metal_ Ubuntu. For more information on Project Calico, visit [projectcalico.org](http://projectcalico.org) and the [calico-docker repository](https://github.com/projectcalico/calico-docker).

To install Calico on an existing Kubernetes cluster, or for more information on deploying Calico with Kubernetes in a number of other environments take a look at our supported [deployment guides](https://github.com/projectcalico/calico-docker/tree/master/docs/kubernetes).

This guide will set up a simple Kubernetes cluster with a single Kubernetes master and two Kubernetes nodes. We will start the following processes with systemd:

On the Master:
- `kubelet`
- `calico-node`

On each Node:
- `kubelet`
- `kube-proxy`
- `calico-node`

## Prerequisites

1. This guide uses `systemd` for process management. Ubuntu 15.04 supports systemd natively as do a number of other Linux distributions.
2. All machines should have Docker >= 1.7.0 installed.
	- To install Docker on Ubuntu, follow [these instructions](https://docs.docker.com/installation/ubuntulinux/)
3. All machines should have connectivity to each other and the internet.
4. This demo assumes that none of the hosts have been configured with any Kubernetes or Calico software.

## Setup Master

Download the `calico-kubernetes` repository, which contains the necessary configuration for this guide.

```
wget https://github.com/projectcalico/calico-kubernetes/archive/master.tar.gz
tar -xvf master.tar.gz
```

### Install Kubernetes on Master

We'll use the `kubelet` to bootstrap the Kubernetes master processes as containers.

1.) Download and install the `kubelet` and `kubectl` binaries.

```
# Get the Kubernetes Release.
wget https://github.com/kubernetes/kubernetes/releases/download/v1.1.0/kubernetes.tar.gz

# Extract the Kubernetes binaries.
tar -xf kubernetes.tar.gz
tar -xf kubernetes/server/kubernetes-server-linux-amd64.tar.gz

# Install the `kubelet` and `kubectl` binaries.
sudo cp -f kubernetes/server/bin/kubelet /usr/bin
sudo cp -f kubernetes/server/bin/kubectl /usr/bin
```

2.) Install the `kubelet` systemd unit file and start the `kubelet`.

```
# Install the unit file
sudo cp -f calico-kubernetes-master/config/master/kubelet.service /etc/systemd

# Enable the unit file so that it runs on boot
sudo systemctl enable /etc/systemd/kubelet.service

# Start the `kubelet` service
sudo systemctl start kubelet.service
```

3.) Start the other Kubernetes master services using the provided manifest.

```
# Install the provided manifest
sudo mkdir -p /etc/kubernetes/manifests
sudo cp -f calico-kubernetes-master/config/master/kubernetes-master.manifest /etc/kubernetes/manifests
```

You should see the `apiserver`, `controller-manager` and `scheduler` containers running.  It may take some time to download the docker images - you can check if the containers are running using `docker ps`.

### Install Calico on Master

We need to install Calico on the master so that the master can route to the pods in our Kubernetes cluster.

First, start the etcd instance used by Calico.  We'll run this as a static Kubernetes pod.  Before we install it, we'll need to edit the manifest.  Open `calico-kubernetes-master/config/master/calico-etcd.manifest` and replace all instances of `<PRIVATE_IPV4>` with your master's IP address.  Then, copy the file to the `/etc/kubernetes/manifests` directory.

```
sudo cp -f calico-kubernetes-master/config/master/calico-etcd.manifest /etc/kubernetes/manifests
```

> Note: For simplicity, in this demonstration we are using a single instance of etcd.  In a production deployment a distributed etcd cluster is recommended for redundancy.

Now, install Calico.  We'll need the `calicoctl` tool to do this.

```
# Install the `calicoctl` binary
wget https://github.com/projectcalico/calico-docker/releases/download/v0.9.0/calicoctl
chmod +x calicoctl
sudo mv calicoctl /usr/bin

# Fetch the calico/node container
sudo docker pull calico/node:v0.9.0

# Install, enable, and start the Calico service
sudo cp -f calico-kubernetes-master/config/master/calico-node.service /etc/systemd
sudo systemctl enable /etc/systemd/calico-node.service
sudo systemctl start calico-node.service
```

## Setup Nodes

The following steps should be run on each Kubernetes node.

### Configure environment variables for `kubelet` process

1.) Download the `calico-kubernetes` repository, which contains the necessary configuration for this guide.

```
wget https://github.com/projectcalico/calico-kubernetes/archive/master.tar.gz
tar -xvf master.tar.gz
```

2.) Copy the network-environment-template from the `node` directory

```
cp calico-kubernetes-master/config/node/network-environment-template network-environment
```

3.) Edit `network-environment` to represent this node's settings.

4.) Move `network-environment` into `/etc`

```
sudo mv -f network-environment /etc
```

### Install Calico on the Node

We'll install Calico using the provided `calico-node.service` systemd unit file.

```
# Install the `calicoctl` binary
wget https://github.com/projectcalico/calico-docker/releases/download/v0.9.0/calicoctl
chmod +x calicoctl
sudo mv calicoctl /usr/bin

# Fetch the calico/node container
sudo docker pull calico/node:v0.9.0

# Install, enable, and start the Calico service
sudo cp -f calico-kubernetes-master/config/node/calico-node.service /etc/systemd
sudo systemctl enable /etc/systemd/calico-node.service
sudo systemctl start calico-node.service
```

### Install Kubernetes on the Node

1.) Download & Install the Kubernetes binaries.

```
# Get the Kubernetes Release.
wget https://github.com/kubernetes/kubernetes/releases/download/v1.1.0/kubernetes.tar.gz

# Extract the Kubernetes binaries.
tar -xf kubernetes.tar.gz
tar -xf kubernetes/server/kubernetes-server-linux-amd64.tar.gz

# Install the `kubelet` and `kube-proxy` binaries.
sudo cp -f kubernetes/server/bin/kubelet /usr/bin
sudo cp -f kubernetes/server/bin/kube-proxy /usr/bin
```

2.) Install the `kubelet` and `kube-proxy` systemd unit files.

```
# Install the unit files
sudo cp -f calico-kubernetes-master/config/node/kubelet.service /etc/systemd
sudo cp -f calico-kubernetes-master/config/node/kube-proxy.service /etc/systemd

# Enable the unit files so that they run on boot
sudo systemctl enable /etc/systemd/kubelet.service
sudo systemctl enable /etc/systemd/kube-proxy.service

# Start the services
sudo systemctl start kubelet.service
sudo systemctl start kube-proxy.service
```

## Install the DNS Addon

Most Kubernetes deployments will require the DNS addon for service discovery.  For more on DNS service discovery, check [here](../../cluster/addons/dns/).

The config repository for this guide comes with manifest files to start the DNS addon.  To install DNS, do the following on your Master node.

Replace `<MASTER_IP>` in `calico-kubernetes-master/config/master/dns/skydns-rc.yaml` with your Master's IP address.  Then, create `skydns-rc.yaml` and `skydns-svc.yaml` using `kubectl create -f <FILE>`.

## Launch other Services With Calico-Kubernetes

At this point, you have a fully functioning cluster running on Kubernetes with a master and two nodes networked with Calico. You can now follow any of the [standard documentation](../../examples/) to set up other services on your cluster.

## Connectivity to outside the cluster

Because containers in this guide have private `192.168.0.0/16` IPs, you will need NAT to allow connectivity between containers and the internet. However, in a production data center deployment, NAT is not always necessary, since Calico can peer with the data center's border routers over BGP.

### NAT on the nodes

The simplest method for enabling connectivity from containers to the internet is to use an `iptables` masquerade rule. This is the standard mechanism recommended in the [Kubernetes GCE environment](../../docs/admin/networking.md#google-compute-engine-gce).

We need to NAT traffic that has a destination outside of the cluster. Cluster-internal traffic includes the Kubernetes master/nodes, and the traffic within the container IP subnet. A suitable masquerade chain would follow this pattern below, replacing the following variables:
- `CONTAINER_SUBNET`: The cluster-wide subnet from which container IPs are chosen. Run `ETCD_AUTHORITY=127.0.0.1:6666 calicoctl pool show` on the Kubernetes master to find your configured container subnet.
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

In a data center environment, it is recommended to configure Calico to peer with the border routers over BGP. This means that the container IPs will be routable anywhere in the data center, and so NAT is not needed on the nodes (though it may be enabled at the data center edge to allow outbound-only internet connectivity).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/ubuntu-calico.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
