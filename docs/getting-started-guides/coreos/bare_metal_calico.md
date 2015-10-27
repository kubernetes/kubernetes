<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

Bare Metal CoreOS with Kubernetes and Project Calico
------------------------------------------
This guide explains how to deploy a bare-metal Kubernetes cluster on CoreOS using [Calico networking](http://www.projectcalico.org).

Specifically, this guide will have you do the following:
- Deploy a Kubernetes master node on CoreOS using cloud-config
- Deploy two Kubernetes compute nodes with Calico Networking using cloud-config

## Prerequisites

1. At least three bare-metal machines (or VMs) to work with. This guide will configure them as follows
  - 1 Kubernetes Master
  - 2 Kubernetes Nodes
2. Your nodes should have IP connectivity.

## Cloud-config

This guide will use [cloud-config](https://coreos.com/docs/cluster-management/setup/cloudinit-cloud-config/) to configure each of the nodes in our Kubernetes cluster.

For ease of distribution, the cloud-config files required for this demonstration can be found on [GitHub](https://github.com/projectcalico/calico-kubernetes-coreos-demo).

This repo includes two cloud config files:
- `master-config.yaml`: Cloud-config for the Kubernetes master
- `node-config.yaml`: Cloud-config for each Kubernetes compute host

In the next few steps you will be asked to configure these files and host them on an HTTP server where your cluster can access them.

## Building Kubernetes

To get the Kubernetes source, clone the GitHub repo, and build the binaries.

```
git clone https://github.com/kubernetes/kubernetes.git
cd kubernetes
./build/release.sh 
```

Once the binaries are built, host the entire `<kubernetes>/_output/dockerized/bin/<OS>/<ARCHITECTURE>/` folder on an accessible HTTP server so they can be accessed by the cloud-config.  You'll point your cloud-config files at this HTTP server later.

## Download CoreOS

Let's download the CoreOS bootable ISO.  We'll use this image to boot and install CoreOS on each server.

```
wget http://stable.release.core-os.net/amd64-usr/current/coreos_production_iso_image.iso
```

You can also download the ISO from the [CoreOS website](https://coreos.com/docs/running-coreos/platforms/iso/).

## Configure the Kubernetes Master

Once you've downloaded the image, use it to boot your Kubernetes Master server.  Once booted, you should be automatically logged in as the `core` user.

Let's get the master-config.yaml and fill in the necessary variables.  Run the following commands on your HTTP server to get the cloud-config files.

```
git clone https://github.com/Metaswitch/calico-kubernetes-demo.git
cd calico-kubernetes-demo/coreos
```

You'll need to replace the following variables in the `master-config.yaml` file to match your deployment.
- `<SSH_PUBLIC_KEY>`: The public key you will use for SSH access to this server.
- `<KUBERNETES_LOC>`: The address used to get the kubernetes binaries over HTTP.

> **Note:** The config will prepend `"http://"` and append `"/(kubernetes | kubectl | ...)"` to your `KUBERNETES_LOC` variable:, format accordingly

Host the modified `master-config.yaml` file and pull it on to your Kubernetes Master server.

The CoreOS bootable ISO comes with a tool called `coreos-install` which will allow us to install CoreOS to disk and configure the install using cloud-config.  The following command will download and install stable CoreOS, using the master-config.yaml file for configuration.

```
sudo coreos-install -d /dev/sda -C stable -c master-config.yaml
```

Once complete, eject the bootable ISO and restart the server.  When it comes back up, you should have SSH access as the `core` user using the public key provided in the master-config.yaml file.

## Configure the compute hosts

>The following steps will set up a Kubernetes node for use as a compute host.  This demo uses two compute hosts, so you should run the following steps on each.

First, boot up your node using the bootable ISO we downloaded earlier.  You should be automatically logged in as the `core` user.

Let's modify the `node-config.yaml` cloud-config file on your HTTP server.  Make a copy for this node, and fill in the necessary variables.

You'll need to replace the following variables in the `node-config.yaml` file to match your deployment.
- `<HOSTNAME>`: Hostname for this node (e.g. kube-node1, kube-node2)
- `<SSH_PUBLIC_KEY>`: The public key you will use for SSH access to this server.
- `<KUBERNETES_MASTER>`: The IPv4 address of the Kubernetes master.
- `<KUBERNETES_LOC>`: The address to use in order to get the kubernetes binaries over HTTP.
- `<DOCKER_BRIDGE_IP>`: The IP and subnet to use for pods on this node.  By default, this should fall within the 192.168.0.0/16 subnet.

> Note: The DOCKER_BRIDGE_IP is the range used by this Kubernetes node to assign IP addresses to pods on this node.  This subnet must not overlap with the subnets assigned to the other Kubernetes nodes in your cluster.  Calico expects each DOCKER_BRIDGE_IP subnet to fall within 192.168.0.0/16 by default (e.g. 192.168.1.1/24 for node 1), but if you'd like to use pod IPs within a different subnet, simply run `calicoctl pool add <your_subnet>` and select DOCKER_BRIDGE_IP accordingly.

Host the modified `node-config.yaml` file and pull it on to your Kubernetes node.

```
wget http://<http_server_ip>/node-config.yaml
```

Install and configure CoreOS on the node using the following command.

```
sudo coreos-install -d /dev/sda -C stable -c node-config.yaml
```

Once complete, restart the server.  When it comes back up, you should have SSH access as the `core` user using the public key provided in the `node-config.yaml` file.  It will take some time for the node to be fully configured.  Once fully configured, you can check that the node is running with the following command on the Kubernetes master.

```
/home/core/kubectl get nodes
```

## Testing the Cluster

You should now have a functional bare-metal Kubernetes cluster with one master and two compute hosts.
Try running the [guestbook demo](../../../examples/guestbook/) to test out your new cluster!





<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/coreos/bare_metal_calico.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
