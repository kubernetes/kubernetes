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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/getting-started-guides/coreos/bare_metal_calico.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

Bare Metal Kubernetes on CoreOS with Calico Networking
------------------------------------------
This document describes how to deploy Kubernetes with Calico networking on _bare metal_ CoreOS. For more information on Project Calico, visit [projectcalico.org](http://projectcalico.org) and the [calico-docker repository](https://github.com/projectcalico/calico-docker).

To install Calico on an existing Kubernetes cluster, or for more information on deploying Calico with Kubernetes in a number of other environments take a look at our supported [deployment guides](https://github.com/projectcalico/calico-docker/tree/master/docs/kubernetes).

Specifically, this guide will have you do the following:
- Deploy a Kubernetes master node on CoreOS using cloud-config
- Deploy two Kubernetes compute nodes with Calico Networking using cloud-config

## Prerequisites

1. At least three bare-metal machines (or VMs) to work with. This guide will configure them as follows:
  - 1 Kubernetes Master
  - 2 Kubernetes Nodes
2. Your nodes should have IP connectivity.

## Cloud-config

This guide will use [cloud-config](https://coreos.com/docs/cluster-management/setup/cloudinit-cloud-config/) to configure each of the nodes in our Kubernetes cluster.

We'll use two cloud-config files:
- `master-config.yaml`: Cloud-config for the Kubernetes master
- `node-config.yaml`: Cloud-config for each Kubernetes node

## Download CoreOS

Let's download the CoreOS bootable ISO.  We'll use this image to boot and install CoreOS on each server.

```
wget http://stable.release.core-os.net/amd64-usr/current/coreos_production_iso_image.iso
```

> You can also download the ISO from the [CoreOS website](https://coreos.com/docs/running-coreos/platforms/iso/).

## Configure the Kubernetes Master

Once you've downloaded the image, use it to boot your Kubernetes master.  Once booted, you should be automatically logged in as the `core` user.

*On another machine*, download the `calico-kubernetes` repository, which contains the necessary cloud-config files for this guide, and make a copy of the file `master-config-template.yaml`.

```
wget https://github.com/projectcalico/calico-kubernetes/archive/master.tar.gz
tar -xvf master.tar.gz
cp calico-kubernetes-master/config/cloud-config/master-config-template.yaml master-config.yaml
```

You'll need to replace the following variables in the `master-config.yaml` file.
- `<SSH_PUBLIC_KEY>`: The public key you will use for SSH access to this server.

Move the edited `master-config.yaml` to your Kubernetes master machine.  The CoreOS bootable ISO comes with a tool called `coreos-install` which will allow us to install CoreOS and configure the machine using a cloud-config file.  The following command will download and install stable CoreOS using the `master-config.yaml` file we just created for configuration.  Run this on the Kubernetes master.

```
sudo coreos-install -d /dev/sda -C stable -c master-config.yaml
```

Once complete, eject the bootable ISO and restart the server.  When it comes back up, you should have SSH access as the `core` user using the public key provided in the `master-config.yaml` file.  It may take a few minutes for the machine to be fully configured with Kubernetes and Calico.

## Configure the compute hosts

>The following steps will set up a single Kubernetes node for use as a compute host.  Run these steps to deploy each Kubernetes node in your cluster.

First, boot up the node machine using the bootable ISO we downloaded earlier.  You should be automatically logged in as the `core` user.

Make a copy of the `node-config-template.yaml` in the `calico-kubernetes` repository for this machine.

```
cp calico-kubernetes-master/config/cloud-config/node-config-template.yaml node-config.yaml
```

You'll need to replace the following variables in the `node-config.yaml` file to match your deployment.
- `<HOSTNAME>`: Hostname for this node (e.g. kube-node1, kube-node2)
- `<SSH_PUBLIC_KEY>`: The public key you will use for SSH access to this server.
- `<KUBERNETES_MASTER>`: The IPv4 address of the Kubernetes master.

Move the modified `node-config.yaml` to your Kubernetes node machine and install and configure CoreOS on the node using the following command.

```
sudo coreos-install -d /dev/sda -C stable -c node-config.yaml
```

Once complete, eject the bootable disc and restart the server.  When it comes back up, you should have SSH access as the `core` user using the public key provided in the `node-config.yaml` file.  It will take some time for the node to be fully configured.  Once fully configured, you can check that the node is running with the following command on the Kubernetes master.

```
kubectl get nodes
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/coreos/bare_metal_calico.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
