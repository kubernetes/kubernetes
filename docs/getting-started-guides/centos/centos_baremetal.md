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
[here](http://releases.k8s.io/release-1.1/docs/getting-started-guides/centos/centos_baremetal.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Kubernetes Automated Deployment On CentOS
------------------------------------------------

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Starting a Cluster](#starting-a-cluster)
    - [Download binaries](#download-binaries)
    - [Configure and start the kubernetes cluster](#configure-and-start-the-kubernetes-cluster)
    - [Install the Kubernetes command line tools](#install-the-kubernetes-command-line-tools)
    - [Getting started with your cluster](#getting-started-with-your-cluster)
- [Tearing down the cluster](#tearing-down-the-cluster)
- [Trouble shooting](#trouble-shooting)

## Introduction

This document describes how to deploy Kubernetes on CentOS bare metal nodes. The example below will set up a Kubernetes cluster with 2 worker nodes and 1 master. You can scale to any number of nodes with ease.

[Huawei PaaS Open Source Software team](https://github.com/Huawei-PaaS) will maintain this work.

## Prerequisites

1. This guide is tested OK on CentOS 7.
2. Make sure all the hosts can communicate with each other.
3. Make sure all the hosts are able to connect to the Internet to download the necessary files.
4. Make sure you can ssh into all the hosts as well as execute sudo commands without interactive prompts.

## Starting a Cluster

### Download binaries

Git clone the kubernetes github repo to your host.

``` console
[centos@k8s-dev ~]# git clone https://github.com/kubernetes/kubernetes.git
```

Download all the needed releases and extract binaries (into cluster/centos/binaries).

``` console
[centos@k8s-dev ~]# cd kubernetes/cluster/centos
[centos@k8s-dev centos]# ./build.sh all
```

If you want to specify the versions of etcd, flanenl or k8s, change corresponding variables `ETCD_VERSION` , `FLANNEL_VERSION` and `K8S_VERSION` in `config-build.sh`. By default, the scripts use etcd 2.0.12, flannel 0.5.3 and k8s 1.0.4.

### Configure and start the kubernetes cluster

System Information:

| IP Address  |   Role   |
|-------------|----------|
|10.211.55.11 |  Master  |
|10.211.55.12 |   Node   |
|10.211.55.13 |   Node   |

Change the cluster settings in `cluster/centos/config-default.sh` as you need. Here is an example:

```bash
export MASTER="centos@10.211.55.11"
export NODES="centos@10.211.55.12 centos@10.211.55.13"
export NUM_NODES=2
export SERVICE_CLUSTER_IP_RANGE=192.168.3.0/24
export FLANNEL_NET=172.16.0.0/16
```

- `MASTER`: defines the master of your cluster.
- `NODES`: defines the worker nodes, separated with blank space like `<user_1@ip_1> <user_2@ip_2>`.
- `NUM_NODES`: defines the total number of worker nodes.
- `SERVICE_CLUSTER_IP_RANGE`: defines the kubernetes service IP range. Make sure you do define a valid private ip range here, since some IaaS provider may reserve private ips.
- `FLANNEL_NET`: defines the IP range used for flannel overlay network, should not conflict with `SERVICE_CLUSTER_IP_RANGE`.

	**Note**: You can use three private network ranges (shown below) according to rfc1918. Besides you'd better not choose the one that conflicts with your own/existing private network range.
		10.0.0.0        -   10.255.255.255  (10/8 prefix)
		172.16.0.0      -   172.31.255.255  (172.16/12 prefix)
		192.168.0.0     -   192.168.255.255 (192.168/16 prefix)

Run the `<kubernetes>/cluster/kube-up.sh` script to start the cluster:

```console
[centos@k8s-dev centos]# cd ../cluster
[centos@k8s-dev cluster]# KUBERNETES_PROVIDER=centos ./kube-up.sh
```

The scripts will automatically scp necessary files to all the hosts and start k8s services on them.

When the cluster starts up correctly, you will see message shown as below:

```console
Cluster validation succeeded
Done, listing cluster services:

Kubernetes master is running at http://10.211.55.11:8080
```

### Install the Kubernetes command line tools

The [kubectl](../../user-guide/kubectl/kubectl.md) tool controls the Kubernetes cluster manager. It lets you inspect your cluster resources, create, delete, and update components, and much more.You will use it to look at your new cluster and bring up example apps.

The `kubectl` binary locates in the `cluster/centos/binaries` directory.
Add the appropriate binary folder to your `PATH` to access kubectl:

```bash
export PATH=<path/to/kubernetes-directory>/cluster/centos/binaries:$PATH
```

#### Enable bash completion of the Kubernetes command line tools

You may find it useful to enable `kubectl` bash completion:

```
source <path/to/kubernetes-directory>/contrib/completions/bash/kubectl
```

**Note**: This will last for the duration of your bash session. If you want to make this permanent you need to add this line in your bash profile.

Alternatively, on most linux distributions you can also move the completions file to your bash_completions.d like this:

```
cp <path/to/kubernetes-directory>/contrib/completions/bash/kubectl /etc/bash_completion.d/
```

but then you have to update it when you update kubectl.

### Getting started with your cluster

Once `kubectl` is in your path, you can use it to look at your cluster. E.g., running `kubectl get nodes`:

```console
[centos@k8s-dev cluster]# kubectl get nodes
NAME            LABELS                                 STATUS
10.211.55.12    kubernetes.io/hostname=10.211.55.12    Ready
10.211.55.13    kubernetes.io/hostname=10.211.55.13    Ready
```

To try out your new cluster, see [a simple nginx example](../../user-guide/simple-nginx.md), or look in the [examples directory](../../../examples/) for more complete applications. The [guestbook example](../../../examples/guestbook/) is a good "getting started" walkthrough.

## Tearing down the cluster

To remove/delete/teardown the cluster, use the `kube-down.sh` script.

```console
[centos@k8s-dev centos]# cd kubernetes/cluster
[centos@k8s-dev cluster]# KUBERNETES_PROVIDER=centos ./kube-down.sh
```

## Trouble shooting

### Cluster initialization hang

Make sure all the involved binaries are located properly in the `binaries/master` or `binaries/node` directory before you run the `<kubernetes>/cluster/kube-up.sh` script to start the cluster.

If the Kubernetes startup script hangs waiting for the API to be reachable, you can troubleshoot by SSHing into the master and node hosts.

Check the service status and logs with systemd and journal commands such as `sudo systemctl status kube-apiserver.service` and `sudo journalctl -u kube-apiserver.service`.

- Services should be running on the Master:
	- `etcd.service`
    - `kube-apiserver.service`
    - `kube-controller-manager.service`
    - `kube-scheduler.service`
- Services should be running on each Node:
	- `docker.service`
	- `flannel.service`
	- `kube-proxy.service`
	- `kube-kubelet.service`

**Once you fix the issue, you should run `kube-down.sh` to cleanup** after the partial cluster creation, before running `kube-up.sh` to try again.

### Proxy setting

Currently, the Kubernetes startup script doesn't support proxy setting. If your hosts need proxy setting to connect to the internet, you may get stuck in provisionning master or downloading docker images.

One hacky way to get it work is to insert the proxy settings to `cluster/saltbase/salt/generate-cert/make-ca-cert.sh` and `<systemd-services>/docker.service`. We are working to solve it in a more generic way.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/centos/centos_baremetal.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
