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
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/ubuntu.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Kubernetes Deployment On Bare-metal Ubuntu Nodes
------------------------------------------------

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
    - [Starting a Cluster](#starting-a-cluster)
        - [Make *kubernetes* , *etcd* and *flanneld* binaries](#make-kubernetes--etcd-and-flanneld-binaries)
        - [Configure and start the Kubernetes cluster](#configure-and-start-the-kubernetes-cluster)
        - [Deploy addons](#deploy-addons)
        - [Trouble Shooting](#trouble-shooting)

## Introduction

This document describes how to deploy Kubernetes on ubuntu nodes, including 1 Kubernetes master and 3 Kubernetes nodes, and people uses this approach can scale to **any number of nodes** by changing some settings with ease. The original idea was heavily inspired by @jainvipin 's ubuntu single node work, which has been merge into this document.

[Cloud team from Zhejiang University](https://github.com/ZJU-SEL) will maintain this work.

## Prerequisites

*1 The nodes have installed docker version 1.2+ and bridge-utils to manipulate linux bridge*

*2 All machines can communicate with each other, no need to connect Internet (should use private docker registry in this case)*

*3 These guide is tested OK on Ubuntu 14.04 LTS 64bit server, but it can not work with Ubuntu 15 which use systemd instead of upstart and we are fixing this*

*4 Dependencies of this guide: etcd-2.0.12, flannel-0.4.0, k8s-1.0.1, but it may work with higher versions*

*5 All the remote servers can be ssh logged in without a password by using key authentication*


### Starting a Cluster

#### Make *kubernetes* , *etcd* and *flanneld* binaries

First clone the kubernetes github repo, `$ git clone https://github.com/GoogleCloudPlatform/kubernetes.git`

then `$ cd kubernetes/cluster/ubuntu`.

Then run `$ ./build.sh`, this will download all the needed binaries into `./binaries`.

You can customize your etcd version, flannel version, k8s version by changing variable `ETCD_VERSION` , `FLANNEL_VERSION` and `K8S_VERSION` in build.sh, default etcd version is 2.0.12, flannel version is 0.4.0 and K8s version is 1.0.1.

Please make sure that there are `kube-apiserver`, `kube-controller-manager`, `kube-scheduler`, `kubelet`, `kube-proxy`, `etcd`, `etcdctl` and `flannel` in the binaries/master or binaries/minion directory.

> We used flannel here because we want to use overlay network, but please remember it is not the only choice, and it is also not a k8s' necessary dependence. Actually you can just build up k8s cluster natively, or use flannel, Open vSwitch or any other SDN tool you like, we just choose flannel here as an example.

#### Configure and start the Kubernetes cluster

An example cluster is listed as below:

| IP Address|Role |
|---------|------|
|10.10.103.223|   node   |
|10.10.103.162|   node   |
|10.10.103.250| both master and node|

First configure the cluster information in cluster/ubuntu/config-default.sh, below is a simple sample.

```sh
export nodes="vcap@10.10.103.250 vcap@10.10.103.162 vcap@10.10.103.223"

export role="ai i i"

export NUM_MINIONS=${NUM_MINIONS:-3}

export SERVICE_CLUSTER_IP_RANGE=192.168.3.0/24

export FLANNEL_NET=172.16.0.0/16
```

The first variable `nodes` defines all your cluster nodes, MASTER node comes first and separated with blank space like `<user_1@ip_1> <user_2@ip_2> <user_3@ip_3> `

Then the `roles ` variable defines the role of above machine in the same order, "ai" stands for machine acts as both master and node, "a" stands for master, "i" stands for node. So they are just defined the k8s cluster as the table above described.

The `NUM_MINIONS` variable defines the total number of nodes.

The `SERVICE_CLUSTER_IP_RANGE` variable defines the Kubernetes service IP range. Please make sure that you do have a valid private ip range defined here, because some IaaS provider may reserve private ips. You can use below three private network range according to rfc1918. Besides you'd better not choose the one that conflicts with your own private network range.

     10.0.0.0        -   10.255.255.255  (10/8 prefix)

     172.16.0.0      -   172.31.255.255  (172.16/12 prefix)

     192.168.0.0     -   192.168.255.255 (192.168/16 prefix)

The `FLANNEL_NET` variable defines the IP range used for flannel overlay network, should not conflict with above `SERVICE_CLUSTER_IP_RANGE`.

After all the above variables being set correctly, we can use following command in cluster/ directory to bring up the whole cluster.

`$ KUBERNETES_PROVIDER=ubuntu ./kube-up.sh`

The scripts automatically scp binaries and config files to all the machines and start the k8s service on them. The only thing you need to do is to type the sudo password when promoted. The current machine name is shown below, so you will not type in the wrong password.

```console
Deploying minion on machine 10.10.103.223

...

[sudo] password to copy files and start minion: 
```

If all things goes right, you will see the below message from console
`Cluster validation succeeded` indicating the k8s is up.

**All done !**

You can also use `kubectl` command to see if the newly created k8s is working correctly. The `kubectl` binary is under the `cluster/ubuntu/binaries` directory. You can move it into your PATH. Then you can use the below command smoothly.

For example, use `$ kubectl get nodes` to see if all your nodes are in ready status. It may take some time for the nodes ready to use like below.

```console
NAME            LABELS                                 STATUS

10.10.103.162   kubernetes.io/hostname=10.10.103.162   Ready

10.10.103.223   kubernetes.io/hostname=10.10.103.223   Ready

10.10.103.250   kubernetes.io/hostname=10.10.103.250   Ready
```

Also you can run Kubernetes [guest-example](../../examples/guestbook/) to build a redis backend cluster on the k8s．


#### Deploy addons

After the previous parts, you will have a working k8s cluster, this part will teach you how to deploy addons like dns onto the existing cluster.

The configuration of dns is configured in cluster/ubuntu/config-default.sh.

```sh
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"

DNS_SERVER_IP="192.168.3.10"

DNS_DOMAIN="cluster.local"

DNS_REPLICAS=1
```

The `DNS_SERVER_IP` is defining the ip of dns server which must be in the service_cluster_ip_range.

The `DNS_REPLICAS` describes how many dns pod running in the cluster.

After all the above variable have been set. Just type the below command

```console
$ cd cluster/ubuntu

$ KUBERNETES_PROVIDER=ubuntu ./deployAddons.sh
```

After some time, you can use `$ kubectl get pods` to see the dns pod is running in the cluster. Done!

#### On going

We are working on these features which we'd like to let everybody know:

1. Run Kubernetes binaries in Docker using [kube-in-docker](https://github.com/ZJU-SEL/kube-in-docker/tree/baremetal-kube), to eliminate OS-distro differences.

2. Tearing Down scripts: clear and re-create the whole stack by one click.

#### Trouble Shooting

Generally, what this approach did is quite simple:

1. Download and copy binaries and configuration files to proper directories on every node

2. Configure `etcd` using IPs based on input from user

3. Create and start flannel network

So, if you see a problem, **check etcd configuration first**

Please try:

1. Check `/var/log/upstart/etcd.log` for suspicious etcd log

2. Check `/etc/default/etcd`, as we do not have much input validation, a right config should be like:

	```sh
	ETCD_OPTS="-name infra1 -initial-advertise-peer-urls <http://ip_of_this_node:2380> -listen-peer-urls <http://ip_of_this_node:2380> -initial-cluster-token etcd-cluster-1 -initial-cluster infra1=<http://ip_of_this_node:2380>,infra2=<http://ip_of_another_node:2380>,infra3=<http://ip_of_another_node:2380> -initial-cluster-state new"
	```

3. You can use below command
   `$ KUBERNETES_PROVIDER=ubuntu ./kube-down.sh` to bring down the cluster and run
   `$ KUBERNETES_PROVIDER=ubuntu ./kube-up.sh` again to start again.

4. You can also customize your own settings in `/etc/default/{component_name}` after configured success.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/ubuntu.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
