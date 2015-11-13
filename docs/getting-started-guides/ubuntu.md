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
[here](http://releases.k8s.io/release-1.1/docs/getting-started-guides/ubuntu.md).

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
    - [Set up working directory](#set-up-working-directory)
    - [Configure and start the kubernetes cluster](#configure-and-start-the-kubernetes-cluster)
    - [Test it out](#test-it-out)
    - [Deploy addons](#deploy-addons)
    - [Trouble shooting](#trouble-shooting)
- [Upgrading a Cluster](#upgrading-a-cluster)
    - [Test it out](#test-it-out-ii)

## Introduction

This document describes how to deploy kubernetes on ubuntu nodes, 1 master and 3 nodes involved
in the given examples. You can scale to **any number of nodes** by changing some settings with ease.
The original idea was heavily inspired by @jainvipin 's ubuntu single node
work, which has been merge into this document.

[Cloud team from Zhejiang University](https://github.com/ZJU-SEL) will maintain this work.

## Prerequisites

1. The nodes have installed docker version 1.2+ and bridge-utils to manipulate linux bridge.
2. All machines can communicate with each other. Master node needs to be connected to the
Internet to download the necessary files, while worker nodes do not.
3. These guide is tested OK on Ubuntu 14.04 LTS 64bit server, but it can not work with
Ubuntu 15 which uses systemd instead of upstart.
4. Dependencies of this guide: etcd-2.2.1, flannel-0.5.3, k8s-1.0.6, may work with higher versions.
5. All the remote servers can be ssh logged in without a password by using key authentication.


## Starting a Cluster

### Set up working directory

Clone the kubernetes github repo locally

``` console
$ git clone https://github.com/kubernetes/kubernetes.git
```

#### Configure and start the Kubernetes cluster

The startup process will first download all the required binaries automatically.
By default etcd version is 2.2.1, flannel version is 0.5.3 and k8s version is 1.0.6.
You can customize your etcd version, flannel version, k8s version by changing corresponding variables
`ETCD_VERSION` , `FLANNEL_VERSION` and `KUBE_VERSION` like following.

```console
$ export KUBE_VERSION=1.0.5
$ export FLANNEL_VERSION=0.5.0
$ export ETCD_VERSION=2.2.0
```

Note that we use flannel here to set up overlay network, yet it's optional. Actually you can build up k8s
cluster natively, or use flannel, Open vSwitch or any other SDN tool you like.

An example cluster is listed below:

| IP Address  |   Role   |
|-------------|----------|
|10.10.103.223|   node   |
|10.10.103.162|   node   |
|10.10.103.250| both master and node|

First configure the cluster information in cluster/ubuntu/config-default.sh, following is a simple sample.

```sh
export nodes="vcap@10.10.103.250 vcap@10.10.103.162 vcap@10.10.103.223"

export role="ai i i"

export NUM_MINIONS=${NUM_MINIONS:-3}

export SERVICE_CLUSTER_IP_RANGE=192.168.3.0/24

export FLANNEL_NET=172.16.0.0/16
```

The first variable `nodes` defines all your cluster nodes, master node comes first and
separated with blank space like `<user_1@ip_1> <user_2@ip_2> <user_3@ip_3> `

Then the `role` variable defines the role of above machine in the same order, "ai" stands for machine
acts as both master and node, "a" stands for master, "i" stands for node.

The `NUM_MINIONS` variable defines the total number of nodes.

The `SERVICE_CLUSTER_IP_RANGE` variable defines the kubernetes service IP range. Please make sure
that you do have a valid private ip range defined here, because some IaaS provider may reserve private ips.
You can use below three private network range according to rfc1918. Besides you'd better not choose the one
that conflicts with your own private network range.

     10.0.0.0        -   10.255.255.255  (10/8 prefix)

     172.16.0.0      -   172.31.255.255  (172.16/12 prefix)

     192.168.0.0     -   192.168.255.255 (192.168/16 prefix)

The `FLANNEL_NET` variable defines the IP range used for flannel overlay network,
should not conflict with above `SERVICE_CLUSTER_IP_RANGE`.

**Note:** When deploying, master needs to be connected to the Internet to download the necessary files.
If your machines are located in a private network that need proxy setting to connect the Internet,
you can set the config `PROXY_SETTING` in cluster/ubuntu/config-default.sh such as:

     PROXY_SETTING="http_proxy=http://server:port https_proxy=https://server:port"

After all the above variables being set correctly, we can use following command in `cluster/` directory to
bring up the whole cluster.

`$ KUBERNETES_PROVIDER=ubuntu ./kube-up.sh`

The scripts automatically `scp` binaries and config files to all the machines and start kubernetes
service on them. The only thing you need to do is to type the sudo password when promoted.

```console
Deploying node on machine 10.10.103.223
...
[sudo] password to start node: 
```

If everything works correctly, you will see the following message from console indicating the k8s cluster is up.

```console
Cluster validation succeeded
```

### Test it out

You can use `kubectl` command to check if the newly created cluster is working correctly.
The `kubectl` binary is under the `cluster/ubuntu/binaries` directory.
You can make it available via PATH, then you can use the below command smoothly.

For example, use `$ kubectl get nodes` to see if all of your nodes are ready.

```console
$ kubectl get nodes
NAME            LABELS                                 STATUS
10.10.103.162   kubernetes.io/hostname=10.10.103.162   Ready
10.10.103.223   kubernetes.io/hostname=10.10.103.223   Ready
10.10.103.250   kubernetes.io/hostname=10.10.103.250   Ready
```

Also you can run Kubernetes [guest-example](../../examples/guestbook/) to build a redis backend cluster．


### Deploy addons

Assuming you have a starting cluster now, this section will tell you how to deploy addons like DNS
and UI onto the existing cluster.

The configuration of DNS is configured in cluster/ubuntu/config-default.sh.

```sh
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"

DNS_SERVER_IP="192.168.3.10"

DNS_DOMAIN="cluster.local"

DNS_REPLICAS=1
```

The `DNS_SERVER_IP` is defining the ip of dns server which must be in the `SERVICE_CLUSTER_IP_RANGE`.
The `DNS_REPLICAS` describes how many dns pod running in the cluster.

By default, we also take care of kube-ui addon.

```sh
ENABLE_CLUSTER_UI="${KUBE_ENABLE_CLUSTER_UI:-true}"
```

After all the above variables have been set, just type the following command.

```console
$ cd cluster/ubuntu
$ KUBERNETES_PROVIDER=ubuntu ./deployAddons.sh
```

After some time, you can use `$ kubectl get pods --namespace=kube-system` to see the DNS and UI pods are running in the cluster.

### On going

We are working on these features which we'd like to let everybody know:

1. Run kubernetes binaries in Docker using [kube-in-docker](https://github.com/ZJU-SEL/kube-in-docker/tree/baremetal-kube),
to eliminate OS-distro differences.
2. Tearing Down scripts: clear and re-create the whole stack by one click.

### Trouble shooting

Generally, what this approach does is quite simple:

1. Download and copy binaries and configuration files to proper directories on every node.
2. Configure `etcd` for master node using IPs based on input from user.
3. Create and start flannel network for worker nodes.

So if you encounter a problem, check etcd configuration of master node first.

1. Check `/var/log/upstart/etcd.log` for suspicious etcd log
2. You may find following commands useful, the former one to bring down the cluster, while the latter one could start it again.

```console
$ KUBERNETES_PROVIDER=ubuntu ./kube-down.sh
$ KUBERNETES_PROVIDER=ubuntu ./kube-up.sh
```

3. You can also customize your own settings in `/etc/default/{component_name}` and restart it via
`$ sudo service {component_name} restart`.


## Upgrading a Cluster

If you already have a kubernetes cluster, and want to upgrade to a new version,
you can use following command in `cluster/` directory to update the whole cluster
or a specified node to a new version.

```console
$ KUBERNETES_PROVIDER=ubuntu ./kube-push.sh [-m|-n <node id>] <version>
```

It can be done for all components (by default), master(`-m`) or specified node(`-n`).
Upgrading a single node is currently experimental.
If the version is not specified, the script will try to use local binaries. You should ensure all
the binaries are well prepared in the expected directory path cluster/ubuntu/binaries.

```console
$ tree cluster/ubuntu/binaries
binaries/
├── kubectl
├── master
│   ├── etcd
│   ├── etcdctl
│   ├── flanneld
│   ├── kube-apiserver
│   ├── kube-controller-manager
│   └── kube-scheduler
└── minion
    ├── flanneld
    ├── kubelet
    └── kube-proxy
```

You can use following command to get a help.

```console
$ KUBERNETES_PROVIDER=ubuntu ./kube-push.sh -h
```

Here are some examples:

* upgrade master to version 1.0.5: `$ KUBERNETES_PROVIDER=ubuntu ./kube-push.sh -m 1.0.5`
* upgrade node 10.10.103.223 to version 1.0.5 : `$ KUBERNETES_PROVIDER=ubuntu ./kube-push.sh -n 10.10.103.223 1.0.5`
* upgrade master and all nodes to version 1.0.5: `$ KUBERNETES_PROVIDER=ubuntu ./kube-push.sh 1.0.5`

The script will not delete any resources of your cluster, it just replaces the binaries.

### Test it out

You can use the `kubectl` command to check if the newly upgraded kubernetes cluster is working correctly. See
also [test-it-out](ubuntu.md#test-it-out)

To make sure the version of the upgraded cluster is what you expect, you will find these commands helpful.
* upgrade all components or master: `$ kubectl version`. Check the *Server Version*.
* upgrade node 10.10.102.223: `$ ssh -t vcap@10.10.102.223 'cd /opt/bin && sudo ./kubelet --version'`


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/ubuntu.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
