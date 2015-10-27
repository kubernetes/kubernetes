<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
Kubernetes Deployment On Bare-metal Ubuntu Nodes
------------------------------------------------

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Starting a Cluster](#starting-a-cluster)
    - [Download binaries](#download-binaries)
    - [Configure and start the kubernetes cluster](#configure-and-start-the-kubernetes-cluster)
    - [Test it out](#test-it-out)
    - [Deploy addons](#deploy-addons)
    - [Trouble shooting](#trouble-shooting)
- [Upgrading a Cluster](#upgrading-a-cluster)

## Introduction

This document describes how to deploy kubernetes on ubuntu nodes, 1 master and 3 nodes involved
in the given examples. You can scale to **any number of nodes** by changing some settings with ease.
The original idea was heavily inspired by @jainvipin 's ubuntu single node
work, which has been merge into this document.

[Cloud team from Zhejiang University](https://github.com/ZJU-SEL) will maintain this work.

## Prerequisites

1. The nodes have installed docker version 1.2+ and bridge-utils to manipulate linux bridge.
2. All machines can communicate with each other. Master node needs to connect the Internet to download the necessary files, while working nodes do not.
3. These guide is tested OK on Ubuntu 14.04 LTS 64bit server, but it can not work with
Ubuntu 15 which use systemd instead of upstart. We are working around fixing this.
4. Dependencies of this guide: etcd-2.0.12, flannel-0.4.0, k8s-1.0.3, may work with higher versions.
5. All the remote servers can be ssh logged in without a password by using key authentication.


## Starting a Cluster

### Download binaries

First clone the kubernetes github repo

``` console
$ git clone https://github.com/kubernetes/kubernetes.git
```

Then download all the needed binaries into given directory (cluster/ubuntu/binaries)

``` console
$ cd kubernetes/cluster/ubuntu
$ ./build.sh
```

You can customize your etcd version, flannel version, k8s version by changing corresponding variables
`ETCD_VERSION` , `FLANNEL_VERSION` and `KUBE_VERSION` in build.sh, by default etcd version is 2.0.12,
flannel version is 0.4.0 and k8s version is 1.0.3.

Make sure that the involved binaries are located properly in the binaries/master
or binaries/minion directory before you go ahead to the next step .

Note that we use flannel here to set up overlay network, yet it's optional. Actually you can build up k8s
cluster natively, or use flannel, Open vSwitch or any other SDN tool you like.

#### Configure and start the Kubernetes cluster

An example cluster is listed below:

| IP Address  |   Role   |
|-------------|----------|
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

The first variable `nodes` defines all your cluster nodes, MASTER node comes first and
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

**Note:** When deploying, master needs to connect the Internet to download the necessary files. If your machines locate in a private network that need proxy setting to connect the Internet, you can set the config `PROXY_SETTING` in cluster/ubuntu/config-default.sh such as:

     PROXY_SETTING="http_proxy=http://server:port https_proxy=https://server:port"

After all the above variables being set correctly, we can use following command in cluster/ directory to bring up the whole cluster.

`$ KUBERNETES_PROVIDER=ubuntu ./kube-up.sh`

The scripts automatically scp binaries and config files to all the machines and start the k8s service on them.
The only thing you need to do is to type the sudo password when promoted.

```console
Deploying minion on machine 10.10.103.223
...
[sudo] password to copy files and start minion: 
```

If all things goes right, you will see the below message from console indicating the k8s is up.

```console
Cluster validation succeeded
```

### Test it out

You can use `kubectl` command to check if the newly created k8s is working correctly.
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

Also you can run Kubernetes [guest-example](../../examples/guestbook/) to build a redis backend cluster on the k8s．


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

1. Download and copy binaries and configuration files to proper directories on every node
2. Configure `etcd` using IPs based on input from user
3. Create and start flannel network

So if you encounter a problem, **check etcd configuration first**

Please try:

1. Check `/var/log/upstart/etcd.log` for suspicious etcd log
2. Check `/etc/default/etcd`, as we do not have much input validation, a right config should be like:

	```sh
	ETCD_OPTS="-name infra1 -initial-advertise-peer-urls <http://ip_of_this_node:2380> -listen-peer-urls <http://ip_of_this_node:2380> -initial-cluster-token etcd-cluster-1 -initial-cluster infra1=<http://ip_of_this_node:2380>,infra2=<http://ip_of_another_node:2380>,infra3=<http://ip_of_another_node:2380> -initial-cluster-state new"
	```

3. You may find following commands useful, the former one to bring down the cluster, while
the latter one could start it again.

    ```console
    $ KUBERNETES_PROVIDER=ubuntu ./kube-down.sh
    $ KUBERNETES_PROVIDER=ubuntu ./kube-up.sh
    ```

4. You can also customize your own settings in `/etc/default/{component_name}`.


### Upgrading a Cluster

If you already have a kubernetes cluster, and want to upgrade to a new version,
you can use following command in cluster/ directory to update the whole cluster or a specified node to a new version.

```console
$ KUBERNETES_PROVIDER=ubuntu ./kube-push.sh [-m|-n <node id>] <version>
```

It can be done for all components (by default), master(`-m`) or specified node(`-n`).
If the version is not specified, the script will try to use local binaries.You should ensure all the binaries are well prepared in path `cluster/ubuntu/binaries`.

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

Upgrading single node is experimental now. You can use following command to get a help.

```console
$ KUBERNETES_PROVIDER=ubuntu ./kube-push.sh -h
```

Some examples are as follows:

* upgrade master to version 1.0.5: `$ KUBERNETES_PROVIDER=ubuntu ./kube-push.sh -m 1.0.5`
* upgrade node 10.10.103.223 to version 1.0.5 : `$ KUBERNETES_PROVIDER=ubuntu ./kube-push.sh -n 10.10.103.223 1.0.5`
* upgrade master and all nodes to version 1.0.5: `$ KUBERNETES_PROVIDER=ubuntu ./kube-push.sh 1.0.5`

The script will not delete any resources of your cluster, it just replaces the binaries.
You can use `kubectl` command to check if the newly upgraded k8s is working correctly.
For example, use `$ kubectl get nodes` to see if all of your nodes are ready.Or refer to [test-it-out](ubuntu.md#test-it-out)



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/ubuntu.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
