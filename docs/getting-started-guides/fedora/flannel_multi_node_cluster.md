<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
Kubernetes multiple nodes cluster with flannel on Fedora
--------------------------------------------------------

**Table of Contents**

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Master Setup](#master-setup)
- [Node Setup](#node-setup)
- [**Test the cluster and flannel configuration**](#test-the-cluster-and-flannel-configuration)

## Introduction

This document describes how to deploy Kubernetes on multiple hosts to set up a multi-node cluster and networking with flannel. Follow fedora [getting started guide](fedora_manual_config.md) to setup 1 master (fed-master) and 2 or more nodes. Make sure that all nodes have different names (fed-node1, fed-node2 and so on) and labels (fed-node1-label, fed-node2-label, and so on) to avoid any conflict. Also make sure that the Kubernetes master host is running etcd, kube-controller-manager, kube-scheduler, and kube-apiserver services, and the nodes are running docker, kube-proxy and kubelet services. Now install flannel on Kubernetes nodes. flannel on each node configures an overlay network that docker uses. flannel runs on each node to setup a unique class-C container network.

## Prerequisites

1. You need 2 or more machines with Fedora installed.

## Master Setup

**Perform following commands on the Kubernetes master**

* Configure flannel by creating a `flannel-config.json` in your current directory on fed-master. flannel provides udp and vxlan among other overlay networking backend options. In this guide, we choose kernel based vxlan backend. The contents of the json are:

```json
{
    "Network": "18.16.0.0/16",
    "SubnetLen": 24,
    "Backend": {
        "Type": "vxlan",
        "VNI": 1
     }
}
```

**NOTE:** Choose an IP range that is *NOT* part of the public IP address range.

* Add the configuration to the etcd server on fed-master.

```sh
etcdctl set /coreos.com/network/config < flannel-config.json
```

* Verify the key exists in the etcd server on fed-master.

```sh
etcdctl get /coreos.com/network/config
```

## Node Setup

**Perform following commands on all Kubernetes nodes**

* Edit the flannel configuration file /etc/sysconfig/flanneld as follows:

```sh
# Flanneld configuration options

# etcd url location.  Point this to the server where etcd runs
FLANNEL_ETCD="http://fed-master:4001"

# etcd config key.  This is the configuration key that flannel queries
# For address range assignment
FLANNEL_ETCD_KEY="/coreos.com/network"

# Any additional options that you want to pass
FLANNEL_OPTIONS=""
```

**Note:** By default, flannel uses the interface for the default route. If you have multiple interfaces and would like to use an interface other than the default route one, you could add "-iface=" to FLANNEL_OPTIONS. For additional options, run `flanneld --help` on command line.

* Enable the flannel service.

```sh
systemctl enable flanneld
```

* If docker is not running, then starting flannel service is enough and skip the next step.

```sh
systemctl start flanneld
```

* If docker is already running, then stop docker, delete docker bridge (docker0), start flanneld and restart docker as follows. Another alternative is to just reboot the system (`systemctl reboot`).

```sh
systemctl stop docker
ip link delete docker0
systemctl start flanneld
systemctl start docker
```

***

## **Test the cluster and flannel configuration**

* Now check the interfaces on the nodes. Notice there is now a flannel.1 interface, and the ip addresses of docker0 and flannel.1 interfaces are in the same network. You will notice that docker0 is assigned a subnet (18.16.29.0/24 as shown below) on each Kubernetes node out of the IP range configured above. A working output should look like this:

```console
# ip -4 a|grep inet
    inet 127.0.0.1/8 scope host lo
    inet 192.168.122.77/24 brd 192.168.122.255 scope global dynamic eth0
    inet 18.16.29.0/16 scope global flannel.1
    inet 18.16.29.1/24 scope global docker0
```

* From any node in the cluster, check the cluster members by issuing a query to etcd server via curl (only partial output is shown using `grep -E "\{|\}|key|value"`). If you set up a 1 master and 3 nodes cluster, you should see one block for each node showing the subnets they have been assigned. You can associate those subnets to each node by the MAC address (VtepMAC) and IP address (Public IP) that is listed in the output.

```sh
curl -s http://fed-master:4001/v2/keys/coreos.com/network/subnets | python -mjson.tool
```

```json
{
    "node": {
        "key": "/coreos.com/network/subnets",
            {
                "key": "/coreos.com/network/subnets/18.16.29.0-24",
                "value": "{\"PublicIP\":\"192.168.122.77\",\"BackendType\":\"vxlan\",\"BackendData\":{\"VtepMAC\":\"46:f1:d0:18:d0:65\"}}"
            },
            {
                "key": "/coreos.com/network/subnets/18.16.83.0-24",
                "value": "{\"PublicIP\":\"192.168.122.36\",\"BackendType\":\"vxlan\",\"BackendData\":{\"VtepMAC\":\"ca:38:78:fc:72:29\"}}"
            },
            {
                "key": "/coreos.com/network/subnets/18.16.90.0-24",
                "value": "{\"PublicIP\":\"192.168.122.127\",\"BackendType\":\"vxlan\",\"BackendData\":{\"VtepMAC\":\"92:e2:80:ba:2d:4d\"}}"
            }
    }
}
```

* From all nodes, review the `/run/flannel/subnet.env` file.  This file was generated automatically by flannel.

```console
# cat /run/flannel/subnet.env
FLANNEL_SUBNET=18.16.29.1/24
FLANNEL_MTU=1450
FLANNEL_IPMASQ=false
```

* At this point, we have etcd running on the Kubernetes master, and flannel / docker running on Kubernetes nodes. Next steps are for testing cross-host container communication which will confirm that docker and flannel are configured properly.

* Issue the following commands on any 2 nodes:

```console
# docker run -it fedora:latest bash
bash-4.3# 
```

* This will place you inside the container. Install iproute and iputils packages to install ip and ping utilities. Due to a [bug](https://bugzilla.redhat.com/show_bug.cgi?id=1142311), it is required to modify capabilities of ping binary to work around "Operation not permitted" error.

```console
bash-4.3# yum -y install iproute iputils
bash-4.3# setcap cap_net_raw-ep /usr/bin/ping
```

* Now note the IP address on the first node:

```console
bash-4.3# ip -4 a l eth0 | grep inet
    inet 18.16.29.4/24 scope global eth0
```

* And also note the IP address on the other node:

```console
bash-4.3# ip a l eth0 | grep inet
    inet 18.16.90.4/24 scope global eth0
```

* Now ping from the first node to the other node:

```console
bash-4.3# ping 18.16.90.4
PING 18.16.90.4 (18.16.90.4) 56(84) bytes of data.
64 bytes from 18.16.90.4: icmp_seq=1 ttl=62 time=0.275 ms
64 bytes from 18.16.90.4: icmp_seq=2 ttl=62 time=0.372 ms
```

* Now Kubernetes multi-node cluster is set up with overlay networking set up by flannel.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/fedora/flannel_multi_node_cluster.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
