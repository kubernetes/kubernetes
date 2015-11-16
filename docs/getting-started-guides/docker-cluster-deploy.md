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
[here](http://releases.k8s.io/release-1.1/docs/getting-started-guides/docker-cluster-deploy.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Running Multi-Node Kubernetes Using Docker
------------------------------------------

This guide describes how to deploy Kubernetes automatically by using a Docker cluster, regardless of providers (bare metal, GCE, AWS and all OS distributions) differences. You can scale to **any number of nodes** by following the guide.

_Note_:
These instructions are somewhat significantly more advanced than the [single node](docker.md) instructions.  If you are
interested in just starting to explore Kubernetes, we recommend that you start there.

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Overview](#overview)
  - [Bootstrap Docker](#bootstrap-docker)
- [Start a Cluster](#start-a-cluster)
- [Test it out](#test-it-out)
- [Advanced Guide](#advanced-guide)
  - [Add another Node into cluster](#add-another-node-into-cluster)
  - [Support different infrastructures](#support-different-infrastructures)
  - [Start a Single Node Cluster](#start-a-single-node-cluster)
- [Deploy a DNS](#deploy-a-dns)
  - [Customize DNS of the cluster](#customize-dns-of-the-cluster)
- [Customize](#customize)
- [Tear Down](#tear-down)
- [Trouble shooting](#trouble-shooting)

## Prerequisites

1. The nodes have installed Docker with right version. There is a [bug](https://github.com/docker/docker/issues/14106) in Docker 1.7.0 that prevents this from working correctly. Please install Docker 1.6.2 or Docker 1.7.1 or higher.
2. All machines can communicate with each other, no need to connect Internet (but should configure to use
private docker registry in this case).
3. All the remote servers are ssh accessible.
4. If your machines were once deployed before, [Tear Down](#tear-down) first is highly recommended.

## Overview

This guide will set up a multi-node Kubernetes by using docker cluster, consisting of a _master_ node which hosts the API server and orchestrates work
and numbers of _worker_ node which receives work from the master.

Here's a diagram of what the final result will look like:
![Kubernetes Single Node on Docker](k8s-docker.png)

### Bootstrap Docker

This guide also uses a pattern of running two instances of the Docker daemon
   1) A _bootstrap_ Docker instance which is used to start system daemons like `flanneld` and `etcd`
   2) A _main_ Docker instance which is used for the Kubernetes infrastructure and user's scheduled containers

This pattern is necessary because the `flannel` daemon is responsible for setting up and managing the network that interconnects
all of the Docker containers created by Kubernetes.  To achieve this, it must run outside of the _main_ Docker daemon.  However,
it is still useful to use containers for deployment and management, so we create a simpler _bootstrap_ daemon to achieve this.

## Starting a Cluster

An example cluster is listed below:

| IP Address  |   Role   |
|-------------|----------|
|10.10.102.152|   node   |
|10.10.102.150|  master  |

(Optional) We always use latest `hyperkube` release as default, but you can specify k8s version before deployment:


```console
export K8S_VERSION=<your_k8s_version (e.g. 1.1.1)>
```

In `cluster/` directory:

```console
export NODES="vcap@10.10.102.152"
export MASTER="vcap@10.10.102.150"
export KUBERNETES_PROVIDER=docker
./kube-up.sh
```

> Please check `cluster/docker/config-default.sh` for more supported ENVs and default values.

If all things goes right, you will see the below message from console indicating the k8s is up.

```console
Deploy Complete!
... calling validate-cluster 
... Everything is OK! 
```

## Test it out

On every node, you can see there are two containers running by `docker ps`:

```console
kube_in_docker_proxy_xxx
kube_in_docker_kubelet_xxx
```

And on Master node, you can see extra master containers running:

```console
k8s_scheduler.xxx
k8s_apiserver.xxx
k8s_controller-manager.xxx
```

In `cluster` directory, use `$ ./kubectl.sh get nodes` to see if all of your nodes are ready.

```console
$ ./kubectl.sh get nodes
NAME            LABELS                                 STATUS
10.10.102.150   kubernetes.io/hostname=10.10.103.150   Ready
10.10.102.153   kubernetes.io/hostname=10.10.102.153   Ready
```

Then you can run Kubernetes [guest-example](../../examples/guestbook/) to build a Redis backend cluster on the k8s．

## Advanced Guide

Here are some tips for anyone interested in customize or play more with this guide.

### Add another Node into cluster

Adding a Node to existing cluster is quite easy, just enable `NODE_ONLY` mode to clarify you want to provision Node only:

```console
export NODE_ONLY=yes
export NODES="vcap@10.10.102.153"
export MASTER="vcap@10.10.102.150"
```

### Support different infrastructures

We use `baremetal` as default infrastructure provider, which means we only require a batch of machines (either VMs or physical servers) with Docker installed, and then deploy k8s on them remotely by using `scp` & `ssh`.

But we are able to supported any other infrastructure like GCE, AWS etc. So there expected be more provider specific versions based on the existing [docker-baremetal](../../cluster/docker-baremetal/) implementation, e.g. `docker-gce` or `docker-aws`, to support `gcloud ssh` or AWS identity.

And then, just tell scripts you want to deploy on other infrastructure like this:

```console
export NODES="vcap@10.10.102.152"
export MASTER="vcap@10.10.102.150"
export KUBERNETES_PROVIDER=docker
export INFRA=gce
./kube-up.sh
```


Only a few functions need to be rewrite, so welcome to contribute!


### Deploy a Single Node Cluster

Just set your MASTER and NODES to the same machine:

| IP Address  |   Role   |
|-------------|----------|
|10.10.102.150|   node   |
|10.10.102.150|  master  |

> NOTE: We do not recommend deploy a machine as both Master and Node unless you are lack of machines, because e2e test may complain.



## Deploy a DNS

The DNS ENVs are also defined in `cluster/docker/config-default.sh`. You do not need change anything by default.

```console
$ cd cluster/docker
$ KUBERNETES_PROVIDER=docker ./deployAddons.sh
```

After that, you can use `$ ./kubectl get pods --namespace=kube-system` to see the DNS pods are running in the cluster.


#### Customize DNS of the cluster

You can customize your cluster DNS **before deployment**:

```console
$ export DNS_SERVER_IP=<your-own-valid-ip>
$ export DNS_DOMAIN=<your-own-dns-name>
```

Otherwise, you need to re-deploy the cluster in `NODE_ONLY` mode after you changed those two ENVs.

And then:

```console
$ export DNS_REPLICAS=<your-own-number>
$ cd cluster/docker
$ KUBERNETES_PROVIDER=docker ./deploy-addons.sh
```

## Customize

One of the biggest benefits of using Docker to run Kubernetes is users can customize the cluster freely before you running `kube-up.sh`:

### master

You need to enable **master customization mode** before deploy:

```console
export MASTER_CONF=yes
```

The configuration file of Master locates in `cluster/images/hyperkube/master-multi.json`, which will be mounted as volume for Master Pod to consume, you can modify it freely **before deploying**.

In this mode, you can even change the configuration of Master after the deployment has done, see:

1. Login the Master node
2. Change the content in `~/docker/kube-config/master-multi.json`

kubelet will auto-restart the affected master pod after you saved your change.

### kubelet

Except a few basic options defined in `kube-deploy/node.sh`, you can customize the `docker/kube-config/kubelet.env` freely to add or update `kubelet` options **before deploying**.

## Tear Down

In `cluster/` directory, please make sure the Nodes and Master ENVs are set correctly:

```sh
export KUBERNETES_PROVIDER=docker ./kube-down.sh
```

## Trouble Shooting

Although using docker to deploy k8s is much simpler than ordinary way, here're some tips to follow if there's any trouble.

### What did the scripts do?

1. Start a bootstrap daemon
2. Start `flannel` on every node's bootstrap daemon, `etcd` on Master's bootstrap daemon
3. Start `kubelet` & `proxy` containers by using `hyperkube` image on every node
4. `kubelet` on the Master node will start master Pod (contains `api-server`, `controller-manager` & `scheduler`)from a built in `json` file in hyperkube image.

See [docker-multinode/master.md](docker-multinode/master.md) and [docker-multinode/worker.md](docker-multinode/worker.md) for detailed instructions explanation.

### Useful tips

1. Make sure you have access to the images stored in `gcr.io`, otherwise, you need to manually load `hyperkube` image into your docker daemon and `etcd` into docker bootstrap daemon.

2. As we said, there are two kinds of daemon running on a node. The bootstrap daemon works on `-H unix:///var/run/docker-bootstrap.sock` with work_dir `/var/lib/docker-bootstrap`. Thus re-configuring and restarting docker daemon will never influence Etcd and Flanneld.

3. For k8s admins, you should learn to manage process by using docker container, `docker ps`, `docker logs` & `docker exec` solve most problems.

### Limitations

1. Due to `kubelet` runs insider docker container, there may be issues for plug-in supported volume, as Docker does not support mount propagation. See the root cause: [docker #17034](https://github.com/docker/docker/pull/17034). Notice: `hostDir` and `emptyDir` will not be influenced, Secret volume has been fixed by [#13791](https://github.com/kubernetes/kubernetes/pull/13791), but other volume types handled by `kubelet` like NFS volume may be exposed to this problem.

2. ServiceAccounts can not work now: please track [#17213](https://github.com/kubernetes/kubernetes/pull/17213)



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/docker-cluster-deploy.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
