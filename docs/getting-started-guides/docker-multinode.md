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
[here](http://releases.k8s.io/release-1.1/docs/getting-started-guides/docker-multinode.md).

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
- [Starting a Cluster](#starting-a-cluster)
    - [Add another Node into cluster](#add-another-node-into-cluster)
    - [Support different infrastructures](#support-different-infrastructures)
- [Test it out](#test-it-out)
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

Optional, you can specify version before deployment, otherwise, we'll use latest `hyperkube` release as default.

```sh
export K8S_VERSION=<your_k8s_version (e.g. 1.0.3)>
```

In `cluster/` directory:

```sh
export NODES="vcap@10.10.102.152"
export MASTER="vcap@10.10.102.150"
export KUBERNETES_PROVIDER=docker
./kube-up.sh
```

> Please check `cluster/docker/config-default.sh` for more supported ENVs

> If your MASTER appears in NODES, that MASTER will be deployed as **both master & node**. But we do not recommend this as e2e test may complain.

If all things goes right, you will see the below message from console indicating the k8s is up.

```console
Deploy Complete!
... calling validate-cluster 
... Everything is OK! 
```

### Add another Node into cluster

Adding a Node to existing cluster is quite easy, just set `NODE_ONLY` to clarify you want to provision Node only:

```console
$ export MASTER_IP=<your_master_ip (e.g. 1.2.3.4)>
$ cd kubernetes/docs/getting-started-guides/docker-multinode/
$ ./master.sh
```

### Support different infrastructures

We use `baremetal` as default infrastructure provider, which means we only require a batch of machines (VMs or physical servers) with Docker installed, and then deploy k8s on them remotely by using `scp` & `ssh`.

But we also planned to supported any other infrastructure like GCE, AWS etc. So there will be more provider specific versions based on the existing [docker-baremetal](../../cluster/docker-baremetal/) implementation, e.g. `docker-gce` or `docker-aws`, to support `gcloud ssh` or AWS identity.

And then tell scripts you want to deploy on other infrastructure like this:

```sh
export NODES="vcap@10.10.102.152"
export MASTER="vcap@10.10.102.150"
export KUBERNETES_PROVIDER=docker
export INFRA=gce
./kube-up.sh
```


Only a few functions need to be rewrite, so welcome to contribute!

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

As we use `hyperkube` image to run k8s, we **do not** need to compile binaries, please download and extract `kubectl` binary from [releases page](https://github.com/kubernetes/kubernetes/releases).

At last, use `$ kubectl get nodes` to see if all of your nodes are ready.

```console
$ kubectl get nodes
NAME            LABELS                                 STATUS
10.10.102.150   kubernetes.io/hostname=10.10.103.150   Ready
10.10.102.153   kubernetes.io/hostname=10.10.102.153   Ready
```

### Deploy a DNS

See [here](docker-multinode/deployDNS.md) for instructions.

Then you can run Kubernetes [guest-example](../../examples/guestbook/) to build a Redis backend cluster on the k8s．

## Customize

One of the biggest benefits of using Docker to run Kubernetes is users can customize the cluster freely before you running `kube-up.sh`:

### master

You need to enable **customized master mode** before deploy:

```console
$ export MASTER_IP=<your_master_ip (e.g. 1.2.3.4)>
$ cd kubernetes/docs/getting-started-guides/docker-multinode/
$ ./worker.sh
```

The configuration file of Master locates in `cluster/images/hyperkube/master-multi.json`, which will be mounted as volume for Master Pod to consume, you can modify it freely **before deploying**.

In this mode, you can even change the configuration of Master after the deployment has done, see:

1. Login the Master node
2. Change the content in `~/docker/kube-config/master-multi.json`

kubelet will auto-restart the affected master pod.

### kubelet

Except a few basic options defined in `kube-deploy/master.sh|node.sh`, you can customize the `docker/kube-config/kubelet.env` freely to add or update `kubelet` options **before deploying**.

## Tear Down

In `cluster/` directory:

```sh
export NODES="vcap@10.10.102.150 vcap@10.10.102.152"
export KUBERNETES_PROVIDER=docker ./kube-down.sh
```

## Trouble shooting

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

Due to `kubelet` runs insider docker container, there's known issue of secrets volume failure as there's no mount propagation. See: [#13791](https://github.com/kubernetes/kubernetes/pull/13791) , and the root cause: [docker #15648](https://github.com/docker/docker/pull/15648)

`hostDir` and `emptyDir` will not be influenced, but other volume types handled by `kubelet` like NFS volume will also exposed to the issues above


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/docker-multinode.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
