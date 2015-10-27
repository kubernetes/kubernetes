<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
Running Multi-Node Kubernetes Using Docker
------------------------------------------

_Note_:
These instructions are somewhat significantly more advanced than the [single node](docker.md) instructions.  If you are
interested in just starting to explore Kubernetes, we recommend that you start there.

_Note_:
There is a [bug](https://github.com/docker/docker/issues/14106) in Docker 1.7.0 that prevents this from working correctly.
Please install Docker 1.6.2 or Docker 1.7.1.

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Overview](#overview)
  - [Bootstrap Docker](#bootstrap-docker)
- [Master Node](#master-node)
- [Adding a worker node](#adding-a-worker-node)
- [Deploy a DNS](#deploy-a-dns)
- [Testing your cluster](#testing-your-cluster)

## Prerequisites

1. You need a machine with docker of right version installed.

## Overview

This guide will set up a 2-node Kubernetes cluster, consisting of a _master_ node which hosts the API server and orchestrates work
and a _worker_ node which receives work from the master.  You can repeat the process of adding worker nodes an arbitrary number of
times to create larger clusters.

Here's a diagram of what the final result will look like:
![Kubernetes Single Node on Docker](k8s-docker.png)

### Bootstrap Docker

This guide also uses a pattern of running two instances of the Docker daemon
   1) A _bootstrap_ Docker instance which is used to start system daemons like `flanneld` and `etcd`
   2) A _main_ Docker instance which is used for the Kubernetes infrastructure and user's scheduled containers

This pattern is necessary because the `flannel` daemon is responsible for setting up and managing the network that interconnects
all of the Docker containers created by Kubernetes.  To achieve this, it must run outside of the _main_ Docker daemon.  However,
it is still useful to use containers for deployment and management, so we create a simpler _bootstrap_ daemon to achieve this.

You can specify k8s version on very node before install:

```
export K8S_VERSION=<your_k8s_version (e.g. 1.0.3)>
```

Otherwise, we'll use latest `hyperkube` image as default k8s version.

## Master Node

The first step in the process is to initialize the master node.

Clone the Kubernetes repo, and run [master.sh](docker-multinode/master.sh) on the master machine with root:

```sh
cd kubernetes/docs/getting-started-guides/docker-multinode/
./master.sh
```

`Master done!`

See [here](docker-multinode/master.md) for detailed instructions explanation.

## Adding a worker node

Once your master is up and running you can add one or more workers on different machines.

Clone the Kubernetes repo, and run [worker.sh](docker-multinode/worker.sh) on the worker machine with root:

```sh
export MASTER_IP=<your_master_ip (e.g. 1.2.3.4)>
cd kubernetes/docs/getting-started-guides/docker-multinode/
./worker.sh
```

`Worker done!`

See [here](docker-multinode/worker.md) for detailed instructions explanation.

## Deploy a DNS

See [here](docker-multinode/deployDNS.md) for instructions.

## Testing your cluster

Once your cluster has been created you can [test it out](docker-multinode/testing.md)

For more complete applications, please look in the [examples directory](../../examples/)




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/docker-multinode.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
