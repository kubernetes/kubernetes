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
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/azure.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Getting started on Microsoft Azure
----------------------------------

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Getting started with your cluster](#getting-started-with-your-cluster)
- [Tearing down the cluster](#tearing-down-the-cluster)


## Prerequisites

** Azure Prerequisites**

1. You need an Azure account. Visit http://azure.microsoft.com/ to get started.
2. Install and configure the Azure cross-platform command-line interface. http://azure.microsoft.com/en-us/documentation/articles/xplat-cli/
3. Make sure you have a default account set in the Azure cli, using `azure account set`

**Prerequisites for your workstation**

1. Be running a Linux or Mac OS X.
2. Get or build a [binary release](binary_release.md)
3. If you want to build your own release, you need to have [Docker
installed](https://docs.docker.com/installation/).  On Mac OS X you can use
[boot2docker](http://boot2docker.io/).

## Setup

### Starting a cluster

The cluster setup scripts can setup Kubernetes for multiple targets. First modify `cluster/kube-env.sh` to specify azure:

    KUBERNETES_PROVIDER="azure"

Next, specify an existing virtual network and subnet in `cluster/azure/config-default.sh`:

    AZ_VNET=<vnet name>
    AZ_SUBNET=<subnet name>

You can create a virtual network:

    azure network vnet create <vnet name> --subnet-name=<subnet name> --location "West US" -v

Now you're ready.

You can download and install the latest Kubernetes release from [this page](https://github.com/GoogleCloudPlatform/kubernetes/releases), then run the `<kubernetes>/cluster/kube-up.sh` script to start the cluster:

    cd kubernetes
    cluster/kube-up.sh

The script above will start (by default) a single master VM along with 4 worker VMs.  You
can tweak some of these parameters by editing `cluster/azure/config-default.sh`.

### Adding the Kubernetes command line tools to PATH

The [kubectl](../../docs/user-guide/kubectl/kubectl.md) tool controls the Kubernetes cluster manager.  It lets you inspect your cluster resources, create, delete, and update components, and much more.
You will use it to look at your new cluster and bring up example apps.

Add the appropriate binary folder to your `PATH` to access kubectl:

    # OS X
    export PATH=<path/to/kubernetes-directory>/platforms/darwin/amd64:$PATH

    # Linux
    export PATH=<path/to/kubernetes-directory>/platforms/linux/amd64:$PATH

## Getting started with your cluster

See [a simple nginx example](../user-guide/simple-nginx.md) to try out your new cluster.

For more complete applications, please look in the [examples directory](../../examples/).

## Tearing down the cluster

```sh
cluster/kube-down.sh
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/azure.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
