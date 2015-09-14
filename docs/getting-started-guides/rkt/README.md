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
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/rkt/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Run Kubernetes with rkt

This document describes how to run Kubernetes using [rkt](https://github.com/coreos/rkt) as a container runtime.
We still have [a bunch of work](http://issue.k8s.io/8262) to do to make the experience with rkt wonderful, please stay tuned!

### **Prerequisite**

- [systemd](http://www.freedesktop.org/wiki/Software/systemd/) should be installed on the machine and should be enabled. The minimum version required at this moment (2015/09/01) is 219
  *(Note that systemd is not required by rkt itself, we are using it here to monitor and manage the pods launched by kubelet.)*

- Install the latest rkt release according to the instructions [here](https://github.com/coreos/rkt).
  The minimum version required for now is [v0.8.0](https://github.com/coreos/rkt/releases/tag/v0.8.0).

- Note that for rkt version later than v0.7.0, `metadata service` is not required for running pods in private networks. So now rkt pods will not register the metadata service be default.

### Local cluster

To use rkt as the container runtime, we need to supply `--container-runtime=rkt` and `--rkt-path=$PATH_TO_RKT_BINARY` to kubelet. Additionally we can provide `--rkt-stage1-image` flag
as well to select which [stage1 image](https://github.com/coreos/rkt/blob/master/Documentation/running-lkvm-stage1.md) we want to use.

If you are using the [hack/local-up-cluster.sh](../../../hack/local-up-cluster.sh) script to launch the local cluster, then you can edit the environment variable `CONTAINER_RUNTIME`, `RKT_PATH` and `RKT_STAGE1_IMAGE` to
set these flags:

```console
$ export CONTAINER_RUNTIME=rkt
$ export RKT_PATH=$PATH_TO_RKT_BINARY
$ export RKT_STAGE1_IMAGE=PATH=$PATH_TO_STAGE1_IMAGE
```

Then we can launch the local cluster using the script:

```console
$ hack/local-up-cluster.sh
```

### CoreOS cluster on Google Compute Engine (GCE)

To use rkt as the container runtime for your CoreOS cluster on GCE, you need to specify the OS distribution, project, image:

```console
$ export KUBE_OS_DISTRIBUTION=coreos
$ export KUBE_GCE_MINION_IMAGE=<image_id>
$ export KUBE_GCE_MINION_PROJECT=coreos-cloud
$ export KUBE_CONTAINER_RUNTIME=rkt
```

You can optionally choose the version of rkt used by setting `KUBE_RKT_VERSION`:

```console
$ export KUBE_RKT_VERSION=0.8.0
```

Then you can launch the cluster by:

```console
$ kube-up.sh
```

Note that we are still working on making all containerized the master components run smoothly in rkt. Before that we are not able to run the master node with rkt yet.

### CoreOS cluster on AWS

To use rkt as the container runtime for your CoreOS cluster on AWS, you need to specify the provider and OS distribution:

```console
$ export KUBERNETES_PROVIDER=aws
$ export KUBE_OS_DISTRIBUTION=coreos
$ export KUBE_CONTAINER_RUNTIME=rkt
```

You can optionally choose the version of rkt used by setting `KUBE_RKT_VERSION`:

```console
$ export KUBE_RKT_VERSION=0.8.0
```

You can optionally choose the CoreOS channel  by setting `COREOS_CHANNEL`:

```console
$ export COREOS_CHANNEL=stable
```

Then you can launch the cluster by:

```console
$ kube-up.sh
```

Note: CoreOS is not supported as the master using the automated launch
scripts. The master node is always Ubuntu.

### Getting started with your cluster

See [a simple nginx example](../../../docs/user-guide/simple-nginx.md) to try out your new cluster.

For more complete applications, please look in the [examples directory](../../../examples/).


### Debugging

Here are severals tips for you when you run into any issues.

##### Check logs

By default, the log verbose level is 2. In order to see more logs related to rkt, we can set the verbose level to 4.
For local cluster, we can set the environment variable: `LOG_LEVEL=4`.
If the cluster is using salt, we can edit the [logging.sls](../../../cluster/saltbase/pillar/logging.sls) in the saltbase.

##### Check rkt pod status

To check the pods' status, we can use rkt command, such as `rkt list`, `rkt status`, `rkt image list`, etc.
More information about rkt command line can be found [here](https://github.com/coreos/rkt/blob/master/Documentation/commands.md)

##### Check journal logs

As we use systemd to launch rkt pods(by creating service files which will run `rkt run-prepared`, we can check the pods' log
using `journalctl`:

- Check the running state of the systemd service:

```console
$ sudo journalctl -u $SERVICE_FILE
```

where `$SERVICE_FILE` is the name of the service file created for the pod, you can find it in the kubelet logs.

##### Check the log of the container in the pod:

```console
$ sudo journalctl -M rkt-$UUID -u $CONTAINER_NAME
```

where `$UUID` is the rkt pod's UUID, which you can find via `rkt list --full`, and `$CONTAINER_NAME` is the container's name.

##### Check Kubernetes events, logs.

Besides above tricks, Kubernetes also provides us handy tools for debugging the pods. More information can be found [here](../../../docs/user-guide/application-troubleshooting.md)

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/rkt/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
