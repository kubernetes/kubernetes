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
We still have [a bunch of work](https://github.com/GoogleCloudPlatform/kubernetes/issues/8262) to do to make the experience with rkt wonderful, please stay tuned!

### **Prerequisite**

- [systemd](http://www.freedesktop.org/wiki/Software/systemd/) should be installed on your machine and should be enabled. The minimum version required at this moment (2015/05/28) is [215](http://lists.freedesktop.org/archives/systemd-devel/2014-July/020903.html).
  *(Note that systemd is not required by rkt itself, we are using it here to monitor and manage the pods launched by kubelet.)*

- Install the latest rkt release according to the instructions [here](https://github.com/coreos/rkt).
  The minimum version required for now is [v0.5.6](https://github.com/coreos/rkt/releases/tag/v0.5.6).

- Make sure the `rkt metadata service` is running because it is necessary for running pod in private network mode.
  More details about the networking of rkt can be found in the [documentation](https://github.com/coreos/rkt/blob/master/Documentation/networking.md).

  To start the `rkt metadata service`, you can simply run:

  ```console
  $ sudo rkt metadata-service
  ```

  If you want the service to be running as a systemd service, then:

  ```console
  $ sudo systemd-run rkt metadata-service
  ```

  Alternatively, you can use the [rkt-metadata.service](https://github.com/coreos/rkt/blob/master/dist/init/systemd/rkt-metadata.service) and [rkt-metadata.socket](https://github.com/coreos/rkt/blob/master/dist/init/systemd/rkt-metadata.socket) to start the service.


### Local cluster

To use rkt as the container runtime, you just need to set the environment variable `CONTAINER_RUNTIME`:

```console
$ export CONTAINER_RUNTIME=rkt
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
$ export KUBE_RKT_VERSION=0.5.6
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
$ export KUBE_RKT_VERSION=0.5.6
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


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/rkt/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
