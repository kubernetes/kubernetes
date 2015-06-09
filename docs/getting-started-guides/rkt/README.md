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
  ```shell
  $ sudo rkt metadata-service
  ```

  If you want the service to be running as a systemd service, then:
  ```shell
  $ sudo systemd-run rkt metadata-service
  ```
  Alternatively, you can use the [rkt-metadata.service](https://github.com/coreos/rkt/blob/master/dist/init/systemd/rkt-metadata.service) and [rkt-metadata.socket](https://github.com/coreos/rkt/blob/master/dist/init/systemd/rkt-metadata.socket) to start the service.


### Local cluster

To use rkt as the container runtime, you just need to set the environment variable `CONTAINER_RUNTIME`:
```shell
$ export CONTAINER_RUNTIME=rkt
$ hack/local-up-cluster.sh
```

### CoreOS cluster on GCE

To use rkt as the container runtime for your CoreOS cluster on GCE, you need to specify the OS distribution, project, image:
```shell
$ export KUBE_OS_DISTRIBUTION=coreos
$ export KUBE_GCE_MINION_IMAGE=<image_id>
$ export KUBE_GCE_MINION_PROJECT=coreos-cloud
$ export KUBE_CONTAINER_RUNTIME=rkt
```

You can optionally choose the version of rkt used by setting `KUBE_RKT_VERSION`:
```shell
$ export KUBE_RKT_VERSION=0.5.6
```

Then you can launch the cluster by:
````shell
$ kube-up.sh
```

Note that we are still working on making all containerized the master components run smoothly in rkt. Before that we are not able to run the master node with rkt yet.

### CoreOS cluster on AWS

To use rkt as the container runtime for your CoreOS cluster on AWS, you need to specify the provider and OS distribution:
```shell
$ export KUBERNETES_PROVIDER=aws
$ export KUBE_OS_DISTRIBUTION=coreos
$ export KUBE_CONTAINER_RUNTIME=rkt
```

You can optionally choose the version of rkt used by setting `KUBE_RKT_VERSION`:
```shell
$ export KUBE_RKT_VERSION=0.5.6
```

You can optionally choose the CoreOS channel  by setting `COREOS_CHANNEL`:
```shell
$ export COREOS_CHANNEL=stable
```

Then you can launch the cluster by:
````shell
$ kube-up.sh
```

Note: CoreOS is not supported as the master using the automated launch
scripts. The master node is always Ubuntu.

### Getting started with your cluster
See [a simple nginx example](../../examples/simple-nginx.md) to try out your new cluster.

For more complete applications, please look in the [examples directory](../../examples).


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/rkt/README.md?pixel)]()
