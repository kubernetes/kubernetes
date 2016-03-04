<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


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

- Since release [v1.2.0-alpha.5](https://github.com/kubernetes/kubernetes/releases/tag/v1.2.0-alpha.5),
the [rkt API service](https://github.com/coreos/rkt/blob/master/api/v1alpha/README.md)
must be running on the node.

### Network Setup

rkt uses the [Container Network Interface (CNI)](https://github.com/appc/cni)
to manage container networking. By default, all pods attempt to join a network
called `rkt.kubernetes.io`, which is currently defined [in `rkt.go`]
(https://github.com/kubernetes/kubernetes/blob/v1.2.0-alpha.6/pkg/kubelet/rkt/rkt.go#L91).
In order for pods to get correct IP addresses, the CNI config file must be
edited to add this `rkt.kubernetes.io` network:

#### Using flannel

In addition to the basic prerequisites above, each node must be running
a [flannel](https://github.com/coreos/flannel) daemon. This implies
that a flannel-supporting etcd service must be available to the cluster
as well, apart from the Kubernetes etcd, which will not yet be
available at flannel configuration time. Once it's running, flannel can
be set up with a CNI config like:

```console
$ cat <<EOF >/etc/rkt/net.d/k8s_cluster.conf
{
    "name": "rkt.kubernetes.io",
    "type": "flannel"
}
EOF
```

While `k8s_cluster.conf` is a rather arbitrary name for the config file itself,
and can be adjusted to suit local conventions, the keys and values should be exactly
as shown above. `name` must be `rkt.kubernetes.io` and `type` should be `flannel`.
More details about the flannel CNI plugin can be found
[in the CNI documentation](https://github.com/appc/cni/blob/master/Documentation/flannel.md).

#### On GCE

Each VM on GCE has an additional 256 IP addresses routed to it, so
it is possible to forego flannel in smaller clusters. This makes the
necessary CNI config file a bit more verbose:

```console
$ cat <<EOF >/etc/rkt/net.d/k8s_cluster.conf
{
    "name": "rkt.kubernetes.io",
    "type": "bridge",
    "bridge": "cbr0",
    "isGateway": true,
    "ipam": {
        "type": "host-local",
        "subnet": "10.255.228.1/24",
        "gateway": "10.255.228.1"
    },
    "routes": [
      { "dst": "0.0.0.0/0" }
    ]
}
EOF
```

This example creates a `bridge` plugin configuration for the CNI network, specifying
the bridge name `cbr0`. It also specifies the CIDR, in the `ipam` field.

Creating these files for any moderately-sized cluster is at best inconvenient.
Work is in progress to
[enable Kubernetes to use the CNI by default]
(https://github.com/kubernetes/kubernetes/pull/18795/files).
As that work matures, such manual CNI config munging will become unnecessary
for primary use cases. For early adopters, an initial example shows one way to
[automatically generate these CNI configurations]
(https://gist.github.com/yifan-gu/fbb911db83d785915543)
for rkt.

### Local cluster

To use rkt as the container runtime, we need to supply the following flags to kubelet:
- `--container-runtime=rkt` chooses the container runtime to use. Possible values: 'docker', 'rkt'. Default: 'docker'.
- `--rkt-path=$PATH_TO_RKT_BINARY` sets the path of rkt binary. Leave empty to use the first rkt in $PATH.
- `--rkt-stage1-image` sets the path of the stage1 image. Local paths and http/https URLs are supported. Leave empty to use the 'stage1.aci' that locates in the same directory as the rkt binary.

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
$ export KUBE_GCE_NODE_IMAGE=<image_id>
$ export KUBE_GCE_NODE_PROJECT=coreos-cloud
$ export KUBE_CONTAINER_RUNTIME=rkt
```

You can optionally choose the version of rkt used by setting `KUBE_RKT_VERSION`:

```console
$ export KUBE_RKT_VERSION=0.15.0
```

Then you can launch the cluster by:

```console
$ cluster/kube-up.sh
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

### Different UX with rkt container runtime

rkt and Docker have very different designs, as well as ACI and Docker image format. Users might experience some different experience when switching from one to the other. More information can be found [here](notes.md).

### Debugging

Here are several tips for you when you run into any issues.

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



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/rkt/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
