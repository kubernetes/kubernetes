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
[here](http://releases.k8s.io/release-1.0/docs/admin/cluster-management.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Cluster Management

This document describes several topics related to the lifecycle of a cluster: creating a new cluster,
upgrading your cluster's
master and worker nodes, performing node maintenance (e.g. kernel upgrades), and upgrading the Kubernetes API version of a
running cluster.

## Creating and configuring a Cluster

To install Kubernetes on a set of machines, consult one of the existing [Getting Started guides](../../docs/getting-started-guides/README.md) depending on your environment.

## Upgrading a cluster

The current state of cluster upgrades is provider dependent.

### Master Upgrades

Both Google Container Engine (GKE) and
Compute Engine Open Source (GCE-OSS) support node upgrades via a [Managed Instance Group](https://cloud.google.com/compute/docs/instance-groups/).
Managed Instance Group upgrades sequentially delete and recreate each virtual machine, while maintaining the same
Persistent Disk (PD) to ensure that data is retained across the upgrade.

In contrast, the `kube-push.sh` process used on [other platforms](#other-platforms) attempts to upgrade the binaries in
places, without recreating the virtual machines.

### Node Upgrades

Node upgrades for GKE and GCE-OSS again use a Managed Instance Group, each node is sequentially destroyed and then recreated with new software.  Any Pods that are running
on that node need to be controlled by a Replication Controller, or manually re-created after the roll out.

For other platforms, `kube-push.sh` is again used, performing an in-place binary upgrade on existing machines.

### Upgrading Google Container Engine (GKE)

Google Container Engine automatically updates master components (e.g. `kube-apiserver`, `kube-scheduler`) to the latest
version. It also handles upgrading the operating system and other components that the master runs on.

The node upgrade process is user-initiated and is described in the [GKE documentation.](https://cloud.google.com/container-engine/docs/clusters/upgrade)

### Upgrading open source Google Compute Engine clusters

Upgrades on open source Google Compute Engine (GCE) clusters are controlled by the ```cluster/gce/upgrade.sh``` script.

Its usage is as follows:

```console
cluster/gce/upgrade.sh [-M|-N|-P] -l | <release or continuous integration version> | [latest_stable|latest_release|latest_ci]
  Upgrades master and nodes by default
  -M:  Upgrade master only
  -N:  Upgrade nodes only
  -P:  Node upgrade prerequisites only (create a new instance template)
  -l:  Use local(dev) binaries
```

For example, to upgrade just your master to a specific version (v1.0.2):

```console
cluster/gce/upgrade.sh -M v1.0.2
```

Alternatively, to upgrade your entire cluster to the latest stable release:

```console
cluster/gce/upgrade.sh latest_stable
```

### Other platforms

The `cluster/kube-push.sh` script will do a rudimentary update.  This process is still quite experimental, we
recommend testing the upgrade on an experimental cluster before performing the update on a production cluster.

## Resizing a cluster

If your cluster runs short on resources you can easily add more machines to it if your cluster is running in [Node self-registration mode](node.md#self-registration-of-nodes).
If you're using GCE or GKE it's done by resizing Instance Group managing your Nodes. It can be accomplished by modifying number of instances on `Compute > Compute Engine > Instance groups > your group > Edit group` [Google Cloud Console page](https://console.developers.google.com) or using gcloud CLI:

```
gcloud compute instance-groups managed --zone compute-zone resize my-cluster-minon-group --new-size 42
```

Instance Group will take care of putting appropriate image on new machines and start them, while Kubelet will register its Node with API server to make it available for scheduling. If you scale the instance group down, system will randomly choose Nodes to kill.

In other environments you may need to configure the machine yourself and tell the Kubelet on which machine API server is running.


### Horizontal auto-scaling of nodes (GCE)

If you are using GCE, you can configure your cluster so that the number of nodes will be automatically scaled based on:
 * CPU and memory utilization.
 * Amount of of CPU and memory requested by the pods (called also reservation).

Before setting up the cluster by ```kube-up.sh```, you can set ```KUBE_ENABLE_NODE_AUTOSCALER``` environment variable to ```true``` and export it.
The script will create an autoscaler for the instance group managing your nodes.

The autoscaler will try to maintain the average CPU/memory utilization and reservation of nodes within the cluster close to the target value.
The target value can be configured by ```KUBE_TARGET_NODE_UTILIZATION``` environment variable (default: 0.7) for ``kube-up.sh`` when creating the cluster.
Node utilization is the total node's CPU/memory usage (OS + k8s + user load) divided by the node's capacity.
Node reservation is the total CPU/memory requested by pods that are running on the node divided by the node's capacity.
If the desired numbers of nodes in the cluster resulting from CPU/memory utilization/reservation are different,
the autoscaler will choose the bigger number. The number of nodes in the cluster set by the autoscaler will be limited from ```KUBE_AUTOSCALER_MIN_NODES``` (default: 1)
to ```KUBE_AUTOSCALER_MAX_NODES``` (default: the initial number of nodes in the cluster).

The autoscaler is implemented as a Compute Engine Autoscaler.
The initial values of the autoscaler parameters set by ``kube-up.sh`` and some more advanced options can be tweaked on
`Compute > Compute Engine > Instance groups > your group > Edit group`[Google Cloud Console page](https://console.developers.google.com)
or using gcloud CLI:

```
gcloud preview autoscaler --zone compute-zone <command>
```

Note that autoscaling will work properly only if node metrics are accessible in Google Cloud Monitoring.
To make the metrics accessible, you need to create your cluster with ```KUBE_ENABLE_CLUSTER_MONITORING```
equal to ```google``` or ```googleinfluxdb``` (```googleinfluxdb``` is the default value). Please also make sure
that you have Google Cloud Monitoring API enabled in Google Developer Console.

## Maintenance on a Node

If you need to reboot a node (such as for a kernel upgrade, libc upgrade, hardware repair, etc.), and the downtime is
brief, then when the Kubelet restarts, it will attempt to restart the pods scheduled to it.  If the reboot takes longer,
then the node controller will terminate the pods that are bound to the unavailable node.  If there is a corresponding
replication controller, then a new copy of the pod will be started on a different node.  So, in the case where all
pods are replicated, upgrades can be done without special coordination, assuming that not all nodes will go down at the same time.

If you want more control over the upgrading process, you may use the following workflow:

Mark the node to be rebooted as unschedulable:

```console
kubectl replace nodes $NODENAME --patch='{"apiVersion": "v1", "spec": {"unschedulable": true}}'
```

This keeps new pods from landing on the node while you are trying to get them off.

Get the pods off the machine, via any of the following strategies:
   * Wait for finite-duration pods to complete.
   * Delete pods with:

```console
kubectl delete pods $PODNAME
```

For pods with a replication controller, the pod will eventually be replaced by a new pod which will be scheduled to a new node. Additionally, if the pod is part of a service, then clients will automatically be redirected to the new pod.

For pods with no replication controller, you need to bring up a new copy of the pod, and assuming it is not part of a service, redirect clients to it.

Perform maintenance work on the node.

Make the node schedulable again:

```console
kubectl replace nodes $NODENAME --patch='{"apiVersion": "v1", "spec": {"unschedulable": false}}'
```

If you deleted the node's VM instance and created a new one, then a new schedulable node resource will
be created automatically when you create a new VM instance (if you're using a cloud provider that supports
node discovery; currently this is only Google Compute Engine, not including CoreOS on Google Compute Engine using kube-register). See [Node](node.md) for more details.

## Advanced Topics

### Upgrading to a different API version

When a new API version is released, you may need to upgrade a cluster to support the new API version (e.g. switching from 'v1' to 'v2' when 'v2' is launched)

This is an infrequent event, but it requires careful management. There is a sequence of steps to upgrade to a new API version.

   1. Turn on the new api version.
   1. Upgrade the cluster's storage to use the new version.
   1. Upgrade all config files. Identify users of the old API version endpoints.
   1. Update existing objects in the storage to new version by running `cluster/update-storage-objects.sh`.
   1. Turn off the old API version.

### Turn on or off an API version for your cluster

Specific API versions can be turned on or off by passing --runtime-config=api/<version> flag while bringing up the API server. For example: to turn off v1 API, pass `--runtime-config=api/v1=false`.
runtime-config also supports 2 special keys: api/all and api/legacy to control all and legacy APIs respectively.
For example, for turning off all api versions except v1, pass `--runtime-config=api/all=false,api/v1=true`.
For the purposes of these flags, _legacy_ APIs are those APIs which have been explicitly deprecated (e.g. `v1beta3`).

### Switching your cluster's storage API version

The objects that are stored to disk for a cluster's internal representation of the Kubernetes resources active in the cluster are written using a particular version of the API.
When the supported API changes, these objects may need to be rewritten in the newer API.  Failure to do this will eventually result in resources that are no longer decodable or usable
by the kubernetes API server.

`KUBE_API_VERSIONS` environment variable for the `kube-apiserver` binary which controls the API versions that are supported in the cluster. The first version in the list is used as the cluster's storage version. Hence, to set a specific version as the storage version, bring it to the front of list of versions in the value of `KUBE_API_VERSIONS`.  You need to restart the `kube-apiserver` binary
for changes to this variable to take effect.

### Switching your config files to a new API version

You can use the `kube-version-change` utility to convert config files between different API versions.

```console
$ hack/build-go.sh cmd/kube-version-change
$ _output/local/go/bin/kube-version-change -i myPod.v1beta3.yaml -o myPod.v1.yaml
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/cluster-management.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
