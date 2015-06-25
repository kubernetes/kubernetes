# Cluster Management

This doc is in progress.

## Upgrading a cluster

The `cluster/kube-push.sh` script will do a rudimentary update; it is a 1.0 roadmap item to have a robust live cluster update system.

## Updgrading to a different API version

There is a sequence of steps to upgrade to a new API version.

1. Turn on the new api version
2. Upgrade the cluster's storage to use the new version.
3. Upgrade all config files. Identify users of the old api version endpoints.
4. Update existing objects in the storage to new version by running cluster/update-storage-objects.sh
3. Turn off the old version.

### Turn on or off an API version for your cluster

Specific API versions can be turned on or off by passing --runtime-config=api/<version> flag while bringing up the server. For example: to turn off v1 API, pass --runtime-config=api/v1=false.
runtime-config also supports 2 special keys: api/all and api/legacy to control all and legacy APIs respectively. For example, for turning off all api versions except v1, pass --runtime-config=api/all=false,api/v1=true.

### Switching your cluster's storage API version

KUBE_API_VERSIONS env var controls the API versions that are supported in the cluster. The first version in the list is used as the cluster's storage version. Hence, to set a specific version as the storage version, bring it to the front of list of versions in the value of KUBE_API_VERSIONS.

### Switching your config files to a new API version

You can use the kube-version-change utility to convert config files between different API versions.

```
$ hack/build-go.sh cmd/kube-version-change
$ _output/local/go/bin/kube-version-change -i myPod.v1beta3.yaml -o myPod.v1.yaml
```

### Maintenance on a Node

If you need to reboot a node (such as for a kernel upgrade, libc upgrade, hardware repair, etc.), and the downtime is
brief, then when the Kubelet restarts, it will attempt to restart the pods scheduled to it.  If the reboot takes longer,
then the node controller will terminate the pods that are bound to the unavailable node.  If there is a corresponding
replication controller, then a new copy of the pod will be started on a different node.  So, in the case where all
pods are replicated, upgrades can be done without special coordination.

If you want more control over the upgrading process, you may use the following workflow:
  1. Mark the node to be rebooted as unschedulable:
    `kubectl update nodes $NODENAME --patch='{"apiVersion": "v1", "spec": {"unschedulable": true}}'`. 
    This keeps new pods from landing on the node while you are trying to get them off.
  1. Get the pods off the machine, via any of the following strategies:
    1. wait for finite-duration pods to complete
    1. delete pods with `kubectl delete pods $PODNAME`
    1. for pods with a replication controller, the pod will eventually be replaced by a new pod which will be scheduled to a new node. additionally, if the pod is part of a service, then clients will automatically be redirected to the new pod.
    1. for pods with no replication controller, you need to bring up a new copy of the pod, and assuming it is not part of a service, redirect clients to it.
  1. Work on the node
  1. Make the node schedulable again:
    `kubectl update nodes $NODENAME --patch='{"apiVersion": "v1", "spec": {"unschedulable": false}}'`.  
    If you deleted the node's VM instance and created a new one, then a new schedulable node resource will
    be created automatically when you create a new VM instance (if you're using a cloud provider that supports
    node discovery; currently this is only GCE, not including CoreOS on GCE using kube-register). See [Node](node.md).


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/cluster_management.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/cluster_management.md?pixel)]()
