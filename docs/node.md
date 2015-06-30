# Node

## What is a node?

`Node` is a worker machine in Kubernetes, previously known as `Minion`. Node
may be a VM or physical machine, depending on the cluster. Each node has
the services necessary to run [Pods](pods.md) and be managed from the master
systems. The services include docker, kubelet and network proxy. See
[The Kubernetes Node](design/architecture.md#the-kubernetes-node) section in design
doc for more details.

## Node Status

Node status describes current status of a node. For now, there are the following
pieces of information:

### Node Addresses

<!--- TODO: this section is outdated. There is no HostIP field in the API,
but there are addresses of type InternalIP and ExternalIP -->
Host IP address is queried from cloudprovider and stored as part of node
status. If kubernetes runs without cloudprovider, node's ID will be used.
IP address can change, and there are different kind of IPs, e.g. public
IP, private IP, dynamic IP, ipv6, etc. It makes more sense to save it as
a status rather than spec.

### Node Phase

Node Phase is the current lifecycle phase of node, one of `Pending`,
`Running` and `Terminated`. Node Phase management is under development,
here is a brief overview: In kubernetes, node will be created in `Pending`
phase, until it is discovered and checked in by kubernetes, at which time,
kubernetes will mark it as `Running`. The end of a node's lifecycle is
`Terminated`. A terminated node will not receive any scheduling request,
and any running pods will be removed from the node.

Node with `Running` phase is necessary but not sufficient requirement for
scheduling Pods. For a node to be considered a scheduling candidate, it
must have appropriate conditions, see below.

### Node Condition
Node Condition describes the conditions of `Running` nodes. (However,
it can be present also when node status is different, e.g. `Unknown`)
Current valid condition is `Ready`. In the future, we plan to add more.
`Ready` means kubelet is healthy and ready to accept pods. Different
condition provides different level of understanding for node health.
Node condition is represented as a json object. For example,
the following conditions mean the node is in sane state:
```json
"conditions": [
  {
    "kind": "Ready",
    "status": "True",
    },
]
```

### Node Capacity

Describes the resources available on the node: CPUs, memory and the maximum
number of pods that can be scheduled on this node.

### Node Info
General information about the node, for instance kernel version, kubernetes
version, docker version (if used), OS name. The information is gathered by 
Kubernetes from the node.


## Node Management

Unlike [Pods](pods.md) and [Services](services.md), a Node is not inherently
created by Kubernetes: it is either created from cloud providers like Google Compute Engine,
or from your physical or virtual machines. What this means is that when
Kubernetes creates a node, it only creates a representation for the node.
After creation, Kubernetes will check whether the node is valid or not.
For example, if you try to create a node from the following content:
```json
{
  "kind": "Node",
  "apiVersion": "v1",
  "metadata": {
    "name": "10.240.79.157",
    "labels": {
      "name": "my-first-k8s-node"
    }
  }
}
```

Kubernetes will create a Node object internally (the representation), and
validate the node by health checking based on the `metadata.name` field: we
assume `metadata.name` can be resolved. If the node is valid, i.e. all necessary
services are running, it is eligible to run a Pod; otherwise, it will be
ignored for any cluster activity, until it becomes valid. Note that Kubernetes
will keep invalid node unless explicitly deleted by client, and it will keep
checking to see if it becomes valid.

Currently, there are two agents that interacts with Kubernetes node interface:
Node Controller and Kube Admin.

### Node Controller

Node controller is a component in Kubernetes master which manages Node
objects. It performs two major functions: cluster-wide node synchronization
and single node life-cycle management.

Node controller has a sync loop that creates/deletes Nodes from Kubernetes
based on all matching VM instances listed from cloud provider. The sync period
can be controlled via flag `--node_sync_period`. If a new instance
gets created, Node Controller creates a representation for it. If an existing
instance gets deleted, Node Controller deletes the representation. Note however,
Node Controller is unable to provision the node for you, i.e. it won't install
any binary; therefore, to
join Kubernetes cluster, you as an admin need to make sure proper services are
running in the node. In the future, we plan to automatically provision some node
services. 

### Self-Registration of nodes

When kubelet flag `--register-node` is true (the default), then the kubelet will attempt to
register itself with the API server.  This is the preferred pattern, used by most distros.

For self-registration, the kubelet is started with the following options:
  - `--apiservers=` tells the kubelet the location of the apiserver.
  - `--kubeconfig` tells kubelet where to find credentials to authenticate itself to the apiserver.
  - `--cloud_provider=` tells the kubelet how to talk to a cloud provider to read metadata about itself.
  - `--register-node` tells the kubelet to create its own node resource.

Currently, any kubelet is authorized to create/modify any node resource, but in practice it only creates/modifies
its own.  (In the future, we plan to limit authorization to only allow a kubelet to modify its own Node resource.)

#### Manual Node Administration

A cluster administrator can create and modify Node objects.

If the administrator wishes to create node objects manually, set kubelet flag
`--register-node=false`.

The administrator can modify Node resources (regardless of the setting of `--register-node`).
Modifications include setting labels on the Node, and marking it unschedulable.

Labels on nodes can be used in conjunction with node selectors on pods to control scheduling.

Making a node unscheduleable will prevent new pods from being scheduled to that
node, but will not affect any existing pods on the node.  This is useful as a
preparatory step before a node reboot, etc.  For example, to mark a node
unschedulable, run this command:
```
kubectl replace nodes 10.1.2.3 --patch='{"apiVersion": "v1", "unschedulable": true}'
```

### Node capacity

The capacity of the node (number of cpus and amount of memory) is part of the node resource.
Normally, nodes register themselves and report their capacity when creating the node resource.  If
you are doing [manual node administration](#manual-node-administration), then you need to set node
capacity when adding a node.

The kubernetes scheduler ensures that there are enough resources for all the pods on a node.  It
checks that the sum of the limits of containers on the node is less than the node capacity.  It
includes all containers started by kubelet, but not containers started directly by docker, nor
processes not in containers.  

If you want to explicitly reserve resources for non-Pod processes, you can create a placeholder
pod.  Use the following template:
```
apiVersion: v1
kind: Pod
metadata:
  name: resource-reserver
spec:
  containers:
  - name: sleep-forever
    image: gcr.io/google_containers/pause:0.8.0
    resources:
      limits:
        cpu: 100m
        memory: 100Mi
```
Set the `cpu` and `memory` values to the amount of resources you want to reserve.
Place the file in the manifest directory (`--config=DIR` flag of kubelet).  Do this
on each kubelet where you want to reserve resources.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/node.md?pixel)]()
