# Node

## What is a node?

`Node` is a worker node in Kubernetes, previously known as `Minion`. Node
may be a VM or physical machine, depending on the cluster. Each node has
the services necessary to run [Pods](pods.md) and be managed from the master
systems. The services include docker, kubelet and network proxy. See
[The Kubernetes Node](../DESIGN.md#the-kubernetes-node) section in design
doc for more details.

## Node Status

Node status describes current status of a node. For now, there are three
pieces of information:

### HostIP

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
Node Condition describes the conditions of `Running` nodes. Current valid
condition is `NodeReady`. In the future, we plan to add more. 
`NodeReady` means kubelet is healthy and ready to accept pods. Different 
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

## Node Management

Unlike [Pod](pods.md) and [Service](services.md), `Node` is not inherently
created by Kubernetes: it is either created from cloud providers like GCE,
or from your physical or virtual machines. What this means is that when
Kubernetes creates a node, it only creates a representation for the node.
After creation, Kubernetes will check whether the node is valid or not.
For example, if you try to create a node from the following content:
```json
{
  "id": "10.1.2.3",
  "kind": "Minion",
  "apiVersion": "v1beta1",
  "resources": {
    "capacity": {
      "cpu": 1000,
      "memory": 1073741824
    },
  },
  "labels": {
    "name": "my-first-k8s-node",
  },
}
```

Kubernetes will create a `Node` object internally (the representation), and
validate the node by health checking based on the `id` field: we assume `id`
can be resolved. If the node is valid, i.e. all necessary services are running,
it is eligible to run a `Pod`; otherwise, it will be ignored for any cluster
activity, until it becomes valid. Note that Kubernetes will keep invalid node
unless explicitly deleted by client, and it will keep checking to see if it
becomes valid.

Currently, there are two agents that interacts with Kubernetes node interface:
Node Controller and Kube Admin.

### Node Controller

Node controller is a component in Kubernetes master which manages `Node`
objects. It performs two major functions: cluster-wide node synchronization
and single node life-cycle management.

Node controller has a sync loop that creates/deletes `Node`s from Kubernetes
based on all matching VM instances listed from cloud provider. The sync period
can be controlled via flag "--node_sync_period". If a new instance
gets created, Node Controller creates a representation for it. If an existing
instance gets deleted, Node Controller deletes the representation. Note however,
Node Controller is unable to provision the node for you, i.e. it won't install
any binary; therefore, to
join Kubernetes cluster, you as an admin need to make sure proper services are
running in the node. In the future, we plan to automatically provision some node
services. In case of no cloud provider, Node Controller simply registers all
machines from `--machines` flag, any further interactions need to be done manually
by using `kubectl`. If you are paranoid, leave `--machines` empty and create all
machines from `kubectl` one by one - the two approaches are equivalent.
Optionally you can skip cluster-wide node synchronization with
'--sync_nodes=false' and can use REST api/kubectl cli to add/remove nodes.

Node life-cycle management in the Node Controller is still under development, it
is supposed to manage the Node Status Specification defined above.

### Manual Node Administration

A Kubernetes administrator typically uses `kubectl` to manage `Node`. Similar
to Node Controller, `kubectl` command only creates/deletes node representation.
Note if Kubernetes is running on cloud provider, `kubectl create` a node will
be refused if Node Controller has already synchronized nodes from cloud provider.
Admin can choose to make the node unschedulable using `kubectl`. Unscheduling the node
will not affect any existing pods on the node but it will disable creation of
any new pods on the node. Node unschedulable example:
```
kubectl update nodes 10.1.2.3 --patch='{"apiVersion": "v1beta1", "unschedulable": true}'
```
