# Node

## What is a node?

`Node` is a worker node in Kubernetes, previously known as `Minion`. Node
may be a VM or physical machine, depending on the cluster. Each node has
the services necessary to run [Pods](pods.md) and be managed from the master
systems. The services include docker, kubelet and network proxy. See
[The Kubernetes Node](../DESIGN.md#the-kubernetes-node) section in design
doc for more details.

## Node Status

Node Status is still under development. See:
[#2164](https://github.com/GoogleCloudPlatform/kubernetes/issues/2164),
[#2315](https://github.com/GoogleCloudPlatform/kubernetes/pull/2315).

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
machines from `-machines` flag, any futher interactions need to be done manually
by using `kubectl`. If you are paranoid, leave `-machines` empty and create all
machines from `kubectl` one by one - the two approaches are equivalent.

Node life-cycle management in the Node Controller is still under development, it
is supposed to manage the Node Status Specification defined above.

### Manual Node Administration

A Kubernetes administrator typically uses `kubectl` to manage `Node`. Similar
to Node Controller, `kubectl` command only creates/deletes node representation.
Note if Kubernetes is running on cloud provider, `kubectl create` a node will
be refused if Node Controller has already synchronized nodes from cloud provider.
