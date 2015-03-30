# Node

## What is a node?

`Node` is a worker node in LMKTFY, previously known as `Minion`. Node
may be a VM or physical machine, depending on the cluster. Each node has
the services necessary to run [Pods](pods.md) and be managed from the master
systems. The services include docker, lmktfylet and network proxy. See
[The LMKTFY Node](../DESIGN.md#the-lmktfy-node) section in design
doc for more details.

## Node Status

Node status describes current status of a node. For now, there are three
pieces of information:

### HostIP

Host IP address is queried from cloudprovider and stored as part of node
status. If lmktfy runs without cloudprovider, node's ID will be used.
IP address can change, and there are different kind of IPs, e.g. public
IP, private IP, dynamic IP, ipv6, etc. It makes more sense to save it as
a status rather than spec.

### Node Phase

Node Phase is the current lifecycle phase of node, one of `Pending`,
`Running` and `Terminated`. Node Phase management is under development,
here is a brief overview: In lmktfy, node will be created in `Pending`
phase, until it is discovered and checked in by lmktfy, at which time,
lmktfy will mark it as `Running`. The end of a node's lifecycle is
`Terminated`. A terminated node will not receive any scheduling request,
and any running pods will be removed from the node.

Node with `Running` phase is necessary but not sufficient requirement for
scheduling Pods. For a node to be considered a scheduling candidate, it
must have appropriate conditions, see below.

### Node Condition
Node Condition describes the conditions of `Running` nodes. Current valid
conditions are `NodeReachable`, `NodeReady` and `NodeSchedulable`. In the
future, we plan to add more. `NodeReachable` means the node can be reached
within the cluster. `NodeReady` means the lmktfylet returns StatusOK for HTTP
health check. `NodeSchedulable` means node is allowed to schedule any new
pods and is controlled by 'unschedulable' field in node spec.
Different condition provides different level of understanding for node
health. LMKTFY will make a comprehensive scheduling decision based on the
information. Node condition is represented as a json object. For example, the
following conditions mean the node is reachable from its cluster, node is in
sane state but not allowed to accept new pods:
```json
"conditions": [
  {
    "kind": "Reachable",
    "status": "True",
  },
  {
    "kind": "Ready",
    "status": "True",
  },
  {
    "kind": "Schedulable",
    "status": "False",
  },
]
```

## Node Management

Unlike [Pod](pods.md) and [Service](services.md), `Node` is not inherently
created by LMKTFY: it is either created from cloud providers like GCE,
or from your physical or virtual machines. What this means is that when
LMKTFY creates a node, it only creates a representation for the node.
After creation, LMKTFY will check whether the node is valid or not.
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
    "name": "my-first-lmktfy-node",
  },
}
```

LMKTFY will create a `Node` object internally (the representation), and
validate the node by health checking based on the `id` field: we assume `id`
can be resolved. If the node is valid, i.e. all necessary services are running,
it is eligible to run a `Pod`; otherwise, it will be ignored for any cluster
activity, until it becomes valid. Note that LMKTFY will keep invalid node
unless explicitly deleted by client, and it will keep checking to see if it
becomes valid.

Currently, there are two agents that interacts with LMKTFY node interface:
Node Controller and LMKTFY Admin.

### Node Controller

Node controller is a component in LMKTFY master which manages `Node`
objects. It performs two major functions: cluster-wide node synchronization
and single node life-cycle management.

Node controller has a sync loop that creates/deletes `Node`s from LMKTFY
based on all matching VM instances listed from cloud provider. The sync period
can be controlled via flag "--node_sync_period". If a new instance
gets created, Node Controller creates a representation for it. If an existing
instance gets deleted, Node Controller deletes the representation. Note however,
Node Controller is unable to provision the node for you, i.e. it won't install
any binary; therefore, to
join LMKTFY cluster, you as an admin need to make sure proper services are
running in the node. In the future, we plan to automatically provision some node
services. In case of no cloud provider, Node Controller simply registers all
machines from `--machines` flag, any futher interactions need to be done manually
by using `lmktfyctl`. If you are paranoid, leave `--machines` empty and create all
machines from `lmktfyctl` one by one - the two approaches are equivalent.
Optionally you can skip cluster-wide node synchronization with
'--sync_nodes=false' and can use REST api/lmktfyctl cli to add/remove nodes.

Node life-cycle management in the Node Controller is still under development, it
is supposed to manage the Node Status Specification defined above.

### Manual Node Administration

A LMKTFY administrator typically uses `lmktfyctl` to manage `Node`. Similar
to Node Controller, `lmktfyctl` command only creates/deletes node representation.
Note if LMKTFY is running on cloud provider, `lmktfyctl create` a node will
be refused if Node Controller has already synchronized nodes from cloud provider.
Admin can choose to make the node unschedulable using `lmktfyctl`. Unscheduling the node
will not affect any existing pods on the node but it will disable creation of
any new pods on the node. Node unschedulable example:
```
lmktfyctl update nodes 10.1.2.3 --patch='{"apiVersion": "v1beta1", "unschedulable": true}'
```
