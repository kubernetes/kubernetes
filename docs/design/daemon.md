# DaemonSet in Kubernetes

**Author**: Ananya Kumar (@AnanyaKumar)

**Status**: Implemented.

This document presents the design of the Kubernetes DaemonSet, describes use
cases, and gives an overview of the code.

## Motivation

Many users have requested for a way to run a daemon on every node in a
Kubernetes cluster, or on a certain set of nodes in a cluster. This is essential
for use cases such as building a sharded datastore, or running a logger on every
node. In comes the DaemonSet, a way to conveniently create and manage
daemon-like workloads in Kubernetes.

## Use Cases

The DaemonSet can be used for user-specified system services, cluster-level
applications with strong node ties, and Kubernetes node services. Below are
example use cases in each category.

### User-Specified System Services:

Logging: Some users want a way to collect statistics about nodes in a cluster
and send those logs to an external database. For example, system administrators
might want to know if their machines are performing as expected, if they need to
add more machines to the cluster, or if they should switch cloud providers. The
DaemonSet can be used to run a data collection service (for example fluentd) on
every node and send the data to a service like ElasticSearch for analysis.

### Cluster-Level Applications

Datastore: Users might want to implement a sharded datastore in their cluster. A
few nodes in the cluster, labeled ‘app=datastore’, might be responsible for
storing data shards, and pods running on these nodes might serve data. This
architecture requires a way to bind pods to specific nodes, so it cannot be
achieved using a Replication Controller. A DaemonSet is a convenient way to
implement such a datastore.

For other uses, see the related [feature request](https://issues.k8s.io/1518)

## Functionality

The DaemonSet supports standard API features:
  - create
  - The spec for DaemonSets has a pod template field.
  - Using the pod’s nodeSelector field, DaemonSets can be restricted to operate
over nodes that have a certain label. For example, suppose that in a cluster
some nodes are labeled ‘app=database’. You can use a DaemonSet to launch a
datastore pod on exactly those nodes labeled ‘app=database’.
  - Using the pod's nodeName field, DaemonSets can be restricted to operate on a
specified node.
  - The PodTemplateSpec used by the DaemonSet is the same as the PodTemplateSpec
used by the Replication Controller.
  - The initial implementation will not guarantee that DaemonSet pods are
created on nodes before other pods.
  - The initial implementation of DaemonSet does not guarantee that DaemonSet
pods show up on nodes (for example because of resource limitations of the node),
but makes a best effort to launch DaemonSet pods (like Replication Controllers
do with pods). Subsequent revisions might ensure that DaemonSet pods show up on
nodes, preempting other pods if necessary.
  - The DaemonSet controller adds an annotation:
```"kubernetes.io/created-by: \<json API object reference\>"```
  - YAML example:

    ```YAML
  apiVersion: extensions/v1beta1
  kind: DaemonSet
  metadata:
    labels:
      app: datastore
    name: datastore
  spec:
    template:
      metadata:
        labels:
          app: datastore-shard
      spec:
        nodeSelector: 
          app: datastore-node
        containers:
          name: datastore-shard
          image: kubernetes/sharded
          ports:
            - containerPort: 9042
              name: main
```

  - commands that get info:
    - get (e.g. kubectl get daemonsets)
    - describe
  - Modifiers:
    - delete (if --cascade=true, then first the client turns down all the pods
controlled by the DaemonSet (by setting the nodeSelector to a uuid pair that is
unlikely to be set on any node); then it deletes the DaemonSet; then it deletes
the pods)
    - label
    - annotate
    - update operations like patch and replace (only allowed to selector and to
nodeSelector and nodeName of pod template)
    - DaemonSets have labels, so you could, for example, list all DaemonSets
with certain labels (the same way you would for a Replication Controller).

In general, for all the supported features like get, describe, update, etc,
the DaemonSet works in a similar way to the Replication Controller. However,
note that the DaemonSet and the Replication Controller are different constructs.

### Persisting Pods

  - Ordinary liveness probes specified in the pod template work to keep pods
created by a DaemonSet running.
  - If a daemon pod is killed or stopped, the DaemonSet will create a new
replica of the daemon pod on the node.

### Cluster Mutations

  - When a new node is added to the cluster, the DaemonSet controller starts
daemon pods on the node for DaemonSets whose pod template nodeSelectors match
the node’s labels.
  - Suppose the user launches a DaemonSet that runs a logging daemon on all
nodes labeled “logger=fluentd”. If the user then adds the “logger=fluentd” label
to a node (that did not initially have the label), the logging daemon will
launch on the node. Additionally, if a user removes the label from a node, the
logging daemon on that node will be killed.

## Alternatives Considered

We considered several alternatives, that were deemed inferior to the approach of
creating a new DaemonSet abstraction.

One alternative is to include the daemon in the machine image. In this case it
would run outside of Kubernetes proper, and thus not be monitored, health
checked, usable as a service endpoint, easily upgradable, etc.

A related alternative is to package daemons as static pods. This would address
most of the problems described above, but they would still not be easily
upgradable, and more generally could not be managed through the API server
interface.

A third alternative is to generalize the Replication Controller. We would do
something like: if you set the `replicas` field of the ReplicationControllerSpec
to -1, then it means "run exactly one replica on every node matching the
nodeSelector in the pod template." The ReplicationController would pretend
`replicas` had been set to some large number -- larger than the largest number
of nodes ever expected in the cluster -- and would use some anti-affinity
mechanism to ensure that no more than one Pod from the ReplicationController
runs on any given node. There are two downsides to this approach. First,
there would always be a large number of Pending pods in the scheduler (these
will be scheduled onto new machines when they are added to the cluster). The
second downside is more philosophical: DaemonSet and the Replication Controller
are very different concepts. We believe that having small, targeted controllers
for distinct purposes makes Kubernetes easier to understand and use, compared to
having larger multi-functional controllers (see
["Convert ReplicationController to a plugin"](http://issues.k8s.io/3058) for
some discussion of this topic).

## Design

#### Client

- Add support for DaemonSet commands to kubectl and the client. Client code was
added to pkg/client/unversioned. The main files in Kubectl that were modified are
pkg/kubectl/describe.go and pkg/kubectl/stop.go, since for other calls like Get, Create,
and Update, the client simply forwards the request to the backend via the REST
API.

#### Apiserver

- Accept, parse, validate client commands
- REST API calls are handled in pkg/registry/daemonset
  - In particular, the api server will add the object to etcd
  - DaemonManager listens for updates to etcd (using Framework.informer)
- API objects for DaemonSet were created in expapi/v1/types.go and
expapi/v1/register.go
- Validation code is in expapi/validation

#### Daemon Manager

- Creates new DaemonSets when requested. Launches the corresponding daemon pod
on all nodes with labels matching the new DaemonSet’s selector.
- Listens for addition of new nodes to the cluster, by setting up a
framework.NewInformer that watches for the creation of Node API objects. When a
new node is added, the daemon manager will loop through each DaemonSet. If the
label of the node matches the selector of the DaemonSet, then the daemon manager
will create the corresponding daemon pod in the new node.
- The daemon manager creates a pod on a node by sending a command to the API
server, requesting for a pod to be bound to the node (the node will be specified
via its hostname.)

#### Kubelet

- Does not need to be modified, but health checking will occur for the daemon
pods and revive the pods if they are killed (we set the pod restartPolicy to
Always). We reject DaemonSet objects with pod templates that don’t have
restartPolicy set to Always.

## Open Issues

- Should work similarly to [Deployment](http://issues.k8s.io/1743).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/daemon.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
