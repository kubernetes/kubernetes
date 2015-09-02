# Daemon Controller in Kubernetes

**Author**: Ananya Kumar (@AnanyaKumar)

**Status**: Draft proposal; prototype in progress.

This document presents the design of the Kubernetes daemon controller, describes use cases, and gives an overview of the code.

## Motivation

Many users have requested for a way to run a daemon on every node in a Kubernetes cluster, or on a certain set of nodes in a cluster. This is essential for use cases such as building a sharded datastore, or running a logger on every node. In comes the daemon controller, a way to conveniently create and manage daemon-like workloads in Kubernetes.  

## Use Cases

The daemon controller can be used for user-specified system services, cluster level applications with strong node ties, and Kubernetes node services. Below are example use cases in each category.

### User-Specified System Services:
Logging: Some users want a way to collect statistics about nodes in a cluster and send those logs to an external database. For example, system administrators might want to know if their machines are performing as expected, if they need to add more machines to the cluster, or if they should switch cloud providers. The daemon controller can be used to run a data collection service (for example fluentd) and send the data to a service like ElasticSearch for analysis.

### Cluster-Level Applications
Datastore: Users might want to implement a sharded datastore in their cluster. A few nodes in the cluster, labeled ‘datastore’, might be responsible for storing data shards, and pods running on these nodes might serve data. This architecture requires a way to bind pods to specific nodes, so it cannot be achieved using a Replication Controller. A daemon controller is a convenient way to implement such a datastore.

For other uses, see the related [feature request](https://github.com/GoogleCloudPlatform/kubernetes/issues/1518)

## Functionality

The Daemon Controller supports standard API features:
- create
  - The spec for daemon controllers has a pod template field.
  - Using the pod’s node selector field, Daemon controllers can be restricted to operate over nodes that have a certain label. For example, suppose that in a cluster some nodes are labeled ‘database’. You can use a daemon controller to launch a datastore pod on exactly those nodes labeled ‘database’.
  - Using the pod's node name field, Daemon controllers can be restricted to operate on a specified node.
  - The spec for pod templates that run with the Daemon Controller is the same as the spec for pod templates that run with the Replication Controller, except there will not be a ‘replicas’ field (exactly 1 daemon pod will be launched per node).
  - We will not guarantee that daemon pods show up on nodes before regular pods - run ordering is out of scope for this controller. 
  - The initial implementation of Daemon Controller does not guarantee that Daemon pods show up on nodes (for example because of resource limitations of the node), but makes a best effort to launch Daemon pods (like Replication Controllers do with pods). Subsequent revisions might ensure that Daemon pods show up on nodes, pushing out other pods if necessary.
  - A daemon controller named “foo” adds a “controller: foo” annotation to all the pods that it creates
  - YAML example:
```YAML
  apiVersion: v1
  kind: Daemon
  metadata:
    labels:
      name: datastore
    name: datastore
  spec:
    template:
      metadata:
        labels:
          name: datastore-shard
      spec:
        node-selector: 
          name: datastore-node
        containers:
          name: datastore-shard
          image: kubernetes/sharded
          ports:
            - containerPort: 9042
              name: main
```
  - commands that get info
    - get (e.g. kubectl get dc)
    - describe
  - Modifiers
    - delete
    - stop: first we turn down all the pods controller by the daemon (by setting the nodeName to a non-existed name). Then we turn down the daemon controller.
    - label
    - update
    - Daemon controllers have labels, so you could, for example, list all daemon controllers with a certain label (the same way you would for a Replication Controller).
  - In general, for all the supported features like get, describe, update, etc, the Daemon Controller works in a similar way to the Replication Controller. However, note that the Daemon Controller and the Replication Controller are different constructs.

### Persisting Pods
  - Ordinary health checks specified in the pod template work to keep pods created by a Daemon Controller running.
  - If a daemon pod is killed or stopped, the daemon controller will create a new replica of the daemon pod on the node.

### Cluster Mutations
  - When a new node is added to the cluster the daemon controller starts the daemon on the node (if the node’s labels match the user-specified selectors). This is a big advantage of the Daemon Controller compared to alternative ways of launching daemons and configuring clusters.
  - Suppose the user launches a daemon controller that runs a logging daemon on all nodes labeled “tolog”. If the user then adds the “tolog” label to a node (that did not initially have the “tolog” label), the logging daemon will launch on the node. Additionally, if a user removes the “tolog” label from a node, the logging daemon on that node will be killed.

## Alternatives Considered

An alternative way to launch daemons is to avoid going through the API server, and instead provide ways to package the daemon into the node. For example, users could:

1. Include the daemon in the machine image
2. Use config files to launch daemons
3. Use static pod manifests to launch daemon pods when the node initializes

These alternatives don’t work as well because the daemons won’t be well integrated into Kubernetes. In particular,

1. In alternatives (1) and (2), health checking for the daemons would need to be re-implemented, or would not exist at all (because the daemons are not run inside pods). In the current proposal, the Kubelet will health-check daemon pods and restart them if necessary.
2. In alternatives (1) and (2), binding services to a group of daemons is difficult (which is needed in use cases such as the sharded data store use case described above), because the daemons are not run inside pods
3. A big disadvantage of these methods is that adding new daemons in existing nodes is difficult (for example, if a cluster manager wants to add a logging daemon after a cluster has been deployed).
4. The above alternatives are less user-friendly. Users need to learn two ways of launching pods: using the API when launching pods associated with Replication Controllers, and using manifests when launching daemons. So in the alternatives, deployment is more difficult.
5. It’s difficult to upgrade binaries launched in any of those three ways.

Another alternative is for the user to explicitly assign pods to specific nodes (using the Pod spec) when creating pods. A big disadvantage of this alternative is that the user would need to manually check whether new nodes satisfy the desired labels, and if so add the daemon to the node. This makes deployment painful, and could lead to costly mistakes (if a certain daemon is not launched on a new node which it is supposed to run on). In essence, every user will be re-implementing the Daemon Controller for themselves.

A third alternative is to generalize the Replication Controller. We could add a field for the user to specify that she wishes to bind pods to certain nodes in the cluster. Or we could add a field to the pod-spec allowing the user to specify that each node can have exactly one instance of a pod (so the user would create a Replication Controller with a very large number of replicas, and set the anti-affinity field to true preventing more than one pod with that label from being scheduled onto a single node). The disadvantage of these methods is that the Daemon Controller and the Replication Controller are very different concepts. The Daemon Controller operates on a per-node basis, while the Replication Controller operates on a per-job basis (in particular, the Daemon Controller will take action when a node is changed or added). So presenting them as different concepts makes for a better user interface. Having small and directed controllers for distinct purposes makes Kubernetes easier to understand and use, compared to having one controller to rule them all.

## Design

#### Client
- Add support for daemon controller commands to kubectl and the client. Client code was added to client/unversioned. The main files in Kubectl that were modified are kubectl/describe.go and kubectl/stop.go, since for other calls like Get, Create, and Update, the client simply forwards the request to the backend via the REST API.

#### Apiserver
- Accept, parse, validate client commands
- REST API calls are handled in registry/daemon
  - In particular, the api server will add the object to etcd
  - DaemonManager listens for updates to etcd (using Framework.informer)
- API objects for Daemon Controller were created in expapi/v1/types.go and expapi/v1/register.go
- Validation code is in expapi/validation

#### Daemon Manager
- Creates new daemon controllers when requested. Launches the corresponding daemon pod on all nodes with labels matching the new daemon controller’s selector.
- Listens for addition of new nodes to the cluster, by setting up a framework.NewInformer that watches for the creation of Node API objects. When a new node is added, the daemon manager will loop through each daemon controller. If the label of the node matches the selector of the daemon controller, then the daemon manager will create the corresponding daemon pod in the new node.
- The daemon manager creates a pod on a node by sending a command to the API server, requesting for a pod to be bound to the node (the node will be specified via its hostname)

#### Kubelet
- Does not need to be modified, but health checking will occur for the daemon pods and revive the pods if they are killed (we set the pod restartPolicy to Always). We reject Daemon Controller objects with pod templates that don’t have restartPolicy set to Always.

## Testing

Unit Tests:
Each component was unit tested, fakes were implemented when necessary. For example, when testing the client, a fake API server was used.

End to End Tests:
One end-to-end test was implemented. The end-to-end test verified that the daemon manager runs the daemon on every node, that when a daemon pod is stopped it restarts, that the daemon controller can be reaped (stopped), and that the daemon adds/removes daemon pods appropriately from nodes when their labels change.

## Open Issues
- See how this can work with [Deployment design](https://github.com/GoogleCloudPlatform/kubernetes/issues/1743).
