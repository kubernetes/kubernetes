# Daemon Controller in Kubernetes

**Author**: Ananya Kumar (@AnanyaKumar)

**Status**: Draft proposal; prototype in progress.

This document presents the design of a daemon controller for Kubernetes, outlines relevant Kubernetes concepts, describes use cases, and lays out milestones for its development.

## Motivation

In Kubernetes, a Replication Controller ensures that the specified number of a specified pod are running in the cluster at all times (pods are restarted if they are killed). With the Replication Controller, users cannot control which nodes their pods run on - Kubernetes decides how to schedule the pods onto nodes. However, many users want control over how certain pods are scheduled. In particular, many users have requested for a way to run a daemon on every node in the cluster, or on a certain set of nodes in the cluster. This is essential for use cases such as building a sharded datastore, or running a logger on every node. In comes the daemon controller, a way to conveniently create and manage daemon-like workloads in Kubernetes.  

## Use Cases

The daemon controller can be used for user-specified system services, cluster level applications with strong node ties, and Kubernetes node services. Below are example use cases in each category.

### User-Specified System Services:
Logging: Some users want a way to collect statistics about nodes in a cluster and send those logs to an external database. For example, system administrators might want to know if their machines are performing as expected, if they need to add more machines to the cluster, or if they should switch cloud providers. The daemon controller can be used to run a data collection service (for example fluentd) and send the data to a service like ElasticSearch for analysis.

### Cluster-Level Applications
Datastore: Users might want to implement a sharded datastore in their cluster. A few nodes in the cluster, labeled ‘datastore’, might be responsible for storing data shards, and pods running on these nodes might serve data. This architecture requires a way to bind pods to specific nodes, so it cannot be achieved using a Replication Controller. A daemon controller is a convenient way to implement such a datastore.

For other uses, see the related [feature request](https://github.com/GoogleCloudPlatform/kubernetes/issues/1518)

## Functionality

The Daemon Controller will support standard API features:
- create
  - The spec for daemon controllers will have a pod template field.  Note that daemon controllers will not have a nodename field.
  - Using the pod’s node selector field, Daemon controllers can be restricted to operate over nodes that have a certain label. For example, suppose that in a cluster some nodes are labeled ‘database’. You can use a daemon controller to launch a datastore pod on exactly those nodes labeled ‘database’.
  - The spec for pod templates that run with the Daemon Controller is the same as the spec for pod templates that run with the Replication Controller, except there will not be a ‘replicas’ field (exactly 1 daemon will be launched per node).
  - We will not guarantee that daemon pods show up on nodes before regular pods - run ordering is out of scope for this controller
  - The Daemon Controller will not guarantee that Daemon pods show up on nodes (for example because of resource limitations of the node), but will make a best effort to launch Daemon pods (like Replication Controllers do with pods)
  - A daemon controller named “foo” will add a “controller: foo” annotation to all the pods that it creates
  - YAML example:
```YAML
  apiVersion: v1
  kind: DaemonController
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
    - stop: first we turn down the Daemon Controller foo, and then we turn down all pods matching the query “controller: foo”
    - label
    - update
    - extension: rolling-update
    - Daemon controllers will have labels, so you could, for example, list all daemon controllers with a certain label (the same way you would for a Replication Controller).
  - In general, for all the supported features like get, describe, update, etc, the Daemon Controller will work in a similar way to the Replication Controller. However, note that the Daemon Controller and the Replication Controller are different constructs.

### Health checks
  - Ordinary health checks specified in the pod template will of course work to keep pods created by a Daemon Controller running.

### Cluster Mutations
  - When a new node is added to the cluster the daemon controller should start the daemon on the node (if the node’s labels match the user-specified selectors). This is a big advantage of the Daemon Controller compared to alternative ways of launching daemons and configuring clusters.
  - Suppose the user launches a daemon controller that runs a logging daemon on all nodes labeled “tolog”. If the user then adds the “tolog” label to a node (that did not initially have the “tolog” label), the logging daemon should be launched on the node. Additionally, if a user removes the “tolog” label from a node, the logging daemon on that node should be killed.

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
- Add support for daemon controller commands to kubectl. The main files that need to be modified are kubectl/cmd/describe.go and kubectl/cmd/rolling_updater.go, since for other calls like Get, Create, and Update, the client simply forwards the request to the backend via the REST API.
#### Apiserver
- Accept, parse, validate client commands
- REST API calls will be handled in registry/daemoncontroller
  - In particular, the api server will add the object to etcd
  - DaemonManager listens for updates to etcd (using Framework.informer)
- API object for Daemon Controller will be created in api/v1/types.go and api/v1/register.go
#### DaemonManager
- Creates new daemon controllers when requested. Launches the corresponding daemon pod on all nodes with labels matching the new daemon controller’s selector.
- Listens for addition of new nodes to the cluster, by setting up a framework.NewInformer that watches for the creation of Node API objects. When a new node is added, the daemon manager will loop through each daemon controller. If the label of the node matches the selector of the daemon controller, then the daemon manager will create the corresponding daemon pod in the new node.
- The daemon manager will create a pod on a node by sending a command to the API server, requesting for a pod to be bound to the node (the node will be specified via its hostname)
#### Kubelet
- Does not need to be modified, but health checking for the daemon pods and revive the pods if they are killed (we will set the pod restartPolicy to Always). We reject Daemon Controller objects with pod templates that don’t have restartPolicy set to Always.

## Milestones

1. Implement REST API skeleton code - users can add, update, delete (and perform other queries) on the Daemon Controller via the REST interface, but the API server won’t perform any action and will simply return a JSON object indicating that the query has been received.
  1. Add code to API server that describes the type and specs of a daemon controller
  2. Add code to validate that a daemon controller satisfies the specs
2. Add Kubectl support. Kubectl queries should be handled and appropriately forwarded to the API server. Also “kubectl get dc <dc name>” needs to be implemented.
3. User should be able to create daemon pods that are launched on every node in the cluster by calling the REST API or Kubectl.
  1. Add code in the daemon manager that creates daemon controller objects, loops through every node in the cluster, and adds the specified pod to the node (via a call to API server)
  2. New nodes added to the cluster will run daemon pods. This involves adding code to daemon manager that listens for node additions, and adds all daemons to the new node
  3. Add support for labels (for launching daemon pods in exactly those nodes with a certain label).
    1. Add an extra field to the spec of the Daemon controller that allows users to specify a node selector. Modify the code that validates the daemon controller spec accordingly.
    2. DaemonManager should check that a node’s label satisfies the daemon controller’s node selector before running a daemon on the node.
    3. DaemonManager should check that a newly added node’s label satisfies the daemon controller’s node selector before running a daemon on the node.
  4. When a node gets a new label, daemon controllers with selectors corresponding to the new label should run their daemon pods on the node. This involves adding code in the Daemon Manager that listens for changes in a node’s label, loops through the Daemon Controllers, and notifies the controllers if they need to take action.
4. Add support for other methods (describe, stop, update, rolling-update, etc)
5. Write documentation for Daemon Controller, and integrate the Daemon Controller into existing documentation articles.

## Testing

Testing policy: A milestone is only considered complete if it has been satisfactorily tested (as determined by code reviewers).

Unit Tests:
Each component will be unit tested, fakes will be implemented when necessary. For example, when testing the client, a fake API server will be used.

Integration Tests:
<WIP>

End to End Tests:
At least one end-to-end test will be implemented. The end-to-end test(s) should test that the daemon controller runs the daemon on every node, that the daemon is added to new nodes joining the cluster, that when a daemon is killed it restarts, and that the daemon controller supports standard features like ‘stop’, ‘describe’, etc. There are some e2e tests that could use a Daemon Controller (e.g. the network test). We will switch over to using the Daemon Controller for these tests.

## Open Issues
- Remove controller from name, Just use Daemon.
- Consider using [experimental API prefix](https://github.com/GoogleCloudPlatform/kubernetes/issues/10009)
- Rolling updates across nodes should be performed according to the [anti-affinity policy in scheduler](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/plugin/pkg/scheduler/api/v1/types.go). We need to figure out how to share that configuration.
- See how this can work with [Deployment design](https://github.com/GoogleCloudPlatform/kubernetes/issues/1743).


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/daemon-controller.md?pixel)]()
