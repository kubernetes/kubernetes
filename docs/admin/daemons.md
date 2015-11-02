<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Daemon Sets

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Daemon Sets](#daemon-sets)
  - [What is a _Daemon Set_?](#what-is-a-daemon-set)
  - [Writing a DaemonSet Spec](#writing-a-daemonset-spec)
    - [Required Fields](#required-fields)
    - [Pod Template](#pod-template)
    - [Pod Selector](#pod-selector)
    - [Running Pods on Only Some Nodes](#running-pods-on-only-some-nodes)
  - [How Daemon Pods are Scheduled](#how-daemon-pods-are-scheduled)
  - [Communicating with DaemonSet Pods](#communicating-with-daemonset-pods)
  - [Updating a DaemonSet](#updating-a-daemonset)
  - [Alternatives to Daemon Set](#alternatives-to-daemon-set)
    - [Init Scripts](#init-scripts)
    - [Bare Pods](#bare-pods)
    - [Static Pods](#static-pods)
    - [Replication Controller](#replication-controller)
  - [Caveats](#caveats)

<!-- END MUNGE: GENERATED_TOC -->

## What is a _Daemon Set_?

A _Daemon Set_ ensures that all (or some) nodes run a copy of a pod.  As nodes are added to the
cluster, pods are added to them.  As nodes are removed from the cluster, those pods are garbage
collected.  Deleting a Daemon Set will clean up the pods it created.

Some typical uses of a Daemon Set are:

- running a cluster storage daemon, such as `glusterd`, `ceph`, on each node.
- running a logs collection daemon on every node, such as `fluentd` or `logstash`.
- running a node monitoring daemon on every node, such as [Prometheus Node Exporter](
  https://github.com/prometheus/node_exporter), `collectd`, New Relic agent, or Ganglia `gmond`.

In a simple case, one Daemon Set, covering all nodes, would be used for each type of daemon.
A more complex setup might use multiple DaemonSets would be used for a single type of daemon,
but with different flags and/or different memory and cpu requests for different hardware types.

## Writing a DaemonSet Spec

### Required Fields

As with all other Kubernetes config, a DaemonSet needs `apiVersion`, `kind`, and `metadata` fields.  For
general information about working with config files, see [here](../user-guide/simple-yaml.md),
[here](../user-guide/configuring-containers.md), and [here](../user-guide/working-with-resources.md).

A DaemonSet also needs a [`.spec`](../devel/api-conventions.md#spec-and-status) section.

### Pod Template

The `.spec.template` is the only required field of the `.spec`.

The `.spec.template` is a [pod template](../user-guide/replication-controller.md#pod-template).
It has exactly the same schema as a [pod](../user-guide/pods.md), except
it is nested and does not have an `apiVersion` or `kind`.

In addition to required fields for a pod, a pod template in a DaemonSet has to specify appropriate
labels (see [pod selector](#pod-selector)).

A pod template in a DaemonSet must have a [`RestartPolicy`](../user-guide/pod-states.md)
 equal to `Always`, or be unspecified, which defaults to `Always`.

### Pod Selector

The `.spec.selector` field is a pod selector.  It works the same as the `.spec.selector` of
a [ReplicationController](../user-guide/replication-controller.md) or
[Job](../user-guide/jobs.md).

If the `.spec.selector` is specified, it must equal the `.spec.template.metadata.labels`.  If not
specified, the are default to be equal.  Config with these unequal will be rejected by the API.

Also you should not normally create any pods whose labels match this selector, either directly, via
another DaemonSet, or via other controller such as ReplicationController.  Otherwise, the DaemonSet
controller will think that those pods were created by it.  Kubernetes will not stop you from doing
this.  Once case where you might want to do this is manually create a pod with a different value on
a node for testing.

### Running Pods on Only Some Nodes

If you specify a `.spec.template.spec.nodeSelector`, then the DaemonSet controller will
create pods on nodes which match that [node
selector](../user-guide/node-selection/README.md).

If you do not specify a `.spec.template.spec.nodeSelector`, then the DaemonSet controller will
create pods on all nodes.

## How Daemon Pods are Scheduled

Normally, the machine that a pod runs on is selected by the Kubernetes scheduler.  However, pods
created by the Daemon controller have the machine already selected (`.spec.nodeName` is specified
when the pod is created, so it is ignored by the scheduler).  Therefore:

 - the [`unschedulable`](node.md#manual-node-administration) field of a node is not respected
   by the daemon set controller.
 - daemon set controller can make pods even when the scheduler has not been started, which can help cluster
   bootstrap.

## Communicating with DaemonSet Pods

Some possible patterns for communicating with pods in a DaemonSet are:

- **Push**: Pods in the Daemon Set are configured to send updates to another service, such
  as a stats database.  They do not have clients.
- **NodeIP and Known Port**: Pods in the Daemon Set use a `hostPort`, so that the pods are reachable
  via the node IPs.  Clients knows the the list of nodes ips somehow, and know the port by convention.
- **DNS**: Create a [headless service](../user-guide/services.md#headless-services) with the same pod selector,
  and then discover DaemonSets using the `endpoints` resource or retrieve multiple A records from
  DNS.
- **Service**: Create a service with the same pod selector, and use the service to reach a
  daemon on a random node. (No way to reach specific node.)

## Updating a DaemonSet

If node labels are changed, the DaemonSet will promptly add pods to newly matching nodes and delete
pods from newly not-matching nodes.

You can modify the pods that a DaemonSet creates.  However, pods do not allow all
fields to be updated.  Also, the DeamonSet controller will use the original template the next
time a node (even with the same name) is created.


You can delete a DeamonSet.  If you specify `--cascade=false` with `kubectl`, then the pods
will be left on the nodes.  You can then create a new DaemonSet with a different template.
the new DaemonSet with the different template will recognize all the existing pods as having
matching labels.  It will not modify or delete them despite a mismatch in the pod template.
You will need to force new pod creation by deleting the pod or deleting the node.

You cannot update a DaemonSet.

Support for updating DaemonSets and controlled updating of nodes is planned.

## Alternatives to Daemon Set

### Init Scripts

It is certainly possible to run daemon processes by directly starting them on a node (e.g using
`init`, `upstartd`, or `systemd`).  This is perfectly fine.  However, there are several advantages to
running such processes via a DaemonSet:

- Ability to monitor and manage logs for daemons in the same way as applications.
- Same config language and tools (e.g. pod templates, `kubectl`) for daemons and applications.
- Future versions of Kubernetes will likely support integration between DaemonSet-created
  pods and node upgrade workflows.
- Running daemons in containers with resource limits increases isolation between daemons from app
  containers.  However, this can also be accomplished by running the daemons in a container but not in a pod
  (e.g. start directly via Docker).

### Bare Pods

It is possible to create pods directly which specify a particular node to run on.  However,
a Daemon Set replaces pods that are deleted or terminated for any reason, such as in the case of
node failure or disruptive node maintenance, such as a kernel upgrade. For this reason, you should
use a Daemon Set rather than creating individual pods.

### Static Pods

It is possible to create pods by writing a file to a certain directory watched by Kubelet.  These
are called [static pods](static-pods.md).
Unlike DaemonSet, static pods cannot be managed with kubectl
or other Kubernetes API clients.  Static pods do not depend on the apiserver, making them useful
in cluster bootstrapping cases.  Also, static pods may be deprecated in the future.

### Replication Controller

Daemon Set are similar to [Replication Controllers](../user-guide/replication-controller.md) in that
they both create pods, and those pods have processes which are not expected to terminate (e.g. web servers,
storage servers).

Use a replication controller for stateless services, like frontends, where scaling up and down the
number of replicas and rolling out updates are more important than controlling exactly which host
the pod runs on.  Use a Daemon Controller when it is important that a copy of a pod always run on
all or certain hosts, and when it needs to start before other pods.

## Caveats

DaemonSet objects are in the [`extensions` API Group](../api.md#api-groups).
DaemonSet is not enabled by default. Enable it by setting
`--runtime-config=extensions/v1beta1/daemonsets=true` on the api server. This can be
achieved by exporting ENABLE_DAEMONSETS=true before running kube-up.sh script
on GCE.

DaemonSet objects effectively have [API version `v1alpha1`](../api.md#api-versioning).
 Alpha objects may change or even be discontinued in future software releases.
However, due to to a known issue, they will appear as API version `v1beta1` if enabled.





<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/daemons.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
