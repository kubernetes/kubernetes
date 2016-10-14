<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.4/docs/proposals/pod-termination.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Pod Safety, Consistency Guarantees, and Storage Implicitions

@smarterclayton @bprashant

October 2016

## Proposal and Motivation

A pod represents the finite execution of one or more related processes on the
cluster. In order to ensure higher level consistent controllers can safely
build on top of pods, the exact guarantees around its lifecycle on the cluster
must be clarified, and it must be possible for higher order controllers
and application authors to correctly reason about the lifetime of those
processes and their access to cluster resources in a distributed computing
environment.

To run most clustered software on Kubernetes, it must be possible to guarantee
**at most once** execution of a particular pet pod at any time on the cluster.
This allows the controller to prevent multiple processes having access to
shared cluster resources believing they are the same entity. When a node
containing a pet is partitioned, the Pet Set must remain consistent (no new
entity will be spawned) but may become unavailable (cluster no longer has
a sufficient number of members). The Pet Set guarante must be strong enough
for an administrator to reason about the state of the cluster by observing
the Kubrenetes API.

In order to reconcile partitions, an actor (human or automated) must decide
when the partition is unrecoverable. The actor may be informed of the failure
in an unambiguous way (e.g. the node was destroyed by a meteor) allowing for
certainty that the processes on that node are terminated, and thus may
resolve the partition by deleting the node and the pods on the node.
Alternatively, the actor may take steps to ensure the partitioned node
cannot return to the cluster or access shared resources - this is known
as **fencing** and is a well understood domain.

This proposal covers the changes necessary to ensure:

* Pet Sets can ensure **at most one** semantics for each individual pet
* Other system components such as the node and namespace controller can
  safely perform their responsibilities without violating that guarantee
* An administrator or higher level controller can signal that a node
  partition is permanent, allowing the Pet Set controller to proceed.
* A fencing controller can take corrective action automatically to heal
  partitions

We will accomplish this by:

* Clarifying which components are allowed to force delete pods (as opposed
  to merely requesting termination)
* Ensuring system components can observe partitioned pods and nodes
  correctly
* Defining how a fencing controller could safely interoperate with
  partitioned nodes and pods to safely heal partitions
* Describing how shared storage components without innate safety
  guarantees can be safely shared on the cluster.


### Current Guarantees for Pod lifecycle

The existing pod model provides the following guarantees:

* A pod is executed on exactly one node
* A pod has the following lifecycle phases:
  * Creation
  * Scheduling
  * Execution
    * Init containers
    * Application containers
  * Termination
  * Deletion
* A pod can only move through its phases in order, and may not return
  to an earlier phase.
* A user may specify an interval on the pod called the **termination
  grace period** that defines the minimum amount of time the pod will
  have to complete the termination phase, and all components will honor
  this interval.
* Once a pod begins termination, its termination grace period can only
  be shortened, not lengthened.

Pod termination is divided into the following steps:

* A component requests the termination of the pod by issuing a DELETE
  to the pod resource with an optional **grace period**
  * If no grace period is provided, the default from the pod is leveraged
* When the kubelet observes the deletion, it starts a timer equal to the
  grace period and performs the following actions:
  * Executes the pre-stop hook, if specified, waiting up to **grace period**
    seconds before continuing
  * Sends the termination signal to the container runtime (SIGTERM or the
    container image's STOPSIGNAL on Docker)
  * Waits 2 seconds, or the remaining grace period, whichever is longer
  * Sends the force termination signal to the container runtime (SIGKILL)
* Once the kubelet observes the container is fully terminated, it issues
  a status update to the REST API for the pod indicating termination, then
  issues a DELETE with grace period = 0.

If the kubelet crashes during the termination process, it will restart the
termination process from the beginning (grace period is reset). This ensures
that a process is always given **at least** grace period to terminate cleanly.

A user may re-issue a DELETE to the pod resource specifying a shorter grace
period, but never a longer one.

Deleting a pod with grace period 0 is called **force deletion** and will
update the pod with a `deletionGracePeriodSeconds` of 0, and then immediately
remove the pod from etcd. Because all communication is asynchronous,
force deleting a pod means that the pod processes may continue
to run for an arbitary amount of time. If a higher level component like the
PetSet controller treats the existence of the pod API object as a strongly
consistent entity, deleting the pod in this fashion will violate the
at-most-one guarantee we wish to offer for pet sets.


### Guarantees provided by replica sets and replication controllers

ReplicaSets and ReplicationControllers both attempt to preserve availability
of their constituent pods over ensuring **at most one** semantics. So a
replica set to scale 1 will immediately create a new pod when it observes an
old pod has been deleted, and as a result at many points in the lifetime of
a replica set there will be 2 copies of a pod's processes running concurrently.
Only access to exclusive resources like storage can prevent that simultaneous
execution.

Deployments, being based on replica sets, can offer no stronger guarantee.


### Concurrent access guarantees for shared storage

A persistent volume that references a strongly consistent storage backend
like AWS EBS, GCE PD, OpenStack Cinder, or Ceph RBD can rely on the storage
API to prevent corruption of the data due to simultaneous access by multiple
clients. However, many commonly deployed storage technologies in the
enterprise offer no such consistency guarantee, or much weaker variants, and
rely on complex systems to control which clients may access the storage.

If a PV is assigned a iSCSI, Fibre Channel, or NFS mount point and that PV
is used by two pods on different nodes simultaneously, concurrent access may
result in corruption, even if the PV or PVC is identified as "read write one".
PVC consumers must ensure these volume types are *never* referenced from
multiple pods without some external synchronization. As described above, it
is not safe to use persistent volumes that lack RWO guarantees with a
replica set or deployment, even at scale 1.


## Proposed changes

### Avoid multiple instances of pods

To ensure that the Pet Set controller can safely use pods and ensure at most
one pod instance is running on the cluster at any time for a given pod name,
it must be possible to make pod deletion strongly consistent.

To do that, we will:

* Give the Kubelet sole responsibility for normal deletion of pods -
  only the Kubelet in the course of normal operation should ever remove a
  pod from etcd (only the Kubelet should force delete)
  * The kubelet must not delete the pod until all processes are confirmed
    terminated.
* Application owners must be free to force delete pods, but they *must*
  understand the implications of doing so, and all client UI must be able
  to communicate those implications.
* All existing controllers in the system must be limited to signaling pod
  termination (starting graceful deletion), and are not allowed to force
  delete a pod.
  * The node controller will no longer be allowed to force delete pods -
    it may only signal deletion by beginning (but not completing) a
    graceful deletion.
  * The GC controller may not force delete pods
  * The namespace controller used to force delete pods, but no longer
    does so. This means a node partition can block namespace deletion
    indefinitely.
  * The pod GC controller may continue to force delete pods on nodes that
    no longer exist if we treat node deletion as confirming permanent
    partition. If we do not, the pod GC controller must not force delete
    pods.
* It must be possible for an administrator to effectively resolve partitions
  manually to allow namespace deletion.
* Deleting a node from etcd can be seen as a signal to the cluster that
  the node is permanently partitioned. We must audit existing components
  to verify this is the case.
  * Alternatively, we could require that pods must be individually
    terminated. Complicates cloud controller interactions, but may be
    easier for admins to reason about.

In the above scheme, force deleting a pod releases the lock on that pod and
allows higher level components to proceed to create a replacement.

It has been requested that force deletion be restricted to privileged users.
That limits the application owner in resolving partitions when the consequences
of force deletion are understood, and not all application owners will be
privileged users. For example, a user may be running a 3 node etcd cluster in a
pet set. If pet 2 becomes partitioned, the user can instruct etcd to remove
pet 2 from the cluster (via direct etcd membership calls), and because a quorum
exists pets 0 and 1 can safely accept that action. The user can then force
delete pet 2 and the pet set controller will be able to recreate that pet on
another node and have it join the cluster safely (pets 0 and 1 constitute a
quorum for membership change).


### Fencing

The changes above allow Pet Sets to ensure at-most-one pod, but provide no
recourse for the automatic resolution of cluster partitions during normal
operation. For that, we propose a **fencing controller** which exists above
the current controller plane and is capable of detecting and automatically
resolving partitions. The fencing controller is an agent empowered to make
similar decisions as a human administrator would make to resolve partitions,
and to take corresponding steps to prevent a dead machine from coming back
to life automatically.

Fencing controllers most benefit services that are not innately replicated
by reducing the amount of time it takes to detect a failure of a node or
process, isolate that node or process so it cannot initiate or receive
communication from clients, and then spawn another process. It is expected
that many PetSets of size 1 would prefer to be fenced, given that most
applications in the real world of size 1 have no other alternative for HA
except reducing mean-time-to-recovery.

While the methods and algorithms may vary, the basic pattern would be:

1. Detect a partitioned pod or node via the Kubernetes API or via external
   means.
2. Decide whether the partition justifies fencing based on priority, policy, or
   service availability requirements.
3. Fence the node or any connected storage using appropriate mechanisms.

For this proposal we only describe the general shape of detection and how existing
Kubernetes components can be leveraged for policy, while the exact implementation
and mechanisms for fencing are left to a future proposal. The fencing controller
would be able to leverage a number of systems including but not limited to:

* Cloud control plane APIs such as machine force shutdown
* Additional agents running on each host to force kill process or trigger reboots
* Agents integrated with or communicating with hypervisors running hosts to stop VMs
* Hardware IPMI interfaces to reboot a host
* Rack level power units to power cycle a blade
* Network routers, backplane switches, software defined networks, or system firewalls
* Storage server APIs to block client access

to appropriately limit the ability of the partitioned system to impact the cluster.

To allow users, clients, and automated systems like the fencing controllers to
observe partitions, we propose an additional responsibility to the node controller
or any future controller that attempts to detect partition. The node controller should
add an additional condition to pods that have been terminated due to a node failing
to heartbeat that indicates that the cause of the deletion was node partition.

It may be desirable for users to be able to request fencing when they suspect a
component is malfunctioning. It is outside the scope of this proposal but would
allow administrators to take an action that is safer than force deletion, and
decide at the end whether to force delete.

How the fencing controller decides to fence is left undefined, but it is likely
it could use a combination of pod forgiveness (as a signal of how much disruption
a pod author is likely to accept) and pod disruption budget (as a measurement of
the amount of disruption already undergone) to measure how much latency between
failure and fencing the app is willing to tolerate. Likewise, it can use its own
understanding of the latency of the various failure detectors - the node controller,
any hypothetical information it gathers from service proxies or node peers, any
heartbeat agents in the system - to describe an upper bound on reaction.


### Storage Consistency

To ensure that shared storage without implicit locking be safe for RWO access, the
Kubernetes storage subsystem should leverage the strong consistency available through
the API server and prevent concurrent execution for some types of persistent volumes.
By leveraging existing concepts, we can allow the scheduler and the kubelet to enforce
a guarantee that an RWO volume can be used on at-most-one node at a time.

In order to properly support region and zone specific storage, Kubernetes adds node
selector restrictions to pods derived from the persistent volume. Expanding this
concept to volume types that have no external metadata to read (NFS, iSCSI) may
result in adding a label selector to PVs that defines the allowed nodes the storage
can run on (this is a common requirement for iSCSI, FibreChannel, or NFS clusters).

Because all nodes in a Kubernetes cluster possess a special node name label, it would
be possible for a controller to observe the scheduling decision of a pod using an
unsafe volume and "attach" that volume to the node, and also observe the deletion of
the pod and "detach" the volume from the node. The node would then require that these
unsafe volumes be "attached" before allowing pod execution. Attach and detach may
be recorded on the PVC or PV as a new field or materialized via the selection labels.

Possible sequence of operations:

1. Cluster administrator creates a RWO iSCSI persistent volume, available only to
   nodes with the label selector `storagecluster=iscsi-1`
2. User requests an RWO volume and is bound to the iSCSI volume
3. The user creates a pod referencing the PVC
4. The scheduler observes the pod must schedule on nodes with `storagecluster=iscsi-1`
   (alternatively this could be enforced in admission) and binds to node `A`
5. The kubelet on node `A` observes the pod references a PVC that specifies RWO which
   requires "attach" to be successful
6. The attach/detach controller observes that a pod has been bound with a PVC that
   requires "attach", and attempts to execute a compare and swap update on the PVC/PV
   attaching it to node `A` and pod 1
7. The kubelet observes the attach of the PVC/PV and executes the pod
8. The user terminates the pod
9. The user creates a new pod that references the PVC
10. The scheduler binds this new pod to node `B`, which also has `storagecluster=iscsi-1`
11. The kubelet on node `B` observes the new pod, but sees that the PVC/PV is bound
    to node `A` and so must wait for detach
12. The kubelet on node `A` completes the deletion of pod 1
13. The attach/detach controller observes the first pod has been deleted and that the
    previous attach of the volume to pod 1 is no longer valid - it performs a CAS
    update on the PVC/PV clearing its attach state.
14. The attach/detach controller observes the second pod has been scheduled and
    attaches it to node `B` and pod 2
15. The kubelet on node `B` observes the attach and allows the pod to execute.

If a partition occurred after step 11, the attach controller would block waiting
for the pod to be deleted, and prevent node `B` from launching the second pod.
The fencing controller, upon observing the partition, could signal the iSCSI servers
to firewall node `A`. Once that firewall is in place, the fencing controller could
break the PVC/PV attach to node `A`, allowing steps 13 onwards to continue.


### User interface changes

Clients today may assume that force deletions are safe. We must appropriately
audit clients to identify this behavior and improve the messages. For instance,
`kubectl delete --grace-period=0` could print a warning and require `--confirm`:

```
$ kubectl delete pod foo --grace-period=0
warning: Force deleting a pod does not wait for the pod to terminate, meaning
         your containers will be stopped asynchronously. Pass --confirm to
         continue
```

Likewise, attached volumes would require new semantics to allow the attachment
to be broken.

Clients should communicate partitioned state more clearly - changing the status
column of a pod list to contain the condition indicating NodeDown would help
users understand what actions they could take.


## Backwards compatibility

On an upgrade, pet sets would not be "safe" until the above behavior is implemented.
All other behaviors should remain as-is.


## Testing

All of the above implementations propose to ensure pods can be treated as components
of a strongly consistent cluster. Since formal proofs of correctness are unlikely in
the foreseeable future, Kubernetes must empirically demonstrate the correctness of
the proposed systems. Automated testing of the mentioned components should be
designed to expose ordering and consistency flaws in the presence of

* Master-node partitions
* Node-node partitions
* Master-etcd partitions
* Concurrent controller execution
* Kubelet failures
* Controller failures

A test suite that can perform these tests in combination with real world pet sets
would be desirable, although possibly non-blocking for this proposal.


## Deferred issues

* Live migration continues to be unsupported on Kubernetes for the foreseeable
  future, and no additional changes will be made to this proposal to account for
  that feature.


## Open Questions

* Should node deletion be treated as "node was down and all processes terminated"
  * Pro: it's a convenient signal that we use in other places today
  * Con: the kubelet recreates its Node object, so if a node is partitioned and
    the admin deletes the node, when the partition is healed the node would be
    recreated, and the processes are *definitely* not terminated
  * Implies we must alter the pod GC controller to only signal graceful deletion,
    and only to flag pods on nodes that don't exist as partitioned, rather than
    force deleting them.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/pod-termination.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
