# Durable Local Volumes: Design Alternatives

## Related Issues
Request for Durable Local Storage: https://github.com/GoogleCloudPlatform/kubernetes/issues/598

The life of a Pod: https://github.com/GoogleCloudPlatform/kubernetes/pull/1406

## Requirements

Kubernetes requirements:
  - Do not endorse pinning minions to named pods.  In general, K8s should provide targeted solutions which model a specific problem making mention of a specific minions. Examples:
          - need specific kernel or kubelet version:  minion has attributes, and pods can require a minion with matching attribute
          - need failure domain spreading: scheduler could provide automatically for pods with same labels.
          - need durable data: this proposal
  - Security and isolation between unrelated pods (HostDirectory does not provide this)
  - Work regardless of CloudProvider.
  - Support local restarts of containers that depend on durable data
    - Desirable for high availability and bootstrapping.
  - A container spec with no volumes should have an identical filesystem view on every container instantation.  
    - Allows writing truly stateless containers.

Application requirements:
  - have a chunk of writeable data which is not automatically deleted by Kubernetes due to certain _events_, described
    below.  (EmptyDirectory does not provide this, nor does the container's writeable file layer.)
  - high speed access (networked storage may not provide this.  local SSD does.).
  - No external orchestration should be required in order to set up the durable storage volume (e.g., creating/mounting host directories).
  - It should be possible to update a pod (e.g., change a container's image version) without losing the durable data.

Application Examples:
  - Mysql database.  Either single instance, or master-slave replicated.  Tables expected in `/var/lib/mysql`, which we
    want to be some kind of durable local volume.  Want to updgrade mysql version without losing tables or incurring
    traffic to recover from replica.
  - Video streaming server holds large cache of videos to stream.  This set of videos changes slowly over time.
  - Mongdb database.  Uses replication for high availability.  Data stored at `/var/lib/mongodb/`, which we want to be
    some kind of durable local volume.  If one of the physical nodes hosting pod unexpectedly reboots, is partitioned,
    or has a hardware failure, want to create a new replica.  Replication controller accomplishes this.  MongoDB will
    automatically fail over to a replica if the lost pod was a master.

## Current Behavior

A Pod has a (current and desired) PodState, which has ContainerManifest that lists the Volumes and Containers of a Pod.
A Container always has a writeable filesystem, and it can attach Volumes to that which are also writeable.  Writes not
to a volume are not visible to other containers.  Writes to an EmptyDirectory volume are visible to other containers in the pod.
 A restarted container can see state in the EmptyDirectory from a previous invokation.  An EmptyDirectory is only shared by containers which
are part of the same pod ID (UID in v1beta3) and is garbage collected when no longer referenced (the PR that adds
garbage collection is not merged, but this description is written assuming it is.)

How specific events are handled:
  - *Container Failure*: Container exits abnormally (out of memory, segfault, assertion failed, etc), or normally, or fails a liveness probe.
    - Kubelet restarts it if policy allows (RestartAlways or RestartOnFailure and exited abnormally).  The
      EmptyDirectory is reused because PodID is the same.  The docker writable layer is discarded.
    - If policy does not allow restart, then it will not run again with the same UID.  Therefore, EmptyDirectory is not reused and is eventually garbage collected.
    - Use of restartPolicy other than Always is not recommended with replicationController (see #1406)
  - *Pod Update*:  desiredState modified, but Pod Name and UID are unchanged (e.g. used updates to new version of docker image)
    - Kubelet sees a mismatch between checksums of current and desired ContainerManifest, so it deletes the
      docker containers and starts them again.  Since Pod UID is unchanged, it reuses the EmptyDirectory. The docker writable layer is discarded.
    - No-restart Updates not currently supported, but may be required in future (e.g. change resource limits, health check
      settings, etc).  In this case, the docker writable layer is not discarded because the docker container keeps
      running.
  - *Replication Count Reduced*: user modifies replicationController resource.
    - Note: current plan for Rolling updates involves reducing the count on one replicationController and increasing on
      another.
    - Pod is deleted by replicationController when replication count is reduced by user.
    - EmptyDirectory is garbage collected.  It will not be reused by same-Name different-UID pods.
  - *Node Partition*: A node is partitioned from network and then rejoins network.
    - Current behavior is that the pod is still running afterwards, with its data
    - Planned behavior described in #1406 is that pod will be terminated and deleted if minion unreachable for longer than some
      threshold.  When kubelet returns and does not see the the pod on the list of pods for its minion, it terminates
      the running pod.  The EmptyDirectory is garbage collected, so data is lost.
  - *Node Reboots*
    - Current behavior is that after a reboot, Kubelet sees no running containers and sees pods listed in etcd, and so
      starts pods.  API shows the pod as running the whole time.  Since the pods have the same UID, then the existing
      EmptyDirectory is reused. 
    - Planned behavior described in #1406 is that a pod might be deleted, depending on the timing of the reboot and the
      node controller threshold.  In this case, a replacement may be created by the replicationController.  There will
      be no resue of the data due to the new pod UID.
  - *Kubelet Restarts*: the kubelet binary crashes or exits normally, then restarts.
    - Currently, Kubelet restart will allow pod to remain running, data intact.
  - *Storage Device Fails*
    - Pod may get errors on reads/writes.  Failure of health check is likely.  See *Container Failure*.  If
      RestartAlways, then pod will likely enter endless fail and restart loop.
    - Failure could happen immediately following reboot, prior to pod restart.  Data lost in this case too.
    - Expect that will need to implement a control loop that detects bad storage hardware and deletes the pods that
      depend on them.

## Alternatives

There appear to be two main design alternatives:
  1. *Durable Pods*: Make pods live longer than reboots, network partitions, etc.  Their data (in one or more EmptyDirectory) is then also long lived.
  1. *New Resource Kind*: Make an object that represents data that has a long lifetime, and allow pods to attach to it.

They are explained in the next two subsections.

### Durable Pods

Changes:
  - Bit(s) on a Pod which indicates whether durability is requested.
  - Kubelet sees whole pod object (currently has only ContainerManifest) to get UID. (Already planned).
  - Kubelet attaches EmptyDirectory when starting pod that wants durable data and has matching UID.
  - Kubelet to inventory existing EmptyDirectory objects after reboot.
  - Kubelet may need checkpoint or the equivalent to track the preserved EmptyDirectory dirs on disk, and to remember
    the UID of stopped-and-about-to-be-recreated pods.
  - A new kind of replicationController which supports in-place updates instead of create/delete of pods.  Needs two podTemplates: one for old state, and one for new state. 

How events handled:
  - Container Failure
    - No change from current.
  - Pod Update
    - Rolling updates would have to change from create/delete to in-place updates.
  - Replication Count Reduced
    - Not able to directly handle this case.
  - Node Reboots
    - Kubelet now detects existing EmptyDirectory after reboot and attaches to matching Pods when starting those Pods.
  - Node Partition
    - No change required under current code.
    - When NodeController (#1406) implemented, it should not delete a pod wanting durability from a node that is
      unreachable (or wait for forgiveness-specified duration before deleting).
  - Kubelet Restarts
    - No change.
  - Storage Device Fails
    - No change.  If a control loop is added to detect failed storage hardware, then it would delete the pods that had
      EmptyDirectories on the failed device.

### New Resource Kind

Changes:
  - Add a new /data REST resource (not necessarily final name) that represents a chunk of data on a storage device
    somewhere.  Has lifetime independent of a pod.
  - Add a new type of [Volume](../volumes.md) that a pod can request with selector to match /data objects that satisfy
    it.
  - Scheduler changes to handle Data.
    - always try to bind an unbound pod that needs a Data to an existing bound matching Data, if one exists.
    - If both a pod and a matching Data are both unbound, and the previous step failed, the scheduler binds the Pod and
      the Data at the same time (to ensure that the Pods constraints are met and that they both fit in the free space of
      the node.)
    - Once a /data is bound, it stays bound until the user deletes the /data.
  - Kubelet changes to support creating a Data, and to detect an existing Data after reboot.
  - Kubecfg support for /data.
  - Replication controller for /data.

How events handled:
  - Container Failure
    - If RestartAlways selected, all volumes (EmptyDirectory, /data volumes) remain. No change.
    - *Changed.*  If other restart policy selected, a new UID but same config pod will reattach to an existing Data.
  - Pod Update
    - No change.
  - Replication Count Reduced
    - *Changed*.  A /data remains until explictly removed.  This allows the Pod Replication Count to be lowered and then raised again
      without loss of data.  Data can be controller by a data-replication-controller, in which case it needs to be adjusted separately.
  - Node Reboots
    - Kubelet fetched from APIserver  what pods and data object the node should have.  It will see existing data objects and leave them
      alone.  It sees that no  containers are running (aside from local-file-configured ones), so it starts them, attaching them to their data.
  - Node Partition
    - Currently, no change.
    - When NodeController implemented, it can terminate all pods on a machine that has been unresponsive for a long
      time.  It does not delete /data.
  - Kubelet Restarts
    - No change.
  - Storage Device Fails
    - No change.  If controller is added to detect failed storage hardware, then it would delete the lost /data object, and /pods.


## Evaluation of Alternatives

Either alternative can easily handle resource limits for Volumes and resource capacities for minions, and new storage
(disk, ssd) resource types.  This is not a deciding factor.

In either alternative, and in the current state, a pod with RestartAlways policy has to handle both the case when it
starts with an empty volume, and when it restarts with a non-empty volume.  This is not a factor.

Either alternative can easily handle a configurable tradeoff between waiting for an unavailable minion to return versus
starting a new instance of the pod and data elsewhere (_forgiveness_).  This is not a deciding factor.

Either alternative can be adapted to allow hedging bets: simultaneously starting a new pod/data pair while waiting for a
lost pod/data to come back.  This is not a deciding factor.

Either alternative could be adapted to support rolling restarts. However, the durable pod approach would significant compromise the flexibility and fungibility intended for replication controllers, because replication controllers have to remember both old and new states for a pod, to do in-place updates.  See #1527.  This is a major point in favor of New Resource Kind.

Either alternative can be adapted to allow a controller to automatically delete pods/datas in case of a storage device
hardware failure.    This is not a deciding factor.

New Resource Kind adds more resource types for users to learn about (though you do not need to know about Data to use
Kubernetes).  This is a point in favor of Durable Pods.

Under the New Resource Kind alternative, a Pod is stateless (assuming it has no HostDirectory or other remote volumes), and
the Data represents state without computation.  This has an attractive simplicity.  Minor point for New Resource Kind.

Under the Durable Pods alternative, the NodeController has to reason about both minions (whether they are likely to be
operating correctly) and about pods (what forgivenesses they want).  Under the New Resource Kind alternative, the
NodeController only needs to reason about minions.  Pods can be terminated as soon as the minion is determined to be
unhealthy.  Separate control systems per group of Data object will decide when to give up on a Data that is on an
unhealthy minion.  The separation of duties is somewhat attactive.  Minor point in favor of New Resource Kind.

In the Durable Pods approach, one has to favor updating an existing pod (keep same Name and UID) over deleting a pod and creating a new
one.  It seems like this may have a number of effects:
  - A config push is not as simple as deleting all the old objects and pushing new ones.  It has to match existing pods
    with new config, and then compose pod updates.  Names could be the thing that matches existing pods to new config. But if names are
    composed by multiple layers of tools, then this could be hard (e.g. tools want to put sofware version id, or release
    tracking id in pod name;  they may want to do this for idempotent creation).  Seems like a big point in favor of New
    Resource Kind.
  - Updates would need to be supported for all fields.  This may be tricky.
  - A pod name cannot be changed to reflect a gradual change in the responsibilities of a pod due to a series of
    updates.   Minor point in favor of New Resource Kind.

New Resource Kind allows scaling data independently of servers, such as to prepare for a spike in traffic without spending
RAM on the servers prior to that spike.  This is a minor point in favor of New Resource Kind.

Pod and Data need to be coscheduled, which adds complexity to the scheduler, and to understanding why a pod cannot
schedule (e.g. data pending, or no free space on machine with my data.)  This is a big point in favor of Durable Pods.

The durability seems like it is an attribute of the data, not the pod.  Consider the case of a pod with two volumes: a
`/tmpfs` EmptyDirectory volume which two containers use for rapid communication that should be cleared when the pod
fails to avoid stale locks or other stale state; and an SSD volume which should be durable.  This seems like a minor
point in favor of New Resource Kind.

New Resource Kind opens up the possibility of having differing permissions for Data versus Pods.  This could be used to put
stricter controls on Data, to prevent accidental erasure, while allowing more flexibility to change Pods (e.g.
autosizing, setting debugging flags, etc).  This seems like a minor point in favor of New Resource Kind.

A /data could be created using a node-local config file, and filled with files using an out-of-band mechanism.  And then Pods could depend on this data.  That might be useful for bootstrapping a cluster.

Overall, it seems like there there are somewhat more points in favor of New Resource Kind.

## Other Alternatives?

A hybrid solution, which requires more thought, is to reuse EmptyDirectories using a key which is longer lived than Pod
UID, but not to expose a separate REST api object for the data of that EmptyDirectory.  A problem would be that
something need to garbage collect those objects, and perhaps in future preempt them and report about their resource
usage.  That suggests that they need their own REST resource.

## TODO

Think about how to handle this use case well:
- Run-once docker container to initialize the database #1589

Think about how "volumes as containers" interacts with this proposal.
- https://github.com/GoogleCloudPlatform/kubernetes/issues/831

