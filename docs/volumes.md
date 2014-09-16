# Volumes

A Volume is a directory, possibly with some data in it, which is exposed to a Container. The file system that a
container experiences is a combination of its Docker image plus any volumes mounted into the container. This is similar
in most regards to what Docker knows as Volumes, but is different in some of the details of how Volumes are defined and
managed.

## Types of Volumes

A Volume is one of the following types:

| Name               | Purpose  |
| ------------------ | -------- |
| HostDirectory      | Escape hatch for access to un-containerized file |
| EmptyDirectory     | Scratch disk space. |
| CachedDirectory    | Stores large sets of files which are too expensive to re-create on each pod restart. |

### HostDirectory

A host directory allows access to the node local filesystem.  Avoid using this type since it may result in pods being
sensitive to which node they run on; because there is no access control on which host files can be accessed; and because
the scheduler and kubelet cannot account for resources used by writing to a HostDirectory.

### EmptyDirectory

If a container fails health check, then the Pods EmptyDirectory persists.
If a pod is deleted by the user or a controller (replicationController does not currently delete pods), then the
EmptyDirectory is deleted.
If the network container goes away, all the containers are recreated, but the EmptyDirectory persists.
EmptyDirectories are used to store logs, for scratch space, and so that containers in the same pod can communicate.

### CachedDirectory

Keeps data around after a pod terminates.  Scheduler can then bind other pods / pod-invokations to the same data.  Pod
does not need to setup data again.

Pod has to be able to detect at startup if it is getting an empty directory, or one from a previous pod invokation, and
validate the contents. 
If a pod crashes, it needs to not leave the data in a bad state, or be able to recover in the next invokation, or signal
to an admin for help.
Existing applications being moved to k8s likely already have mechanisms to deal with these issues, so k8s does not try
to provide its own.

## Comparison of features

| Name             | Empty at pod start | Deleted at pod exit    | Storage | read-write? | Status      |
| ---------------- | ------------------ | ---------------------- | ------- | ----------- | ----------- |
| HostDirectory    | no                 | no                     | local   | yes         | implemented |
| EmptyDirectory   | yes                | yes                    | local   | yes         | implemented |
| CachedDirectory  | maybe              | no;  api call required | local   | yes         | not implemented |

## Data API objects 

All the *Directory types mentioned above are part of a VolumeSource object.  They are not objects in themselves.  They
only represent a view on some data, not the data itself.  

Some *Directory types do have a corresponding API object which represents bytes allocated on a node storage device,
possibly with multiple nearly-identical objects representing multiple copies of similar information.  The following
table summarizes the types and their corresponding object or reason for lack of one.

| Name             | Corresponding object for instances of data |  Explanation |
| ---------------- | ------------------------------------------ | ----------- |
| HostDirectory    | n/a | K8s does not manage node files. | 
| EmptyDirectory   | n/a | Lifetime same as pod, so Pod represents the EmptyDirectory data |
| CachedDirectory  | BoundCachedData | Represents data that lives independent of pod. | 

The system manages Bound*Data objects for the user.  They are write-only like BoundPods.


### CachedData
The CachedDirectory section uses a Label Selector to refer to what CachedData API objects could satisfy it.
A CachedData is created with a Label so it can be found by a Selector.
In one common case, this would be a single unique label value used by one or more fungible CachedData objects, and an exact match
selector used by one or more fungible Pod objects
The use of selectors:
  - allows Pod creation and CachedData creation to happen independently, in any order.
  - allows any of several fungible Pods to use any of several fungible CachedData objects.  (Object Names have to be
    unique, so cannot fill this role.)

A CachedData is analagous to a Pod object in that they both:
 - are expected to have a resource request in a future version of Kubernetes.
 - represent specific resources requested, and which are allocated on a specific node (once bound).
 - have own lifetime
 - can be created manually or created by a controller
 - have labels.
 - get bound to a machine by the scheduler.

The scheduler creates a BoundCachedData in response to creation of a CachedData object.
Only a CachedDirectory has a corresponding user-visible CachedData object.   

| Name        | Who creates | Who deletes |
| ----------- | ----------- | ----------- |
| BoundCachedData | Scheduler | scheduler |
| CachedData | User or controller | User or controller |

Open questions:
  1. what if  more than 1 CachedData on the same machine can satisfy the Pods need?  Arbitrary?
  1. what if two pods could be bound to the same machine, and there is one CachedData that can satisfy either of them.


## Scheduling
When scheduling a Pod that needs a CachedVolume, the scheduler needs to recognize when it should
  - bind the Pod next to an existing BoundCachedVolume.
  - bind a CachedVolume and then a Pod.
Not all implementations are required to handle all possible cases of resource fragmentation.


## Node outage

TODO: figure out this: If a node has an outage such as a reboot or network drop, will the scheduler delete the Pods that
were on it, or move them to a terminated state, or automatically rebind them?

For now, assume that it moves them to a LOST state, and does not automatically rebind the Pods or the CachedData that
were bound to that node.

Fow now, assume that the replicationController is the thing that makes new Pods to replace the LOST ones.

When a node comes back, does the Pod become un-LOST?  Does tbe BoundVolumeCache reappear?

A future controller might be able to reason about choices between waiting for a node to come back versus starting up a
new one, with attendant cost of rebuilding the CachedData.  See also forgiveness concept suggested in #598.  


## Detailed Motivating Examples
### Simplest use of CachedDirectory

User is running a single pod with a database program, say mysql.  Database is backed up nightly to somewhere out of K8s.
During daytime, want to avoid having to restore DB, because this is slow.  
User does the following:
  1. Create CachedData object with label `cached_data=mysql`.
  1. Create Pod with image `mysql` which depends on that CachedData.  The VolumeSource will put this where mysql expects
  to put its database data, e.g. `/var/lib/mysql` within the Containers chroot.
  1. At first setup of k8s, sshes to Pod and runs database restore command.
  1. Sets up monitoring that can detect the case when there is an empty /var/lib/mysql.
  1. In event of machine failure causing pod loss, user gets alert, creates a new CachedData, waits for Pod to bind,
  sshes in, and restores data.

### More sophisticated use of CachedDirectory

Consider a video streaming server that serves a slowly-changing collection of the videos from its flash drive.  These
can be restored from disk, and new ones are periodically added by copying from disk, but this is a slow process.
The video server pod consists of a video server container and a cache-updater program container.

TBD: how to handle healthcheck of both of these, given video server may not run well with no content?

There is one video server podTemplate, and a replicationController that says to keep 10 video server pods running.
The user manually creates 11 CachedData objects (one spare to handle one machine going away without intervention).

When the user wants to canary an upgrade the video server binary version, he ramps down the replication count on the current
podTemplate, and ramps up the count for a new podTemplate for the new version.   No data is lost.

All the CachedData objects have the same `data=videos` label, and both podTemplates make pods whose Volumes include a
CachedDirectory with Selector `data=videos`.

## References
https://github.com/GoogleCloudPlatform/kubernetes/issues/598
https://github.com/GoogleCloudPlatform/kubernetes/issues/97

## Future Work
The following are examples of possible future work, outlined here only to provide context for the volume types described
above.

### Resource Allocation
We will want to account for resources required and used by Pods and Volumes (see #168, #442, #617, #502).  This may
include:
  - A CachedVolume and a Pod, and maybe an EmptyDirectory,  can specify the amount of resources each requires.
  - Kubelet to report resource usage of CachedVolumes, Pods, and maybe other kinds of volumes, and to enforce limits on
    their size.
  - Scheduler to consider free space when binding Pods and CachedVolumes.
  - CachedVolume and EmptyVolume can be backed by different media types, such as a single disk, an ssd with a filesystem, or memory (tmpfs).  Maybe even multiple disks
    combined into a single logical volume, or a hybrid of disk and flash (e.g.[DM-cache](http://en.wikipedia.org/wiki/Dm-cache)).

### Readonly data packages
We may want to introduce one or more kinds of volume that act like a read-only data package.  They might do some of the following:
  - Hold readonly content, such as static webserver content.
  - Be automatically installed and uninstalled by K8s as pods depend on it.
  - Be pre-installed to allow faster scaling of sets of pods that depend on the slow-to-install data.
  - Allow sharing of storage space across pods (for rourcesource efficiency, but not for pod-to-pod interaction)
  - Have an associated controller that manages atomically updating and rolling back static content versions while the Pod keeps serving.
  - Have an associated controller that watches for github changes and updates the pod and volume in response.
  - Provide functionality similar to Docker `volumes_from`
Need to determine how this concept relates to Docker images.

### Saving debugging data
We may want to introduce a kind of volume that holds debugging data.  It might do the following:
  - Start empty, but be kept around after pod termination on a configurable or best-effort basis.
  - Hold things like core files and large debugging log files, for debugging a failed container.
Need to determine whether keeping files like this node-locally is necessary, or copying them to a central repository is
better.

### Remote filesystems
We may want to introduce one or more kinds of volume that represent remote data.  They might do some of the following:
  - Provide POSIX-type access to remotely-stored data.
  - Provide access to NFS, GCE Persistent Disks, GCEPersistentDisk (see #861), etc.
We might rather encourage containerized apps to use direct file api access to cloud storage
systems (such as S3, GCS) in place of POSIX-type access.

## TODO
Figure out why Volume is on ContainerManifest and on podTemplate and which matters for what.

File bug to make HostDirectory flag disable-able for people who do not want that insecurity.
