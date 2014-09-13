# Volumes

A process in a container sees a file system which is a combination of its Docker images, plus any Volumes attached to
it.

## Types of Volumes

A Volume is one of the following types:

| Name               | Purpose  |
| ------------------ | -------- |
| HostDirectory      | Escape hatch for access to un-containerized file |
| EmtpyDirectory     | Scratch disk space. |
| DebugDirectory     | Stores debugging logs, core dumps |
| CachedDirectory    | Stores large sets of files which are too expensive to re-create on each pod restart. |
| PackageDirectory   | Readonly content to be managed by Kubernetes system. |
| RemoteDirectory    | Mount network, cluster, or cloud storage as POSIX file system. |

### HostDirectory

A host directory allows access to the node local filesystem.  Avoid using this type since it may result in pods being
sensitive to which node they run on; because there is no access control on which host files can be accessed; and because
the scheduler and kubelet cannot account for resources used by writing to a HostDirectory.

### EmptyDirectory

Same lifetime as pod.  Use this so Containers in the same pod can communicate, and for scratch space.

### DebugDirectory

A DebugDirectory data is mostly the same as an EmptyDirectory, except it is kept after pod termination on a best-effort
basis.  It is deleted as space is needed by the kubelet for EmptyDirectory, CachedDirectory, or PackageDirectory objects.

TBD: merge this with EmptyDirectory?

### CachedDirectory

Keeps data around after a pod terminates.  Scheduler can then bind other pods / pod-invokations to the same data.  Pod
does not need to setup data again.

Pod has to be able to detect at startup if it is getting an empty directory, or one from a previous pod invokation, and
validate the contents. 
If a pod crashes, it needs to not leave the data in a bad state, or be able to recover in the next invokation, or signal
to an admin for help.
Existing applications being moved to k8s likely already have mechanisms to deal with these issues, so k8s does not try
to provide its own.

### PackageDirectory

Includes a recipe for installing data which k8s can invoke.  Allows data to be installed before pod is ever started.  Enables:
   1. faster pod startup in case of sudden need
   1. sharing of storage space across pods (for resource efficiency, but not for pod-to-pod interaction)
   1. k8s management of data push workflows.
   1. temporal reuse without the application-awareness that is required for CachedDirectory.

To enable space sharing, and temporal reuse, the data must be read-only when accessed by the Pod that depends on it.

The recipe for installing data is TBD.  Some considerations:
   1. For the _install_ phase, the Volume has to be made read-write for the install process..  After it is successfully installed, it enters a
   _use_ phase where it should only be mounted read-only for dependent pods.
   1. The install recipe could just be a reference to a Pod or a literal Pod, which knows how to install data into a
   special sort of EmptyDirectory that does not get deleted.  Successful exit of the Pod makes it transition to _use_
   phase.
   1. Problem with generic pod approach is that if users are not careful, they may define a procedure that is not repeatable every time,
   which is bad. 
   1. Another approach would be to specify an URL to fetch and untar.  Kubelet would invoke this.  Needs to ensure it
   cannot create setuid binaries, etc.

A PackageDirectory is similar to a docker image.  However:
  - Images are composed of layers built on top of a base image which needs to provide certain base linux files.  A Package
    is monolithic, independent of other packages, and typically does not have _root filesystem structure_.
  - Packages can reference other K8s objects in their definitions (Pods) and have K8s-defined behaviors.
  - TODO think more about relationship between these two.
  - Think about this in the context of  #994.

### RemoteDirectory

A RemoteDirectory is for mounting remote storage as a POSIX file system accessible to Containers.
This is especially useful for apps ported from the non-container enviroments.  
Apps written for container environments are encouraged to use direct file api access to cloud storage systems and
databases.

RemoteDirectory is actually a category, and not a specific type.  The category includes GCEPersistentDisk (see #861).


## Comparison of features

| Name             | Empty at pod start | Deleted at pod exit    | Storage | read-write? | Status      |
| ---------------- | ------------------ | ---------------------- | ------- | ----------- | ----------- |
| HostDirectory    | no                 | no                     | local   | yes         | implemented |
| EmptyDirectory   | yes                | yes                    | local   | yes         | implemented |
| DebugDirectory   | yes                | not immediately        | local   | yes         | not implemented |
| CachedDirectory  | maybe              | no;  api call required | local   | yes         | not implemented |
| PackageDirectory | no                 | depends                | local   | no          | not implemented |
| RemoteDirectory  | no                 | never                  | remote  | maybe       | not implemented |

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
| DebugDirectory   | BoundDebugData | Represents data that lives after pod. | 
| CachedDirectory  | BoundCachedData | Represents data that lives independent of pod. | 
| PackageDirectory | BoundPackageData | Represents data that lives independent of pods. |
| RemoteDirectory  | n/a | same lifetime as pod, and no associated state. |

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
 - have a resource request
 - represent specific resources allocated/used on a non-specific node.
 - have own lifetime
 - can be created manually or created by a controller
 - have labels.
 - get bound to a machine by the scheduler.

The scheduler creates a BoundCachedData in response to creation of a CachedData object.
Only a CachedDirectory has a corresponding user-visible CachedData object.   

| Name        | Who creates | Who deletes |
| ----------- | ----------- | ----------- |
| BoundDebugData | Scheduler | kubelet (when space needed) |
| BoundCachedData | Scheduler | scheduler |
| CachedData | User or controller | User or controller |
| BoundPackageData | Scheduler | Scheduler |

In future, we may introduce something user-visible, like a PackageData object, to allow controllers to manage the pre-installation and
persistence of BoundPackageData.  For now, this is not supported.

Open questions:
  1. what if  more than 1 CachedData on the same machine can satisfy the Pods need?  Arbitrary?
  1. what if two pods could be bound to the same machine, and there is one CachedData that can satisfy either of them.


## Resource Requirements

All types except RemoteDirectory and HostDirectory should specify their resource requirements (amount of disk, ssd, ram, etc).
This allows the scheduler to place them where there will be room.  This is pending support for pods specifying resource
requirements.

A cluster admin who allows users to start pods that use HostDirectory is responsible for making sure disk space on nodes is not exhausted.

| Name             |  Resource Reqs? | 
| ---------------- |  -------------- | 
| HostDirectory    |  no             | 
| EmptyDirectory   |  yes            | 
| DebugDirectory   |  yes            | 
| CachedDirectory  |  yes            | 
| PackageDirectory |  yes            | 
| RemoteDirectory  |  TBD            | 

The Kubelet will enforce resource limits on each Volume.  The scheduler will ensure space exists on the node for each
Volume.

All implementations should support requesting just the disk space resource.  Some implementations may:
  - provide an SSD-backed volume if just SSD is requested.
  - provide a tmpfs-backed volume if just RAM is requested.
  - provide a software hybrid drive if multiple resources are requested.
  - provide a software managed striping across physical disks if large DTF values are requested.
  - spread mutliple volume requests across physical device, one per device.

No implementation should support asking for resources that span mutliple physical devices in a way visible to a Volume user.

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

### Upgrade/rollback use of PackageData
A web server pod has 10 replicas.  It uses a lot of static content which comes in a single tarball stored in a release
server. We want to be able to roll out a new version of the static content, and then trigger the servers to restart with
the new data.  We do not mind if one of the 10 replicas is unavailable during an upgrade, and will do a rolling
upgrade.  But, we want the ability to roll-back very quickly to version current - 1 or current - 2.

Therefore, we create a podTemplate with 3 volumes, that are PackageDirectories that in turn install tarballs
`static_content.v11.tgz`, 
`static_content.v12.tgz`,  and
`static_content.v13.tgz`.
The container command line selects `v13`.

To do an emergency rollback, you can update the pods in a rolling fashion to point at `v11`.  No data install delay.

When `v14` is ready, you make a new template that adds `static_content.v13.tgz` and deletes
`static_content.v11.tgz` from the Volumes list.  

The scheduler should recognize when a machine has most of the packages needed, and strongly prefer to bind the new pod
there too.

### Autoscaling use of PackageData
This is the same webserver as the previous example.
We also want to be able to scale up the servers replication count rapidly in case we are mentioned on slashdot or
something.  But we do not want to spin up webserver pods and CPU and RAM until that happens.  But we can afford to keep
the static data installed on some extra node hard drives.

This should be possible by creating dummy pods that use few resources but do trigger package pre-install.

## References
https://github.com/GoogleCloudPlatform/kubernetes/issues/598
https://github.com/GoogleCloudPlatform/kubernetes/issues/97

## TODO

Figure out why Volume is on ContainerManifest and on podTemplate and which matters for what.

File bug to make HostDirectory flag disable-able for people who do not want that insecurity.
