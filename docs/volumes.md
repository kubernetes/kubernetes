In addition to the files provided by the Docker image of a conatiner, a process in the container can also see any attached Volumes. 
A Volume is one of the following types:
| Name             | Empty at pod start | When deleted by k8s?   | Storage    | resource reqs? | read-write? | Status      |
| HostDirectory    | no                 | never                  | node-local | no             | yes         | implemented |
| EmptyDirectory   | yes                | when pod is terminated | node-local | yes            | yes         | implemented |
| DebugDirectory   | yes                | some time after        | node-local | yes            | yes         | not implemented |
| CachedDirectory  | maybe.             | K8s api call required  | node-local | yes            | yes         | not implemented |
| PackageDirectory | no                 | as needed by scheduler | node-local |  yes           | yes by installer, no by pod |  not implemented |
| RemoteDirectory  | no                 | never                  | remote     | yes            | maybe       | not implemented |

Use cases and comments:
| Name               | Use Case | Comments |
| HostDirectory      | a pod with special privileges needs to access node-local files.  |
| EmtpyDirectory     | scratch disk space for large computations.  Could be used for logs.  |
| DebugDirectory     | a place to store debugging logs, output from quick tests, core dumps |
| CachedDirectory    | A video serving application keeps large amounts of hot video files on local disk or flash. |
| PackageDirectory   | A web server has a large amount of static content.  |
| RemoteDirectory   | a tree of files on NFS or a cluster filesystem      | For legacy apps needing unix file semantics.  New apps are recommended to use direct file api access to remote storage. |

## API objects
TODO: figure out why Volume is on ContainerManifest and on podTemplate and which matters for what.

All the *Directory types mentioned above are part of a VolumeSource object.

A CachedDirectory or PackageDirectory has to be created in the API before it can be reference by a Pod in its VolumeSource section.

The CachedDirectory, PackageDirectory also have corresponding API objects which represent the actual data itself:
BoundCachedDirectory, BoundPackageDirectory.  TBD if DebugDirectory has a Bound counterpart.

## Node outage

If a node has an outage such as a reboot or network drop, the pods on it may die or be terminated when the node rejoins the cluster.
However, CachedDirectory and PackageDirectory do not.  They represent data that is expensive to install.

## CachedDirectory

When the first pod ever to use a BoundCachedData is started, it needs to know how to populate the cache.
If it crashes, it needs to not leave the data in a bad state, or be able to recover in the next invokation, or signal to an admin for help.
Existing applications being moved to k8s probably already have dealt with this.

When subsequent invokations of a pod that depend on the same-name CachedDirectory start, they see the files from the previous invokation.

## PackageDirectory
Consider a webserver with a lot of static data which takes a long time to install.
Consider that we need to scale it up rapidly but do not want to create pods and waste CPU and ram when it is not scaled up.
We can pre-install Packages and have some policy and controller to manage the number and locations of these cold-spares.

These need to be readonly if we want to be sure these are in a valid state after they have been used by a pod, and then scaled down.
Also, if we want to have several pods share a common large chunk of data.

In order to separate installation of data from a pod running, the PackageDirectory has to define some kind of install command.
Is this tar, or is this a Pod with one container that needs to run to completion?  Or something else.  How does this related to images?

TODO: consider removing this one, as it is the one I am least sure about.


## Pod portability
Using HostDirectory may break pod portability across nodes if nodes have different images.

Using RemoteDirectory or CachedDirectory may cause pods to have different behavior on different invokations, and to write bad state which causes future invokations of the pod to crash.  However, using any network-accessible state can cause this too.

## Scaling and migration


## Resource Requirements

All types except RemoteDirectory and HostDirectory should specify their resource requirements (amount of disk, ssd, ram, etc).
This allows the scheduler to place them where there will be room.

A cluster admin who allows users to start pods that use HostDirectory is responsible for making sure disk space on nodes is not exhausted.

## Storage space management
DebugDirectory data is kept after pod termination.  It is deleted or archived as space is needed by the kubelet for Pods, CachedDirectory, or PackageDirectory objects.
Policy for this TBD.

A user or a controller is responsible for creating and deleting CachedDirectories, and ensuring sufficient free space in the cluster.

## Scheduling
A CachedDirectory object gets bound by the scheduler to create a BoundCachedDirectory.
A used or controller can make a decision to delete a binding and influence the scheduler to bind the CachedDirectory elsewhere, if thinks it knows better.

## Naming

DebugDirectory could have a name derived from the UUID of the pod that created it?  What about restarts?  Pod should not have to specify _which_ one it wants.
However, CachedDirectory and PackageDirectory need names so that a pod can depend on a different format/version of data.

## Other use cases
HDFS running in a pod.  Good case for HostDirectory to store its data on the local disks?


