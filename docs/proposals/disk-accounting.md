<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

**Author**: Vishnu Kannan

**Last** **Updated**: 11/16/2015

**Status**: Pending Review

This proposal is an attempt to come up with a means for accounting disk usage in Kubernetes clusters that are running docker as the container runtime. Some of the principles here might apply for other runtimes too.

### Why is disk accounting necessary?

As of kubernetes v1.1 clusters become unusable over time due to the local disk becoming full. The kubelets on the node attempt to perform garbage collection of old containers and images, but that doesn’t prevent running pods from using up all the available disk space.

Kubernetes users have no insight into how the disk is being consumed.

Large images and rapid logging can lead to temporary downtime on the nodes. The node has to free up disk space by deleting images and containers. During this cleanup, existing pods can fail and new pods cannot be started. The node will also transition into an `OutOfDisk` condition, preventing more pods from being scheduled to the node.

Automated eviction of pods that are hogging the local disk is not possible since proper accounting isn’t available.

Since local disk is a non-compressible resource, users need means to restrict usage of local disk by pods and containers. Proper disk accounting is a prerequisite. As of today, a misconfigured low QoS class pod can end up bringing down the entire cluster by taking up all the available disk space (misconfigured logging for example)

### Goals

1. Account for disk usage on the nodes.

2. Compatibility with the most common docker storage backends - devicemapper, aufs and overlayfs

3. Provide a roadmap for enabling disk as a schedulable resource in the future.

4. Provide a plugin interface for extending support to non-default filesystems and storage drivers.

### Non Goals

1. Compatibility with all storage backends. The matrix is pretty large already and the priority is to get disk accounting to on most widely deployed platforms.

2. Support for filesystems other than ext4 and xfs.

### Introduction

Disk accounting in Kubernetes cluster running with docker is complex because of the plethora of ways in which disk gets utilized by a container.

Disk can be consumed for:

1. Container images

2. Container’s writable layer

3. Container’s logs - when written to stdout/stderr and default logging backend in docker is used.

4. Local volumes - hostPath, emptyDir, gitRepo, etc.

As of Kubernetes v1.1, kubelet exposes disk usage for the entire node and the container’s writable layer for aufs docker storage driver.
This information is made available to end users via the heapster monitoring pipeline.

#### Image layers

Image layers are shared between containers (COW) and so accounting for images is complicated.

Image layers will have to be accounted as system overhead.

As of today, it is not possible to check if there is enough disk space available on the node before an image is pulled.

#### Writable Layer

Docker creates a writable layer for every container on the host. Depending on the storage driver, the location and the underlying filesystem of this layer will change.

Any files that the container creates or updates (assuming there are no volumes) will be considered as writable layer usage.

The underlying filesystem is whatever the docker storage directory resides on. It is ext4 by default on most distributions, and xfs on RHEL.

#### Container logs

Docker engine provides a pluggable logging interface. Kubernetes is currently using the default logging mode which is `local file`. In this mode, the docker daemon stores bytes written by containers to their stdout or stderr, to local disk. These log files are contained in a special directory that is managed by the docker daemon. These logs are exposed via `docker logs` interface which is then exposed via kubelet and apiserver APIs. Currently, there is a hard-requirement for persisting these log files on the disk.

#### Local Volumes

Volumes are slightly different from other local disk use cases. They are pod scoped. Their lifetime is tied to that of a pod. Due to this property accounting of volumes will also be at the pod level.

As of now, the volume types that can use local disk directly are ‘HostPath’, ‘EmptyDir’, and ‘GitRepo’. Secretes and Downwards API volumes wrap these primitive volumes.
Everything else is a network based volume.

‘HostPath’ volumes map in existing directories in the host filesystem into a pod. Kubernetes manages only the mapping. It does not manage the source on the host filesystem.

In addition to this, the changes introduced by a pod on the source of a hostPath volume is not cleaned by kubernetes once the pod exits. Due to these limitations, we will have to account hostPath volumes to system overhead. We should explicitly discourage use of HostPath in read-write mode.

`EmptyDir`, `GitRepo` and other local storage volumes map to a directory on the host root filesystem, that is managed by Kubernetes (kubelet). Their contents are erased as soon as the pod exits. Tracking and potentially restricting usage for volumes is possible.

### Docker storage model

Before we start exploring solutions, let’s get familiar with how docker handles storage for images, writable layer and logs.

On all storage drivers, logs are stored under `<docker root dir>/containers/<container-id>/`

The default location of the docker root directory is `/var/lib/docker`.

Volumes are handled by kubernetes.
*Caveat: Volumes specified as part of Docker images are not handled by Kubernetes currently.*

Container images and writable layers are managed by docker and their location will change depending on the storage driver. Each image layer and writable layer is referred to by an ID. The image layers are read-only. Once saved, existing writable layers can be frozen. Saving feature is not of importance to kubernetes since it works only on immutable images.

*Note: Image layer IDs can be obtained by running `docker history -q --no-trunc <imagename>`*

##### Aufs

Image layers and writable layers are stored under `/var/lib/docker/aufs/diff/<id>`.

The writable layers ID is equivalent to that of the container ID.

##### Devicemapper

Each container and each image gets own block device. Since this driver works at the block level, it is not possible to access the layers directly without mounting them. Each container gets its own block device while running.

##### Overlayfs

Image layers and writable layers are stored under `/var/lib/docker/overlay/<id>`.

Identical files are hardlinked between images.

The image layers contain all their data under a `root` subdirectory.

Everything under  `/var/lib/docker/overlay/<id>` are files required for running the container, including its writable layer.

### Improve disk accounting

Disk accounting is dependent on the storage driver in docker. A common solution that works across all storage drivers isn't available.

I’m listing a few possible solutions for disk accounting below along with their limitations.

We need a plugin model for disk accounting. Some storage drivers in docker will require special plugins.

#### Container Images

As of today, the partition that is holding docker images is flagged by cadvisor, and it uses filesystem stats to identify the overall disk usage of that partition.

Isolated usage of just image layers is available today using `docker history <image name>`.
But isolated usage isn't of much use because image layers are shared between containers and so it is not possible to charge a single pod for image disk usage.

Continuing to use the entire partition availability for garbage collection purposes in kubelet, should not affect reliability.
We might garbage collect more often.
As long as we do not expose features that require persisting old containers, computing image layer usage wouldn’t be necessary.

Main goals for images are
1. Capturing total image disk usage
2. Check if a new image will fit on disk.

In case we choose to compute the size of image layers alone, the following are some of the ways to achieve that.

*Note that some of the strategies mentioned below are applicable in general to other kinds of storage like volumes, etc.*

##### Docker History

It is possible to run `docker history` and then create a graph of all images and corresponding image layers.
This graph will let us figure out the disk usage of all the images.

**Pros**
* Compatible across storage drivers.

**Cons**
* Requires maintaining an internal representation of images.

##### Enhance docker

Docker handles the upload and download of image layers. It can embed enough information about each layer. If docker is enhanced to expose this information, we can statically identify space about to be occupied by read-only image layers, even before the image layers are downloaded.

A new [docker feature](https://github.com/docker/docker/pull/16450) (docker pull --dry-run) is pending review, which outputs the disk space that will be consumed by new images. Once this feature lands, we can perform feasibility checks and reject pods that will consume more disk space that what is current availability on the node.

Another option is to expose disk usage of all images together as a first-class feature.

**Pros**

* Works across all storage drivers since docker abstracts the storage drivers.

* Less code to maintain in kubelet.

**Cons**

* Not available today.

* Requires serialized image pulls.

* Metadata files are not tracked.

##### Overlayfs and Aufs

####### `du`

We can list all the image layer specific directories, excluding container directories, and run `du` on each of those directories.

**Pros**:

* This is the least-intrusive approach.

* It will work off the box without requiring any additional configuration.

**Cons**:

* `du` can consume a lot of cpu and memory. There have been several issues reported against the kubelet in the past that were related to `du`.

* It is time consuming. Cannot be run frequently. Requires special handling to constrain resource usage - setting lower nice value or running in a sub-container.

* Can block container deletion by keeping file descriptors open.


####### Linux gid based Disk Quota

[Disk quota](https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_Linux/6/html/Storage_Administration_Guide/ch-disk-quotas.html) feature provided by the linux kernel can be used to track the usage of image layers. Ideally, we need `project` support for disk quota, which lets us track usage of directory hierarchies using `project ids`. Unfortunately, that feature is only available for zfs filesystems. Since most of our distributions use `ext4` by default, we will have to use either `uid` or `gid` based quota tracking.

Both `uids` and `gids` are meant for security. Overloading that concept for disk tracking is painful and ugly. But, that is what we have today.

Kubelet needs to define a gid for tracking image layers and make that gid or group the owner of `/var/lib/docker/[aufs | overlayfs]` recursively. Once this is done, the quota sub-system in the kernel will report the blocks being consumed by the storage driver on the underlying partition.

Since this number also includes the container’s writable layer, we will have to somehow subtract that usage from the overall usage of the storage driver directory. Luckily, we can use the same mechanism for tracking container’s writable layer. Once we apply a different `gid` to the container’s writable layer, which is located under `/var/lib/docker/<storage_driver>/diff/<container_id>`, the quota subsystem will not include the container’s writable layer usage.

Xfs on the other hand support project quota which lets us track disk usage of arbitrary directories using a project. Support for this feature in ext4 is being reviewed. So on xfs, we can use quota without having to clobber the writable layer's uid and gid.

**Pros**:

* Low overhead tracking provided by the kernel.


**Cons**

* Requires updates to default ownership on docker’s internal storage driver directories. We will have to deal with storage driver implementation details in any approach that is not docker native.

* Requires additional node configuration - quota subsystem needs to be setup on the node. This can either be automated or made a requirement for the node.

* Kubelet needs to perform gid management. A range of gids have to allocated to the kubelet for the purposes of quota management. This range must not be used for any other purposes out of band. Not required if project quota is available.

* Breaks `docker save` semantics. Since kubernetes assumes immutable images, this is not a blocker. To support quota in docker, we will need user-namespaces along with custom gid mapping for each container. This feature does not exist today. This is not an issue with project quota.

*Note: Refer to the [Appendix](#appendix) section more real examples on using quota with docker.*

**Project Quota**

Project Quota support for ext4 is currently being reviewed upstream. If that feature lands in upstream sometime soon, project IDs will be used to disk tracking instead of uids and gids.


##### Devicemapper

Devicemapper storage driver will setup two volumes, metadata and data, that will be used to store image layers and container writable layer. The volumes can be real devices or loopback. A Pool device is created which uses the underlying volume for real storage.

A new thinly-provisioned volume, based on the pool, will be created for running container’s.

The kernel tracks the usage of the pool device at the block device layer. The usage here includes image layers and container’s writable layers.

Since the kubelet has to track the writable layer usage anyways, we can subtract the aggregated root filesystem usage from the overall pool device usage to get the image layer’s disk usage.

Linux quota and `du` will not work with device mapper.

A docker dry run option (mentioned above) is another possibility.


#### Container Writable Layer

###### Overlayfs / Aufs

Docker creates a separate directory for the container’s writable layer which is then overlayed on top of read-only image layers.

Both the previously mentioned options of `du` and `Linux Quota` will work for this case as well.

Kubelet can use `du` to track usage and enforce `limits` once disk becomes a schedulable resource. As mentioned earlier `du` is resource intensive.

To use Disk quota, kubelet will have to allocate a separate gid per container. Kubelet can reuse the same gid for multiple instances of the same container (restart scenario). As and when kubelet garbage collects dead containers, the usage of the container will drop.

If local disk becomes a schedulable resource, `linux quota` can be used to impose `request` and `limits` on the container writable layer.
`limits` can be enforced using hard limits. Enforcing `request` will be tricky. One option is to enforce `requests` only when the disk availability drops below a threshold (10%). Kubelet can at this point evict pods that are exceeding their requested space. Other options include using `soft limits` with grace periods, but this option is complex.

###### Devicemapper

FIXME: How to calculate writable layer usage with devicemapper?

To enforce `limits` the volume created for the container’s writable layer filesystem can be dynamically [resized](https://jpetazzo.github.io/2014/01/29/docker-device-mapper-resize/), to not use more than `limit`. `request` will have to be enforced by the kubelet.


#### Container logs

Container logs are not storage driver specific. We can use either `du` or `quota` to track log usage per container. Log files are stored under `/var/lib/docker/containers/<container-id>`.

In the case of quota, we can create a separate gid for tracking log usage. This will let users track log usage and writable layer’s usage individually.

For the purposes of enforcing limits though, kubelet will use the sum of logs and writable layer.

In the future, we can consider adding log rotation support for these log files either in kubelet or via docker.


#### Volumes

The local disk based volumes map to a directory on the disk. We can use `du` or `quota` to track the usage of volumes.

There exists a concept called `FsGroup` today in kubernetes, which lets users specify a gid for all volumes in a pod. If that is set, we can use the `FsGroup` gid for quota purposes. This requires `limits` for volumes to be a pod level resource though.


### Yet to be explored

* Support for filesystems other than ext4 and xfs like `zfs`

* Support for Btrfs

It should be clear at this point that we need a plugin based model for disk accounting. Support for other filesystems both CoW and regular can be added as and when required. As we progress towards making accounting work on the above mentioned storage drivers, we can come up with an abstraction for storage plugins in general.


### Implementation Plan and Milestones

#### Milestone 1 - Get accounting to just work!

This milestone targets exposing the following categories of disk usage from the kubelet - infrastructure (images, sys daemons, etc), containers (log + writable layer) and volumes.

* `du` works today. Use `du` for all the categories and ensure that it works on both on aufs and overlayfs.

* Add device mapper support.

* Define a storage driver based pluggable disk accounting interface in cadvisor.

* Reuse that interface for accounting volumes in kubelet.

* Define a disk manager module in kubelet that will serve as a source of disk usage information for the rest of the kubelet.

* Ensure that the kubelet metrics APIs (/apis/metrics/v1beta1) exposes the disk usage information. Add an integration test.


#### Milestone 2 - node reliability

Improve user experience by doing whatever is necessary to keep the node running.

NOTE: [`Out of Resource Killing`](https://github.com/kubernetes/kubernetes/issues/17186) design is a prerequisite.

* Disk manager will evict pods and containers based on QoS class whenever the disk availability is below a critical level.

* Explore combining existing container and image garbage collection logic into disk manager.

Ideally, this phase should be completed before v1.2.


#### Milestone 3 - Performance improvements

In this milestone, we will add support for quota and make it opt-in. There should be no user visible changes in this phase.

* Add gid allocation manager to kubelet

* Reconcile gids allocated after restart.

* Configure linux quota automatically on startup. Do not set any limits in this phase.

* Allocate gids for pod volumes, container’s writable layer and logs, and also for image layers.

* Update the docker runtime plugin in kubelet to perform the necessary `chown’s` and `chmod’s` between container creation and startup.

* Pass the allocated gids as supplementary gids to containers.

* Update disk manager in kubelet to use quota when configured.


#### Milestone 4 - Users manage local disks

In this milestone, we will make local disk a schedulable resource.

* Finalize volume accounting - is it at the pod level or per-volume.

* Finalize multi-disk management policy. Will additional disks be handled as whole units?

* Set aside some space for image layers and rest of the infra overhead - node allocable resources includes local disk.

* `du` plugin triggers container or pod eviction whenever usage exceeds limit.

* Quota plugin sets hard limits equal to user specified `limits`.

* Devicemapper plugin resizes writable layer to not exceed the container’s disk `limit`.

* Disk manager evicts pods based on `usage` - `request` delta instead of just QoS class.

* Sufficient integration testing to this feature.


### Appendix


#### Implementation Notes

The following is a rough outline of the testing I performed to corroborate by prior design ideas.

Test setup information

* Testing was performed on GCE virtual machines

* All the test VMs were using ext4.

* Distribution tested against is mentioned as part of each graph driver.

##### AUFS testing notes:

Tested on Debian jessie

1. Setup Linux Quota following this [tutorial](https://www.google.com/url?q=https://www.howtoforge.com/tutorial/linux-quota-ubuntu-debian/&sa=D&ust=1446146816105000&usg=AFQjCNHThn4nwfj1YLoVmv5fJ6kqAQ9FlQ).

2. Create a new group ‘x’ on the host and enable quota for that group

    1. `groupadd -g 9000 x`

    2. `setquota -g 9000 -a 0 100 0 100` // 100 blocks (4096 bytes each*)

    3. `quota -g 9000 -v` // Check that quota is enabled

3. Create a docker container

    4. `docker create -it busybox /bin/sh -c "dd if=/dev/zero of=/file count=10 bs=1M"`

			8d8c56dcfbf5cda9f9bfec7c6615577753292d9772ab455f581951d9a92d169d

4. Change group on the writable layer directory for this container

    5. `chmod a+s /var/lib/docker/aufs/diff/8d8c56dcfbf5cda9f9bfec7c6615577753292d9772ab455f581951d9a92d169d`

    6. `chown :x /var/lib/docker/aufs/diff/8d8c56dcfbf5cda9f9bfec7c6615577753292d9772ab455f581951d9a92d169d`

5. Start the docker container

    7. `docker start 8d`

    8. Check usage using quota and group ‘x’

		```shell
			$ quota -g x -v

			Disk quotas for group x (gid 9000): 

			Filesystem  **blocks**   quota   limit   grace   files   quota   limit   grace

			/dev/sda1   **10248**       0       0               3       0       0
		```

	Using the same workflow, we can add new sticky group IDs to emptyDir volumes and account for their usage against pods.

	Since each container requires a gid for the purposes of quota, we will have to reserve ranges of gids  for use by the kubelet. Since kubelet does not checkpoint its state, recovery of group id allocations will be an interesting problem. More on this later.

Track the space occupied by images after it has been pulled locally as follows.

*Note: This approach requires serialized image pulls to be of any use to the kubelet.*

1. Create a group specifically for the graph driver

    1. `groupadd -g 9001 docker-images`

2. Update group ownership on the ‘graph’ (tracks image metadata) and ‘storage driver’ directories.

    2. `chown -R :9001 /var/lib/docker/[overlay | aufs]`

    3. `chmod a+s /var/lib/docker/[overlay | aufs]`

    4. `chown -R :9001 /var/lib/docker/graph`

    5. `chmod a+s /var/lib/docker/graph`

3. Any new images pulled or containers created will be accounted to the `docker-images` group by default.

4. Once we update the group ownership on newly created containers to a different gid, the container writable layer’s specific disk usage gets dropped from this group.

#### Overlayfs

Tested on Ubuntu 15.10.

Overlayfs works similar to Aufs. The path to the writable directory for container writable layer changes.

* Setup Linux Quota following this [tutorial](https://www.google.com/url?q=https://www.howtoforge.com/tutorial/linux-quota-ubuntu-debian/&sa=D&ust=1446146816105000&usg=AFQjCNHThn4nwfj1YLoVmv5fJ6kqAQ9FlQ).

* Create a new group ‘x’ on the host and enable quota for that group

    * `groupadd -g 9000 x`

    * `setquota -g 9000 -a 0 100 0 100` // 100 blocks (4096 bytes each*)

    * `quota -g 9000 -v` // Check that quota is enabled

* Create a docker container

    * `docker create -it busybox /bin/sh -c "dd if=/dev/zero of=/file count=10 bs=1M"`

        * `b8cc9fae3851f9bcefe922952b7bca0eb33aa31e68e9203ce0639fc9d3f3c61`

* Change group on the writable layer’s directory for this container

    * `chmod -R a+s  /var/lib/docker/overlay/b8cc9fae3851f9bcefe922952b7bca0eb33aa31e68e9203ce0639fc9d3f3c61b/*`

    * `chown -R :9000 /var/lib/docker/overlay/b8cc9fae3851f9bcefe922952b7bca0eb33aa31e68e9203ce0639fc9d3f3c61b/*`

* Check quota before and after running the container.

    ```shell
	   $ quota -g x -v

		Disk quotas for group x (gid 9000): 

		Filesystem  blocks   quota   limit   grace   files   quota   limit   grace

       /dev/sda1      48       0       0              19       0       0
   ```

    * Start the docker container

        * `docker start b8`

    * ```shell
	  quota -g x -v

	Disk quotas for group x (gid 9000):

     Filesystem  **blocks**   quota   limit   grace   files   quota   limit   grace

    /dev/sda1   **10288**       0       0                20      0       0

	```

##### Device mapper

Usage of Linux Quota should be possible for the purposes of volumes and log files.

Devicemapper storage driver in docker uses ["thin targets"](https://www.kernel.org/doc/Documentation/device-mapper/thin-provisioning.txt). Underneath there are two block devices devices - “data” and “metadata”, using which more block devices are created for containers. More information [here](http://www.projectatomic.io/docs/filesystems/).

These devices can be loopback or real storage devices.

The base device has a maximum storage capacity. This means that the sum total of storage space occupied by images and containers cannot exceed this capacity.

By default, all images and containers are created from an initial filesystem with a 10GB limit. 

A separate filesystem is created for each container as part of start (not create).

It is possible to [resize](https://jpetazzo.github.io/2014/01/29/docker-device-mapper-resize/) the container filesystem.  

For the purposes of image space tracking, we can 

####Testing notes:

* ```shell
$ docker info

...

Storage Driver: devicemapper

 Pool Name: **docker-8:1-268480-pool**

 Pool Blocksize: 65.54 kB

 Backing Filesystem: extfs

 Data file: /dev/loop0

 Metadata file: /dev/loop1

 Data Space Used: 2.059 GB

 Data Space Total: 107.4 GB

 Data Space Available: 48.45 GB

 Metadata Space Used: 1.806 MB

 Metadata Space Total: 2.147 GB

 Metadata Space Available: 2.146 GB

 Udev Sync Supported: true

 Deferred Removal Enabled: false

 Data loop file: /var/lib/docker/devicemapper/devicemapper/data

 Metadata loop file: /var/lib/docker/devicemapper/devicemapper/metadata

 Library Version: 1.02.99 (2015-06-20)
```

```shell
$ dmsetup table docker-8\:1-268480-pool 

0 209715200 thin-pool 7:1 7:0 **128** 32768 1 skip_block_zeroing
```

128 is the data block size

Usage from kernel for the primary block device

```shell
$ dmsetup status docker-8\:1-268480-pool 

0 209715200 thin-pool 37 441/524288 **31424/1638400** - rw discard_passdown queue_if_no_space -
```

Usage/Available - 31424/1638400

Usage in MB = 31424 * 512 * 128 (block size from above) bytes = 1964 MB

Capacity in MB = 1638400 * 512 * 128 bytes = 100 GB

#### Log file accounting

* Setup Linux quota for a container as mentioned above.

* Update group ownership on the following directories to that of the container group ID created for graphing. Adapting the examples above:

    * `chmod -R a+s  /var/lib/docker/**containers**/b8cc9fae3851f9bcefe922952b7bca0eb33aa31e68e9203ce0639fc9d3f3c61b/*`

    * `chown -R :9000 /var/lib/docker/**container**/b8cc9fae3851f9bcefe922952b7bca0eb33aa31e68e9203ce0639fc9d3f3c61b/*`

##### Testing titbits

* Ubuntu 15.10 doesn’t ship with the quota module on virtual machines. [Install ‘linux-image-extra-virtual’](http://askubuntu.com/questions/109585/quota-format-not-supported-in-kernel) package to get quota to work.

* Overlay storage driver needs kernels >= 3.18. I used Ubuntu 15.10 to test Overlayfs.

* If you use a non-default location for docker storage, change `/var/lib/docker` in the examples to your storage location.





<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/disk-accounting.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
