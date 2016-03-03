<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/admin/garbage-collection.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Garbage Collection

- [Introduction](#introduction)
- [Image Collection](#image-collection)
- [Container Collection](#container-collection)
- [User Configuration](#user-configuration)

### Introduction

Garbage collection is a helpful function of kubelet that will clean up unreferenced images and unused containers. kubelet will perform garbage collection for containers every minute and garbage collection for images every five minutes.

External garbage collection tools are not recommended as these tools can potentially break the behavior of kubelet by removing containers expected to exist.

### Image Collection

Kubernetes manages lifecycle of all images through imageManager, with the cooperation
of cadvisor.

The policy for garbage collecting images takes two factors into consideration:
`HighThresholdPercent` and `LowThresholdPercent`. Disk usage above the the high threshold
will trigger garbage collection. The garbage collection will delete least recently used images until the low
threshold has been met.

### Container Collection

The policy for garbage collecting containers considers three user-defined variables. `MinAge` is the minimum age at which a container can be garbage collected. `MaxPerPodContainer` is the maximum number of dead containers any single
pod (UID, container name) pair is allowed to have. `MaxContainers` is the maximum number of total dead containers. These variables can be individually disabled by setting 'Min Age' to zero and setting 'MaxPerPodContainer' and 'MaxContainers' respectively to less than zero.

Kubelet will act on containers that are unidentified, deleted, or outside of the boundaries set by the previously mentioned flags. The oldest containers will generally be removed first. 'MaxPerPodContainer' and 'MaxContainer' may potentially conflict with each other in situations where retaining the maximum number of containers per pod ('MaxPerPodContainer') would go outside the allowable range of global dead containers ('MaxContainers'). 'MaxPerPodContainer' would be adjusted in this situation: A worst case scenario would be to downgrade 'MaxPerPodContainer' to 1 and evict the oldest containers. Additionally, containers owned by pods that have been deleted are removed once they are older than `MinAge`.

Containers that are not managed by kubelet are not subject to container garbage collection.

### User Configuration

Users can adjust the following thresholds to tune image garbage collection with the following kubelet flags :

1. `image-gc-high-threshold`, the percent of disk usage which triggers image garbage collection.
Default is 90%.
2. `image-gc-low-threshold`, the percent of disk usage to which image garbage collection attempts
to free. Default is 80%.

We also allow users to customize garbage collection policy through the following kubelet flags:

1. `minimum-container-ttl-duration`, minimum age for a finished container before it is
garbage collected. Default is 1 minute.
2. `maximum-dead-containers-per-container`, maximum number of old instances to retain
per container. Default is 2.
3. `maximum-dead-containers`, maximum number of old instances of containers to retain globally.
Default is 100.

Containers can potentially be garbage collected before their usefulness has expired. These containers can contain logs and other data that can be useful for troubleshooting. A sufficiently large value for `maximum-dead-containers-per-container` is highly recommended to allow at least 2 dead containers to be retained per expected container. A higher value for `maximum-dead-containers` is also recommended for a similiar reason.
See [this issue](https://github.com/kubernetes/kubernetes/issues/13287) for more details.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/garbage-collection.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
