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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Running Kubelet in a Chroot

Authors: Vishnu Kannan \<vishh@google.com\>, Euan Kemp \<euan.kemp@coreos.com\>, Brandon Philips \<brandon.philips@coreos.com\>

## Introduction

The Kubelet is a critical component of Kubernetes that must be run on every node in a cluster.

However, right now it's not always easy to run it *correctly*. The Kubelet has
a number of dependencies that must exist in its filesystem, including various
mount and network utilities. Missing any of these can lead to unexpected
differences between Kubernetes hosts. For example, the Google Container VM
image (GCI) is missing various mount commands even though the Kernel supports
those filesystem types. Similarly, CoreOS Linux intentionally doesn't ship with
many mount utilities or socat in the base image. Other distros have a related
problem of ensuring these dependencies are present and versioned appropriately
for the Kubelet.

In order to solve this problem, it's proposed that running the Kubelet in a
prepackaged chroot should be a supported, recommended, way of running a fully
functioning Kubelet.

## The Kubelet Chroot

The easiest way to express all filesystem dependencies of the Kubelet comprehensively is to ship a filesystem image and run the Kubelet within it. The [hyperkube image](../../cluster/images/hyperkube/) already provides such a filesystem.

Even though the hyperkube image is distributed as a container, this method of
running the Kubelet intentionally is using a chroot and is neither a container nor pod.

The kubelet chroot will essentially operate as follows:

```
container-download-and-extract gcr.io/google_containers/hyperkube:v1.4.0 /path/to/chroot
mount --make-shared /var/lib/kubelet
mount --rbind /var/lib/kubelet /path/to/chroot/var/lib/kubelet
# And many more mounts, omitted
...
chroot /path/to/kubelet /usr/bin/hyperkube kubelet
```

Note: Kubelet might need access to more directories on the host and we intend to identity mount all those directories into the chroot. A partial list can be found in the CoreOS kubelet-wrapper script.
This logic will also naturally be abstracted so it's no more difficult for the user to run the Kubelet.

Currently, the Kubelet does not need access to arbitrary paths on the host (as
hostPath volumes are managed entirely by the docker daemon process, including
SELinux context applying), so Kubelet makes no operations at those paths). This
will likely change in the future, at which point a shared bindmount of `/` will
be made available at a known path in the Kubelet chroot. This change will
necessarily be more intrusive.

## Current Use

This method of running the Kubelet is already in use by users of CoreOS Linux. The details of this implementation are found in the [kubelet wrapper documentation](https://coreos.com/kubernetes/docs/latest/kubelet-wrapper.html).

## Implementation

### Target Distros

The two distros which benefit the most from this change are GCI and CoreOS. Initially, these changes will only be implemented for those distros.

This work will also only initially target the GCE provider and `kube-up` method of deployment.

#### Hyperkube Image Packaging

The Hyperkube image is distributed as part of an official release to the `gcr.io/google_containers` registry, but is not included along with the `kube-up` artifacts used for deployment.

This will need to be remediated in order to complete this proposal.

### Testing & Rollout

In order to ensure the paths remain complete, e2e tests *must* be run against a
Kubelet operating in this manner as part of the submit queue.

To ensure that this feature does not unduly impact others, it will be added to
GCI, but gated behind a feature-flag for a sort confidence-building period
(e.g.  `KUBE_RUN_HYPERKUBE_IMAGE=false`). A temporary non-blocking e2e job will
be added with that option. If the results look clean after a week, the
deployment option can be removed and the GCI image can completely switch over.

Once that testing is in place, it can be rolled out across other distros as
desired.


#### Everything else

In the initial implementation, rkt or docker can be used to extract the rootfs of the hyperkube image. rkt fly or a systemd unit (using [`RootDirectory`](https://www.freedesktop.org/software/systemd/man/systemd.exec.html#RootDirectory=)) can be used to perform the needed setup, chroot, and execution of the kubelet within that rootfs.



## FAQ

#### Will this replace or break other installation options?

Other installation options include using RPMs, DEBs, and simply running the statically compiled Kubelet binary.

All of these methods will continue working as they do now. In the future they may choose to also run the kubelet in this manner, but they don't necessarily have to.


#### Is this running the kubelet as a pod?

This is different than running the Kubelet as a pod. Rather than using namespaces, it uses only a chroot and shared bind mounts.

## Alternatives

#### Container + Shared bindmounts

Instead of using a chroot with shared bindmounts, a proper pod or container could be used if the container supported shared bindmounts.

This introduces some additional complexity in requiring something more than just the bare minimum. It also relies on having a container runtime available and puts said runtime in the critical path for the Kubelet.

#### "Dependency rootfs" aware kubelet

The Kubelet could be made aware of the rootfs containing all its dependencies, but not chrooted into it (e.g. started with a `--dependency-root-dir=/path/to/extracted/container` flag).

The Kubelet could then always search for the binary it wishes to run in that path first and prefer it, as well as preferring libraries in that path. It would effectively run all dependencies similar to the following:

```bash
export PATH=${dep_root}/bin:${dep_root}/usr/bin:...
export LD_LIBRARY_PATH=${dep_root}/lib:${dep_root}/usr/lib:...
# Run 'mount':
$ ${dep_root}/lib/x86_64-linux-gnu/ld.so --inhibit-cache mount $args
```

**Downsides**:

This adds significant complexity and, due to the dynamic library hackery, might require some container-specific knowledge of the Kubelet or a rootfs of a predetermined form.

This solution would also have to still solve the packaging of that rootfs, though the solution would likely be identical to the solution for distributing the chroot-kubelet-rootfs.

#### Waiting for Flexv2 + port-forwarding changes

The CRI effort plans to change how [port-forward](https://github.com/kubernetes/kubernetes/issues/29579) works, towards a method which will not depend explicitly on socat or other networking utilities.

Similarly, for the mount utilities, the [Flex Volume v2](https://github.com/kubernetes/features/issues/93) feature is aiming to solve this utility.


**Downsides**:

This requires waiting on other features which might take a signficant time to land. It also could end up not fully fixing the problem (e.g. pushing down port-forwarding to the runtime doesn't ensure the runtime doesn't rely on host utilities).

The Flex Volume feature is several releases out from fully replacing the current volumes as well.

Finally, it's likely there are dependencies that neither of these proposals cover.

## Non-Alternatives

#### Pod + containerized flag

Currently, there's a `--containerized` flag. This flag doesn't actually remove the dependency on mount utilities on the node though, so does not solve the problem described here. It also is under consideration for [removal](https://issues.k8s.io/18776).

## Open Questions

#### Why not a mount namespace?

#### Timeframe

1.6?


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubelet-rootfs-distribution.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
