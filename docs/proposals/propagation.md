# HostPath Volume Propagation

## Abstract

A proposal to add support for propagation mode in HostPath volume, which allows
mounts within containers to visible outside the container and mounts after pods
creation visible to containers. Propagation [modes] (https://www.kernel.org/doc/Documentation/filesystems/sharedsubtree.txt) contains "shared", "slave", "private",
"unbindable". Out of them, docker supports "shared" / "slave" / "private".

Several existing issues and PRs were already created regarding that particular
subject:
* Capability to specify mount propagation mode of per volume with docker [#20698] (https://github.com/kubernetes/kubernetes/pull/20698)
* Set propagation to "shared" for hostPath volume [#31504] (https://github.com/kubernetes/kubernetes/pull/31504)

## Use Cases

1. (From @Kaffa-MY) Our team attempts to containerize flocker with zfs as back-end
storage, and launch them in DaemonSet. Containers in the same flocker node need
to read/write and share the same mounted volume. Currently the volume mount
propagation mode cannot be specified between the host and the container, and then
the volume mount of each container would be isolated from each other.
This use case is also referenced by Containerized Volume Client Drivers - Design
Proposal [#22216] (https://github.com/kubernetes/kubernetes/pull/22216)

1. (From @majewsky) I'm currently putting the [OpenStack Swift object storage] (https://github.com/openstack/swift) into
k8s on CoreOS. Swift's storage services expect storage drives to be mounted at
/srv/node/{drive-id} (where {drive-id} is defined by the cluster's ring, the topology
description data structure which is shared between all cluster members). Because
there are several such services on each node (about a dozen, actually), I assemble
/srv/node in the host mount namespace, and pass it into the containers as a hostPath
volume.
Swift is designed such that drives can be mounted and unmounted at any time (most
importantly to hot-swap failed drives) and the services can keep running, but if
the services run in a private mount namespace, they won't see the mounts/unmounts
performed on the host mount namespace until the containers are restarted.
The slave mount namespace is the correct solution for this AFAICS. Until this
becomes available in k8s, we will have to have operations restart containers manually
based on monitoring alerts.

1. (From @victorgp) When using CoreOS that does not provides external fuse systems
like, in our case, GlusterFS, and you need a container to do the mounts. The only
way to see those mounts in the host, hence also visible by other containers, is by
sharing the mount propagation.

1. (From @YorikSar) For OpenStack project, Neutron, we need network namespaces
created by it to persist across reboot of pods with Neutron agents. Without it
we have unnecessary data plane downtime during rolling update of these agents.
Neutron L3 agent creates interfaces and iptables rules for each virtual router
in a separate network namespace. For managing them it uses ip netns command that
creates persistent network namespaces by calling unshare(CLONE_NEWNET) and then
bind-mounting new network namespace's inode from /proc/self/ns/net to file with
specified name in /run/netns dir. These bind mounts are the only references to
these namespaces that remain.
When we restart the pod, its mount namespace is destroyed with all these bind
mounts, so all network namespaces created by the agent are gone. For them to
survive we need to bind mount a dir from host mount namespace to container one
with shared flag, so that all bind mounts are propagated across mount namespaces
and references to network namespaces persist.


## Implementation Alternatives

### Add an option in VolumeMount API

The new `VolumeMount` will look like:

```go
type VolumeMount struct {
	// Required: This must match the Name of a Volume [above].
	Name string `json:"name"`
	// Optional: Defaults to false (read-write).
	ReadOnly bool `json:"readOnly,omitempty"`
	// Required.
	MountPath string `json:"mountPath"`
	// Optional.
	Propagation string `json:"propagation"`
}
```

Opinion against this:

1. This will affect all volumes, while only HostPath need this.

1. This need API change, which is discouraged.

### Add an option in HostPathVolumeSource

The new `HostPathVolumeSource` will look like:

```go
const (
	PropagationShared  PropagationMode = "Shared"
	PropagationSlave   PropagationMode = "Slave"
	PropagationPrivate PropagationMode = "Private"
)

type HostPathVolumeSource struct {
	Path string `json:"path"`
	// Mount the host path with propagation mode specified. Docker only.
	Propagation PropagationMode `json:"propagation,omitempty"`
}
```

Opinion against this:

1. This need API change, which is discouraged.

1. All containers use this volume will share the same propagation mode.

1. (From @jonboulle) May cause cross-runtime compatibility issue.

### Make HostPath shared for privileged containers, slave for non-privileged.

Given only HostPath needs this feature, and CAP_SYS_ADMIN access is needed when
making mounts inside container, we can bind propagation mode with existing option
privileged, or we can introduce a new option in SecurityContext to control this.

The propagation mode could be determined by the following logic:

```go
// Environment check to ensure "rshared" is supported.
if !dockerNewerThanV110 || !mountPathIsShared {
	return ""
}
if container.SecurityContext.Privileged {
	return "rshared"
} else {
	return "rslave"
}
```

Opinion against this:

1. This changes the behavior of existing config.

1. (From @euank) "shared" is not correctly supported by some kernels, we need
runtime support matrix and when that will be addressed.

1. This may cause silently fail and be a debuggability nightmare on many
distros.

1. (From @euank) Changing those mountflags may make docker even less stable,
this may lock up kernel accidently or potentially leak mounts.


## Decision

We will take 'Make HostPath shared for privileged containers, slave for
non-privileged', an environment check and an WARNING log will be emitted about
whether propagation mode is supported.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/propagation.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
