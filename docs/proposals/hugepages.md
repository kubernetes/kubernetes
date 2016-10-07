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

# Huge Page Support

## Abstract

A proposal to allow huge page use by applications running in a Kubernetes
cluster.

A pod should be able to have a number of huge pages for use by the
application.  The scheduler should be able have visibility into the node
capacity of huge pages, for each huge page size, and make a decision about if
the pod can be scheduled on that node.  The kubelet should report the number of
available huge pages and set up the environment such that the pod can
successfully use the number of huge pages requested in the pod definition.

## Motivation

Huge page support is needed for many large memory HPC workloads to achieve
acceptable performance levels.

This proposal is part of a larger effort to better support High Performance
Computing (HPC) workloads in Kubernetes.

### Scope

This proposal only includes pre-allocated huge pages configured on the node by
the administrator at boot time or by manual dynamic allocation.  It does not
discuss the kubelet attempting to allocate huge pages dynamically in an attempt
to accommodate a scheduling pod or the use of Transparent Huge Pages (THP). THP
do not require knowledge by Kubernetes at all, but simply requires the node to
have THP enabled and the application `madvise()` with `MADV_HUGEPAGES`
memory regions it desires to be backed by huge pages.  Note that THP might lead
to performance degradation on nodes with high memory utilization or
fragmentation due to the defragmenting efforts of THP, which can lock memory
pages.  For this reason, some applications may be designed to (or recommend) use
pre-allocated huge pages instead of THP.

The proposal is also limited to x86_64 support where two huge page sizes are
supported: 2MB and 1GB.  The design, however, should accommodate additional huge
page sizes available on other architectures.

**NOTE: This design, as currently proposed, requires the use of pod-level
cgroups, which are currently not available but are under development by
@derekwaynecarr**

## Background

Huge pages are a hardware feature designed to reduce pressure on the Translation
Lookaside Buffer (TLB).  The TLB is a small hardware cache of
virtual-to-physical page mappings.  If the virtual address passed in a hardware
instruction can be found in the TLB, the mapping can be determined quickly.  If
not, a TLB miss occurs and the hardware must walk the in-memory page table to
discover a physical mapping for the virtual address.

Take a program that operates on a large 2MB structure as an example.  If the
program accesses that space in such a way that one byte in each regular 4k page
is accessed, 2MB/4kB = 512 TLB entries are needed to map the address range.  Each
TLB miss results in an expensive walk of the page table.  However, if the
allocation is backed by a 2MB huge page, only 1 TLB entry is required
resulting in a highly likelihood that entry will remain in the cache and hit on
accesses to the entire 2MB structure.

On x86_64, there are two huge page sizes: 2MB and 1GB.  1GB huge pages are also
called gigantic pages.  1GB must be enabled on kernel boot line with
`hugepagesz=1g`. Huge pages, especially 1GB ones, should to be allocated
early before memory fragments (i.e. at/near boot time) to increase the
likelihood that they can be allocated successfully with minimal memory migration
(i.e. defrag) required.

## Use Cases

The class of applications that benefit from huge pages typically have
- A large memory working set
- A sensitivity to memory access latency

Example applications include:
- Java applications can back the heap with huge pages using the `-XX:+UseLargePages` option.
- In-memory databases

Applications can generally use huge pages by calling
- `mmap()` with `MAP_ANONYMOUS | MAP_HUGETLB` and use it as anonymous memory
- `shmget()` with `SHM_HUGETLB` and use it as a shared memory segment (see Known Issues).

### Pod Specification

```
apiVersion: v1
kind: Pod
metadata:
  name: example
spec:
  containers:
...
    resources:
      requests:
	    hugepages: "10"
      limits:
	    hugepages: "10"
  nodeSelector:
    kubernetes.io/huge-page-size: "2MB"
```

Huge pages can not be overcommitted on a node.

While a system may support multiple huge pages sizes, it is assumed that nodes
configured with huge pages will only use one huge page size, namely the default
page size in `cat /proc/meminfo | grep Hugepagesize`.  In Linux, this is 2MB
unless overridden by `default_hugepagesz=1g` in the kernel boot parameters.

The huge page size for the node will be reported by the kubelet as a label
`alpha.kubernetes.io/huge-page-size` on the node resource.  This is done
because there are a variety of huge page sizes across different hardware
architecture and making a new resource field for each size doesn't scale.  Pods
can do a nodeSelector on this label to land on a system with a particular huge
page size.  This is similiar to how the `beta.kubernetes.io/arch` label
operates.

## Limits and Quota

LimitRange should be able to define minimum and maximum constraints for huge
pages, and Quota should be able to count them.

## Implementation

### Phase 0: Design Agreement

**Target 1.5**

Get design approval

### Phase 1: Add huge page support

**Target 1.5+**

Implement huge page support with pod-level cgroups to enforce per-pod huge page
limits (not yet available).  Enforcing huge page limits with pod-level cgroups
avoids, at least temporarily, the need for 1) `docker` to support the
`hugetlb` cgroup controller directly and 2) adding huge pages to the
Container Runtime Interface (CRI)

pkg/api/types.go (and v1/types.go)

```
const (
	// CPU, in cores. (500m = .5 cores)
	ResourceCPU ResourceName = "cpu"
	// Memory, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	ResourceMemory ResourceName = "memory"
...
	ResourceHugePages ResourceName = "hugepages"
)
```

The kubelet will report the total/available huge pages statistics from cadvisor
in the node status as the huge page capacity/available.

Modifications needed `setNodeStatusMachineInfo` in `pkg/kubelet/kubelet_node_status.go`
and `CapacityFromMachineInfo` in `pkg/kubelet/cadvisor/util.go`.

The kubelet will also need to create the `alpha.kubernetes.io/huge-page-size`
label for its node resource (if self registering).

pkg/api/unversioned/well_known_labels.go

```
const (
...
	LabelArch = "beta.kubernetes.io/arch"
	LabelHugePageSize = "alpha.kubernetes.io/huge-page-size"
)
```

The scheduler will need to ensure any huge page request defined in the pod spec can be fulfilled by a candidate node.

cAdvisor will need to be modified to return the number of available huge pages.
This is already supported in [runc/libcontainer](../../vendor/github.com/opencontainers/runc/libcontainer/cgroups/utils.go)

### Phase 2: Expose huge pages in CRI

*WIP*

info/v1/machine.go

```
type MachineInfo struct {
...
	HugePages int `json:"huge_pages"`
}
```

Add `hugetlb` cgroup controller support to docker (TODO: add docker/docker issue/PR) and expose it via the engine-api

engine-api/types/container/host_config.go

```
type Resources struct {
...
	Ulimits              []*units.Ulimit // List of ulimits to be set in the container
	HugePages            int64 // Huge pages limit
...
}
```

## Known Issues

### Huge pages as shared memory

For the Java use case the JVM maps the huge pages as a shared memory segment and
memlocks them to prevent the system from moving or swapping them out.

There are several issues here:
- The user running the Java app must be a member of the gid set in the `vm.huge_tlb_shm_group` sysctl
- sysctl `kernel.shmmax` must allow the size of the shared memory segment
- The user's memlock ulimits must allow the size of the shared memory segment

`vm.huge_tlb_shm_group` is not namespaced.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/hugepages.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
