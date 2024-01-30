## Purpose

This repository contains the definitions for the Container Runtime Interface (CRI).
CRI is a plugin interface which enables kubelet to use a wide variety of container runtimes,
without the need to recompile. CRI consists of a protocol buffers and gRPC API.
Read more about CRI API at [kubernetes docs](https://kubernetes.io/docs/concepts/architecture/cri/).

The repository [kubernetes/cri-api](https://github.com/kubernetes/cri-api) is a mirror of https://github.com/kubernetes/kubernetes/tree/master/staging/src/k8s.io/cri-api.
Please do **not** file issues or submit PRs against the [kubernetes/cri-api](https://github.com/kubernetes/cri-api)
repository as it is readonly, all development is done in [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).

The CRI API is defined in [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes)
repository and is **only** intended to be used for kubelet to container runtime 
interactions, or for node-level troubleshooting using a tool such as `crictl`.
It is **not** a common purpose container runtime API for general use, and is intended
to be Kubernetes-centric. We try to avoid it, but there may be logic within a container
runtime that optimizes for the order or specific parameters of call(s) that the kubelet
makes.

## Version skew policy

On a single Node there may be installed multiple components implementing
different versions of CRI API.

For example, on a single node there might be:

- _Kubelet_ may call into _Container Runtime_ (e.g. [containerd](https://containerd.io))
  and _Image Service Proxy_ (e.g. [stargz-snapshotter](https://github.com/containerd/stargz-snapshotter)).
  _Container Runtime_ may be versioned with the OS Image, _Kubelet_ is installed
  by system administrator and _Image Service proxy_ is versioned by the third party vendor.
- _Image Service Proxy_ calls into _Container Runtime_.
- _CRI tools_ (e.g. [crictl](https://kubernetes.io/docs/tasks/debug/debug-cluster/crictl/))
  may be installed by end user to troubleshoot, same as a third party daemonsets.
  All of them are used to call into the _Container Runtime_ to collect container information.

So on a single node it may happen that _Container Runtime_ is serving a newer
version'd kubelet and older versioned crictl. This is a supported scenario within
the version skew policy.

### Version Skew Policy for CRI API

CRI API has two versions:
- Major semantic version (known versions are `v1alpha2` ([removed in 1.26](https://kubernetes.io/blog/2022/12/09/kubernetes-v1-26-release/#cri-v1alpha2-removed)), `v1`).
- Kubernetes version (for example: `@1.23`). Note, the `cri-api` Golang library is versioned as `0.23` as it doesn't guarantee Go types backward compatibility.

Major semantic version (e.g. `v1`) is used to introduce breaking changes
and major new features that are incompatible with the current API.

Kubernetes version is used to indicate a specific feature set implemented
on top of the major semantic version. All changes made without the change
of a major semantic version API must be backward and forward compatible.

- _Kubelet_ must work with the older _Container Runtime_ if it implements
  the same semantic version of CRI API (e.g. `v1`) of up to three Kubernetes minor
  versions back. New features implemented in CRI API must be gracefully degraded.
  For example, _Kubelet_ of version 1.26 must work with _Container Runtime_
  implementing `k8s.io/cri-api@v0.23.0`+.
- _Kubelet_ must work with _Container Runtime_ if it implements
  the same semantic version of CRI API (e.g. `v1`) of up to
  three minor versions up. New features implemented in CRI API must not change
  behavior of old method calls and response values. For example, _Kubelet_ of
  version 1.22 must work with _Container Runtime_ implementing `k8s.io/cri-api@v0.25.5`.


## Versioning

This library contains go classes generated from the CRI API protocol buffers and gRPC API.

The library versioned as `0.XX` as Kubernetes doesn't provide any guarantees
on backward compatibility of Go wrappers between versions. However CRI API itself
(protocol buffers and gRPC API) is marked as stable `v1` version and it is
backward compatible between versions.

Versions like `v0.<minor>.<patch>` (e.g. `v0.25.5`) are considered stable.
It is discouraged to introduce CRI API changes in patch releases and recommended
to use versions like `v0.<minor>.0`.

All alpha and beta versions (e.g. `k8s.io/cri-api@v0.26.0-beta.0`) should be
backward and forward compatible.

## Feature development

Some features development requires changes in CRI API and corresponding changes
in _Container Runtime_. Coordinating between Kubernetes branches and release
versions and _Container Runtime_ versions is not always trivial.

The recommended feature development flow is following:

- Review proposed CRI API changes during the KEP review stage.
  Some field names and types may not be spelled out exactly at this stage.
- Locally implement a prototype that implement changes in both - Kubernetes and Container Runtime.
- Submit a Pull Request for Kubernetes implementing CRI API changes alongside the feature code.
  Feature must be developed to degrade gracefully when used with older Container Runtime
  according to the Version Skew policy.
- Once PR is merged, wait for the next Kubernetes release tag being produced.
  Find the corresponding CRI API tag (e.g. `k8s.io/cri-api@v0.26.0-beta.0`).
- This tag can be used to implement the feature in Container Runtime. It is recommended
  to switch to the stable tag like (`k8s.io/cri-api@v0.26.0`) once available.

## Change history

Here is the change history of the Container Runtime Interface protocol:

### v1.20

`git diff v1.19.0 v1.20.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- CRI [v1 introduced](https://github.com/kubernetes/kubernetes/pull/96387)

### v1.21

`git diff v1.20.0 v1.21.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

No changes

### v1.22

`git diff v1.21.0 v1.22.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [Windows host process support](https://github.com/kubernetes/kubernetes/pull/99576)
  - `PodSandboxConfig` has `windows` field  of type `WindowsPodSandboxConfig`
  - New type `WindowsPodSandboxConfig` introduced
  - New type `WindowsSandboxSecurityContext` introduced
  - The type `WindowsContainerSecurityContext` has a new `host_process` boolean field

- [Feature: add unified on CRI to support cgroup v2](https://github.com/kubernetes/kubernetes/pull/102578)
 - The type `LinuxContainerResources` has a new field `unified` which is a map of strings

- [Alpha node swap support](https://github.com/kubernetes/kubernetes/pull/102823)
  - The type `LinuxContainerResources` has a new `memory_swap_limit_in_bytes` int64 field

### v1.23

`git diff v1.22.0 v1.23.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [CRI: add fields for pod level stats to satisfy the /stats/summary API](https://github.com/kubernetes/kubernetes/pull/102789)
  - New functions `PodSandboxStats`, `ListPodSandboxStats` with the corresponding types of request and response objects are introduced

- [pass sandbox resource requirements over CRI](https://github.com/kubernetes/kubernetes/pull/104886)
  - New fields on `LinuxPodSandboxConfig`: `overhead` and `resources` of type `LinuxContainerResources`.

- [prevents garbage collection from removing pinned images](https://github.com/kubernetes/kubernetes/pull/103299)
  - The type `Image` has a new boolean field `pinned`

### v1.24

`git diff v1.23.0 v1.24.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [Update CRI-API Capabilities to include a field that allows us to set ambient capabilities](https://github.com/kubernetes/kubernetes/pull/104620)
  - The type `Capability` has a new string field `add_ambient_capabilities`

- [CRI-API - Add rootfs size to WindowsContainerResources](https://github.com/kubernetes/kubernetes/pull/108894)
  - The type `WindowsContainerResources` has a new int64 field `rootfs_size_in_bytes`

### v1.25

`git diff v1.24.0 v1.25.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`


- [kubelet: add CRI definitions for user namespaces](https://github.com/kubernetes/kubernetes/pull/110535)
  - The new type `UserNamespace` introduced to represent user namespaces id mapping
  - The type `NamespaceOption` has a new field `userns_options` of type `UserNamespace`

- [Minimal checkpointing support](https://github.com/kubernetes/kubernetes/pull/104907)
  - The new method `CheckpointContainer` introduced with the corresponding request and response types

- [Update CRI API to support Evented PLEG](https://github.com/kubernetes/kubernetes/pull/111642)
  - The new streaming method `GetContainerEvents` is introduced with the corresponding request and response types

- [CRI changes to support in-place pod resize](https://github.com/kubernetes/kubernetes/pull/111645)
  - The new type `ContainerResources` is introduced
  - The type `ContainerStatus` has a new field `resources` of type `ContainerResources`
  - The semantic of `UpdateContainerResources` updated. The method must be implemented as synchronous and return error on failure

### v1.26

`git diff v1.25.0 v1.26.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [CRI: Add Windows Podsandbox Stats](https://github.com/kubernetes/kubernetes/pull/110754)
  - Added fields to the type `WindowsPodSandboxStats` expressing stats required to be collected from windows pods.

- [Windows hostnetwork alpha](https://github.com/kubernetes/kubernetes/pull/112961)
  - New type `WindowsNamespaceOption` introduced
  - The type `WindowsSandboxSecurityContext` has a new field `namespace_options` of type `WindowsNamespaceOption`

- [Improve the API description of `PodSecurityContext.SupplementalGroups` to clarify its unfamiliar behavior](https://github.com/kubernetes/kubernetes/pull/113047)
  - Clarified the expected behavior of `SupplementalGroups` field of `PodSecurityContext`

- [Add Support for Evented PLEG](https://github.com/kubernetes/kubernetes/pull/111384)
  - The type `ContainerEventResponse` updated: the field `pod_sandbox_metadata` removed and fields `pod_sandbox_status` and `containers_statuses` added.
  - The type `PodSandboxStatusResponse` has a new fields `containers_statuses` and `timestamp`

### v1.27

`git diff v1.26.0 v1.27.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [CRI: Add CDI device info for containers](https://github.com/kubernetes/kubernetes/pull/115891/)
  - New type `CDIDevice` was introduced and added to container config

- [Add mappings for volumes](https://github.com/kubernetes/kubernetes/pull/116377)
  - Added new fields to the type `Mount` expressing runtime UID/GID mappings for the mount.

### v1.28

`git diff v1.27.0 v1.28.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [cri-api: fix comment lines about PROPAGATION_PRIVATE](https://github.com/kubernetes/kubernetes/pull/115704)
  - Fixed comment lines about PROPAGATION_PRIVATE

- [Add user specified image to CRI ContainerConfig](https://github.com/kubernetes/kubernetes/pull/118652)
  - Added the `user_specified_image` field to type `ImageSpec`

- [kubelet: get cgroup driver config from CRI ](https://github.com/kubernetes/kubernetes/pull/118770)
  - Added rpc for querying runtime configuration
  - Added cavieats about cgroup driver field

- [Add swap to stats to Summary API and Prometheus endpoints (/stats/summary and /metrics/resource)](https://github.com/kubernetes/kubernetes/pull/118865)
  - Added `SwapUsage` type
  - Added `SwapUsage` field to `ContainerStats` type

- [Expose commit memory used in WindowsMemoryUsage struct](https://github.com/kubernetes/kubernetes/pull/119238)
  - Added the `commit_memory_bytes` field to type `WindowsMemoryUsage`

### v1.29

`git diff v1.28.0 v1.29.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [Add runtime handler field to ImageSpec struct](https://github.com/kubernetes/kubernetes/pull/121121)
  - Added `runtime_handler` field to type `ImageSpec`

## Community, discussion, contribution, and support

Learn how to engage with the Kubernetes community on the [community
page](http://kubernetes.io/community/).

You can reach the maintainers of this repository at:

- Slack: #sig-node (on https://kubernetes.slack.com -- get an
  invite at [slack.kubernetes.io](https://slack.kubernetes.io))
- Mailing List:
  https://groups.google.com/forum/#!forum/kubernetes-sig-node

### Code of Conduct

Participation in the Kubernetes community is governed by the [Kubernetes
Code of Conduct](code-of-conduct.md).

### Contribution Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information. Please note that [kubernetes/cri-api](https://github.com/kubernetes/cri-api)
is a readonly mirror repository, all development is done at [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).
