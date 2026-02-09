> ⚠️ **This is an automatically published [staged repository](https://git.k8s.io/kubernetes/staging#external-repository-staging-area) for Kubernetes**.   
> Contributions, including issues and pull requests, should be made to the main Kubernetes repository: [https://github.com/kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).  
> This repository is read-only for importing, and not used for direct contributions.  
> See [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

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

## Version skew policy and feature development

Please read about:

- [CRI API version skew policy](https://kubernetes.dev/docs/code/cri-api-version-skew-policy/)
- [Kubernetes feature development and container runtimes](https://kubernetes.dev/docs/code/cri-api-dev-policies/)

## Community, discussion, contribution, and support

Learn how to engage with the Kubernetes community on the [community
page](https://www.k8s.dev/community/).

You can reach the maintainers of this repository at:

- Slack: #sig-node (on https://kubernetes.slack.com -- get an
  invite at [slack.kubernetes.io](https://slack.kubernetes.io))
- Mailing List:
  https://groups.google.com/a/kubernetes.io/g/sig-node

Issues can be filed at https://github.com/kubernetes/kubernetes/issues. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Code of Conduct

Participation in the Kubernetes community is governed by the [Kubernetes
Code of Conduct](code-of-conduct.md).

### Contribution Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information. Please note that [kubernetes/cri-api](https://github.com/kubernetes/cri-api)
is a readonly mirror repository, all development is done at [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).

## Change history

Here is the change history of the Container Runtime Interface protocol. The change history is maintained manually:

### v1.35

`git diff v1.34.0 1.35.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

No changes

### v1.34

`git diff v1.33.0 v1.34.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- Removed the [gogo dependency](https://github.com/kubernetes/kubernetes/pull/128653)
- [Added `debug_redact` flags](https://github.com/kubernetes/kubernetes/pull/133135) to the following fields of `AuthConfig`: `password`, `auth`, `identity_token`, `registry_token`.

### v1.33

`git diff v1.32.0 v1.33.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [Clarify the behavior when the host_port value is set to 0 in CRI](https://github.com/kubernetes/kubernetes/pull/130512)
  - Added clarifying comment to the [host_port] field of [PortMapping].

- [\[KEP-4639\] Graduate image volume sources to beta](https://github.com/kubernetes/kubernetes/pull/130135)
  - Added `image_sub_path` to the  type `Mount` to represent the subpath inside the image to mount.

- [\[FG:InPlacePodVerticalScaling\] Add UpdatePodSandboxResources CRI method](https://github.com/kubernetes/kubernetes/pull/128123)
  - New method `UpdatePodSandboxResources` to synchronously updates the PodSandboxConfig with the pod-level resource configuration.
  - Added the `UpdatePodSandboxResourcesRequest` type to pass as an argument to `UpdatePodSandboxResources`.
  - Added the `UpdatePodSandboxResourcesResponse` empty type to return from the `UpdatePodSandboxResources`.

- [Withdraw alpha support for HostNetwork containers on Windows](https://github.com/kubernetes/kubernetes/pull/130250)
  - Added clarifying comment on the `network` field of the type `WindowsNamespaceOption` as HostNetwork containers are not supported.

- [Surface Pressure Stall Information (PSI) metrics](https://github.com/kubernetes/kubernetes/pull/130701)
  - Added `io` field to the types `LinuxPodSandboxStats` and `ContainerStats` to represent the IO usage.
  - Added  `psi` field to the type `CpuUsage`.
  - Added types `IoUsage`, `PsiStats`, and `PsiData` to represent IO usage and PSI statistics.

- [KEP 4960: Container Stop Signals](https://github.com/kubernetes/kubernetes/pull/130556)
  - Added the field `stop_signal` to the `ContainerConfig` type.
  - Added the enum `Signal` listing all possible stop signals.

### v1.32

`git diff v1.31.0 v1.32.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [CRI: Add field to support CPU affinity on Windows](https://github.com/kubernetes/kubernetes/pull/124285)
  - CRI field `affinity_cpus` to `WindowsContainerResources` struct to support CPU affinity on Windows.
    This field will be used by Windows CPU manager to set the logical processors to affinitize
    for a particular container down to containerd/hcsshim.

### v1.31

`git diff v1.30.0 v1.31.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [KEP-3619: Add NodeStatus.Features.SupplementalGroupsPolicy API and e2e](https://github.com/kubernetes/kubernetes/pull/125470)
  - Added `features` field to the type `StatusResponse` for the runtime to kubelet handshake on what features are supported

- [KEP-3619: Fine-grained SupplementalGroups control](https://github.com/kubernetes/kubernetes/pull/117842)
  - Added `supplemental_groups_policy` field to types `LinuxContainerSecurityContext` and `LinuxSandboxSecurityContext`
  - Added `user` field to the type `ContainerStatus` to represent actual user for the container 

- [[KEP-4639] Add OCI VolumeSource CRI API](https://github.com/kubernetes/kubernetes/pull/125659)
  - Added `image` field to the type `Mount` to represent the OCI VolumeSource 

### v1.30

`git diff v1.29.0 v1.30.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [Recursive Read-only (RRO) mounts](https://github.com/kubernetes/kubernetes/pull/123272)
  - Added RuntimeHandler and RuntimeHandlerFeatures type
  - Added `recursive_read_only` field to type `Mount`
  - Added `runtime_handlers` field to type `StatusResponse`

- [Add user_namespaces field to RuntimeHandlerFeatures](https://github.com/kubernetes/kubernetes/pull/123356)
  - Added `user_namespaces` field to type `RuntimeHandlerFeatures`

- [Add image_id to CRI Container message](https://github.com/kubernetes/kubernetes/pull/123508)
  - Added `image_id` field to type `Container`

- [Add image_id to CRI ContainerStatus message](https://github.com/kubernetes/kubernetes/pull/123583)
  - Added `image_id` field to type `ContainerStatus`

### v1.29

`git diff v1.28.0 v1.29.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [Add runtime handler field to ImageSpec struct](https://github.com/kubernetes/kubernetes/pull/121121)
  - Added `runtime_handler` field to type `ImageSpec`

- [Add container filesystem to the ImageFsInfoResponse](https://github.com/kubernetes/kubernetes/pull/120914)
  - Added `container_filesystems` field to type `ImageFsInfoResponse`

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

### v1.27

`git diff v1.26.0 v1.27.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [CRI: Add CDI device info for containers](https://github.com/kubernetes/kubernetes/pull/115891/)
  - New type `CDIDevice` was introduced and added to container config

- [Add mappings for volumes](https://github.com/kubernetes/kubernetes/pull/116377)
  - Added new fields to the type `Mount` expressing runtime UID/GID mappings for the mount.

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

### v1.24

`git diff v1.23.0 v1.24.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [Update CRI-API Capabilities to include a field that allows us to set ambient capabilities](https://github.com/kubernetes/kubernetes/pull/104620)
  - The type `Capability` has a new string field `add_ambient_capabilities`

- [CRI-API - Add rootfs size to WindowsContainerResources](https://github.com/kubernetes/kubernetes/pull/108894)
  - The type `WindowsContainerResources` has a new int64 field `rootfs_size_in_bytes`

### v1.23

`git diff v1.22.0 v1.23.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- [CRI: add fields for pod level stats to satisfy the /stats/summary API](https://github.com/kubernetes/kubernetes/pull/102789)
  - New functions `PodSandboxStats`, `ListPodSandboxStats` with the corresponding types of request and response objects are introduced

- [pass sandbox resource requirements over CRI](https://github.com/kubernetes/kubernetes/pull/104886)
  - New fields on `LinuxPodSandboxConfig`: `overhead` and `resources` of type `LinuxContainerResources`.

- [prevents garbage collection from removing pinned images](https://github.com/kubernetes/kubernetes/pull/103299)
  - The type `Image` has a new boolean field `pinned`

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

### v1.21

`git diff v1.20.0 v1.21.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

No changes

### v1.20

`git diff v1.19.0 v1.20.0 -- staging/src/k8s.io/cri-api/pkg/apis/runtime/v1/api.proto`

- CRI [v1 introduced](https://github.com/kubernetes/kubernetes/pull/96387)
