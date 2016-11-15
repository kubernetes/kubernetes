# Support HostPath volume existence qualifiers

## Introduction

A Host volume source is probably the simplest volume type to define, needing
only a single path. However, that simplicity comes with many assumptions and
caveats.

This proposal describes one of the issues associated with Host volumes &mdash;
their silent and implicit creation of directories on the host &mdash; and
proposes a solution.

## Problem

Right now, under Docker, when a bindmount references a hostPath, that path will
be created as an empty directory, owned by root, if it does not already exist.
This is rarely what the user actually wants because hostPath volumes are
typically used to express a dependency on an existing external file or
directory.
This concern was raised during the [initial
implementation](https://github.com/docker/docker/issues/1279#issuecomment-22965058)
of this behavior in Docker and it was suggested that orchestration systems
could better manage volume creation than Docker, but Docker does so as well
anyways.

To fix this problem, I propose allowing a pod to specify whether a given
hostPath should exist prior to the pod running, whether it should be created,
and what it should exist as.
I also propose the inclusion of a default value which matches the current
behavior to ensure backwards compatibility.

To understand exactly when this behavior will or won't be correct, it's
important to look at the use-cases of Host Volumes.
The table below broadly classifies the use-case of Host Volumes and asserts
whether this change would be of benefit to that use-case.

### HostPath volume Use-cases

| Use-case | Description | Examples | Benefits from this change? | Why? |
|:---------|:------------|:---------|:--------------------------:|:-----|
| Accessing an external system, data, or configuration | Data or a unix socket is created by a process on the host, and a pod within kubernetes consumes it | [fluentd-es-addon](https://github.com/kubernetes/kubernetes/blob/74b01041cc3feb2bb731cc243ab0e4515bef9a84/cluster/saltbase/salt/fluentd-es/fluentd-es.yaml#L30), [addon-manager](https://github.com/kubernetes/kubernetes/blob/808f3ecbe673b4127627a457dc77266ede49905d/cluster/gce/coreos/kube-manifests/kube-addon-manager.yaml#L23), [kube-proxy](https://github.com/kubernetes/kubernetes/blob/010c976ce8dd92904a7609483c8e794fd8e94d4e/cluster/saltbase/salt/kube-proxy/kube-proxy.manifest#L65), etc | :white_check_mark: | Fails faster and with more useful messages, and won't run when basic assumptions are false (e.g. that docker is the runtime and the docker.sock exists) |
| Providing data to external systems | Some pods wish to publish data to the host for other systems to consume, sometimes to a generic directory and sometimes to more component-specific ones | Kubelet core components which bindmount their logs out to `/var/log/*.log` so logrotate and other tools work with them | :white_check_mark: | Sometimes, but not always. It's directory-specific whether it not existing will be a problem. |
| Communicating between instances and versions of yourself | A pod can use a hostPath directory as a sort of cache and, as opposed to an emptyDir, persist the directory between versions of itself | [etcd](https://github.com/kubernetes/kubernetes/blob/fac54c9b22eff5c5052a8e3369cf8416a7827d36/cluster/saltbase/salt/etcd/etcd.manifest#L84), caches | :x: | It's pretty much always okay to create them |


### Other motivating factors

One additional motivating factor for this change is that under the rkt runtime
paths are not created when they do not exist. This change moves the management
of these volumes into the Kubelet to the benefit of the rkt container runtime.


## Proposed API Change

### Host Volume

I propose that the
[`v1.HostPathVolumeSource`](https://github.com/kubernetes/kubernetes/blob/d26b4ca2859aa667ad520fb9518e0db67b74216a/pkg/api/types.go#L447-L451)
object be changed to include the following additional field:

`Type` - An optional string of `exists|file|device|socket|directory` - If not
set, it will default to a backwards-compatible default behavior described
below.

| Value | Behavior |
|:------|:---------|
| *unset* | If nothing exists at the given path, an empty directory will be created there. Otherwise, behaves like `exists` |
| `exists` | If nothing exists at the given path, the pod will fail to run and provide an informative error message |
| `file` | If a file does not exist at the given path, the pod will fail to run and provide an informative error message |
| `device` | If a block or character device does not exist at the given path, the pod will fail to run and provide an informative error message |
| `socket` | If a socket does not exist at the given path, the pod will fail to run and provide an informative error message |
| `directory` | If a directory does not exist at the given path, the pod will fail to run and provide an informative error message |

Additional possible values, which are proposed to be excluded:

|Value | Behavior | Reason for exclusion |
|:-----|:---------|:---------------------|
| `new-directory` | Like `auto`, but the given path must be a directory if it exists | `auto` mostly fills this use-case |
| `character-device` |  | Granularity beyond `device` shouldn't matter often |
| `block-device` |  | Granularity beyond `device` shouldn't matter often |
| `new-file` | Like file, but if nothing exist an empty file is created instead | In general, bindmounting the parent directory of the file you intend to create addresses this usecase |
| `optional` | If a path does not exist, then do not create any container-mount at all | This would better be handled by a new field entirely if this behavior is desirable |


### Why not as part of any other volume types?

This feature does not make sense for any of the other volume types simply
because all of the other types are already fully qualified. For example, NFS
volumes are known to always be in existence else they will not mount.
Similarly, EmptyDir volumes will always exist as a directory.

Only the HostVolume and SubPath means of referencing a path have the potential
to reference arbitrary incorrect or nonexistent things without erroring out.

### Alternatives

One alternative is to augment Host Volumes with a `MustExist` bool and provide
no further granularity. This would allow toggling between the `auto` and
`exists` behaviors described above. This would likely cover the "90%" use-case
and would be a simpler API. It would be sufficient for all of the examples
linked above in my opionion.

## Kubelet implementation

It's proposed that prior to starting a pod, the Kubelet validates that the
given path meets the qualifications of its type. Namely, if the type is `auto`
the Kubelet will create an empty directory if none exists there, and for each
of the others the Kubelet will perform the given validation prior to running
the pod. This validation might be done by a volume plugin, but further
technical consideration (out of scope of this proposal) is needed.


## Possible concerns

### Permissions

This proposal does not attempt to change the state of volume permissions. Currently, a HostPath volume is created with `root` ownership and `755` permissions. This behavior will be retained. An argument for this behavior is given [here](volumes.md#shared-storage-hostpath).

### SELinux

This proposal should not impact SELinux relabeling. Verifying the presence and
type of a given path will be logically separate from SELinux labeling.
Similarly, creating the directory when it doesn't exist will happen before any
SELinux operations and should not impact it.


### Containerized Kubelet

A containerized kubelet would have difficulty creating directories. The
implementation will likely respect the `containerized` flag, or similar,
allowing it to either break out or be "/rootfs/" aware and thus operate as
desired.

### Racy Validation

Ideally the validation would be done at the time the bindmounts are created,
else it's possible for a given path or directory to change in the duration from
when it's validated and the container runtime attempts to create said mount.

The only way to solve this problem is to integrate these sorts of qualification
into container runtimes themselves.

I don't think this problem is severe enough that we need to push to solve it;
rather I think we can simply accept this minor race, and if runtimes eventually
allow this we can begin to leverage them.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/volume-hostpath-qualifiers.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
