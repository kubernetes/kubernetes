<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubelet and systemd interaction

**Author**: Derek Carr (@derekwaynecarr)

**Status**: Proposed

## Motivation

Many Linux distributions have either adopted, or plan to adopt `systemd` as their init system.

This document describes how the node should be configured, and a set of enhancements that should
be made to the `kubelet` to better integrate with these distributions independent of container
runtime.

## Scope of proposal

This proposal does not account for running the `kubelet` in a container.

## Background on systemd

To help understand this proposal, we first provide a brief summary of `systemd` behavior.

### systemd units

`systemd` manages a hierarchy of `slice`, `scope`, and `service` units.

* `service` - application on the server that is launched by `systemd`; how it should start/stop;
when it should be started; under what circumstances it should be restarted; and any resource
controls that should be applied to it.
* `scope` - a process or group of processes which are not launched by `systemd` (i.e. fork), like
a service, resource controls may be applied
* `slice` - organizes a hierarchy in which `scope` and `service` units are placed.  a `slice` may
contain `slice`, `scope`, or `service` units; processes are attached to `service` and `scope`
units only, not to `slices`. The hierarchy is intended to be unified, meaning a process may
only belong to a single leaf node.

### cgroup hierarchy: split versus unified hierarchies

Classical `cgroup` hierarchies were split per resource group controller, and a process could
exist in different parts of the hierarchy.

For example, a process `p1` could exist in each of the following at the same time:

* `/sys/fs/cgroup/cpu/important/`
* `/sys/fs/cgroup/memory/unimportant/`
* `/sys/fs/cgroup/cpuacct/unimportant/`

In addition, controllers for one resource group could depend on another in ways that were not
always obvious.

For example, the `cpu` controller depends on the `cpuacct` controller yet they were treated
separately.

Many found it confusing for a single process to belong to different nodes in the `cgroup` hierarchy
across controllers.

The Kernel direction for `cgroup` support is to move toward a unified `cgroup` hierarchy, where the
per-controller hierarchies are eliminated in favor of hierarchies like the following:

* `/sys/fs/cgroup/important/`
* `/sys/fs/cgroup/unimportant/`

In a unified hierarchy, a process may only belong to a single node in the `cgroup` tree.

### cgroupfs single writer

The Kernel direction for `cgroup` management is to promote a single-writer model rather than
allowing multiple processes to independently write to parts of the file-system.

In distributions that run `systemd` as their init system, the cgroup tree is managed by `systemd`
by default since it implicitly interacts with the cgroup tree when starting units.  Manual changes
made by other cgroup managers to the cgroup tree are not guaranteed to be preserved unless `systemd`
is made aware.  `systemd` can be told to ignore sections of the cgroup tree by configuring the unit
to have the `Delegate=` option.

See: http://www.freedesktop.org/software/systemd/man/systemd.resource-control.html#Delegate=

### cgroup management with systemd and container runtimes

A `slice` corresponds to an inner-node in the `cgroup` file-system hierarchy.

For example, the `system.slice` is represented as follows:

`/sys/fs/cgroup/<controller>/system.slice`

A `slice` is nested in the hierarchy by its naming convention.

For example, the `system-foo.slice` is represented as follows:

`/sys/fs/cgroup/<controller>/system.slice/system-foo.slice/`

A `service` or `scope` corresponds to leaf nodes in the `cgroup` file-system hierarchy managed by
`systemd`. Services and scopes can have child nodes managed outside of `systemd` if they have been
delegated with the `Delegate=` option.

For example, if the `docker.service` is associated with the `system.slice`, it is
represented as follows:

`/sys/fs/cgroup/<controller>/system.slice/docker.service/`

To demonstrate the use of `scope` units using the `docker` container runtime, if a
user launches a container via `docker run -m 100M busybox`, a `scope` will be created
because the process was not launched by `systemd` itself.  The `scope` is parented by
the `slice` associated with the launching daemon.

For example:

`/sys/fs/cgroup/<controller>/system.slice/docker-<container-id>.scope`

`systemd` defines a set of slices.  By default, service and scope units are placed in
`system.slice`, virtual machines and containers registered with `systemd-machined` are
found in `machine.slice`, and user sessions handled by `systemd-logind` in `user.slice`.

## Node Configuration on systemd

### kubelet cgroup driver

The `kubelet` reads and writes to the `cgroup` tree during bootstrapping
of the node.  In the future, it will write to the `cgroup` tree to satisfy other
purposes around quality of service, etc.

The `kubelet` must cooperate with `systemd` in order to ensure proper function of the
system.  The bootstrapping requirements for a `systemd` system are different than one
without it.

The `kubelet` will accept a new flag to control how it interacts with the `cgroup` tree.

* `--cgroup-driver=` - cgroup driver used by the kubelet. `cgroupfs` or `systemd`.

By default, the `kubelet` should default `--cgroup-driver` to `systemd` on `systemd` distributions.

The `kubelet` should associate node bootstrapping semantics to the configured
`cgroup driver`.

### Node allocatable

The proposal makes no changes to the definition as presented here:
https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/node-allocatable.md

The node will report a set of allocatable compute resources defined as follows:

`[Allocatable] = [Node Capacity] - [Kube-Reserved] - [System-Reserved]`

### Node capacity

The `kubelet` will continue to interface with `cAdvisor` to determine node capacity.

### System reserved

The node may set aside a set of designated resources for non-Kubernetes components.

The `kubelet` accepts the followings flags that support this feature:

* `--system-reserved=` - A set of `ResourceName`=`ResourceQuantity` pairs that
describe resources reserved for host daemons.
* `--system-container=` - Optional resource-only container in which to place all
non-kernel processes that are not already in a container. Empty for no container.
Rolling back the flag requires a reboot. (Default: "").

The current meaning of `system-container` is inadequate on `systemd` environments.
The `kubelet` should use the flag to know the location that has the processes that
are associated with `system-reserved`, but it should not modify the cgroups of
existing processes on the system during bootstrapping of the node.  This is
because `systemd` is the `cgroup manager` on the host and it has not delegated
authority to the `kubelet` to change how it manages `units`.

The following describes the type of things that can happen if this does not change:
https://bugzilla.redhat.com/show_bug.cgi?id=1202859

As a result, the `kubelet` needs to distinguish placement of non-kernel processes
based on the cgroup driver, and only do its current behavior when not on `systemd`.

The flag should be modified as follows:

* `--system-container=` - Name of resource-only container that holds all
non-kernel processes whose resource consumption is accounted under
system-reserved.  The default value is cgroup driver specific.  systemd
defaults to system, cgroupfs defines no default.  Rolling back the flag
requires a reboot.

The `kubelet` will error if the defined `--system-container` does not exist
on `systemd` environments.  It will verify that the appropriate `cpu` and `memory`
controllers are enabled.

### Kubernetes reserved

The node may set aside a set of resources for Kubernetes components:

* `--kube-reserved=:` - A set of `ResourceName`=`ResourceQuantity` pairs that
describe resources reserved for host daemons.

The `kubelet` does not enforce `--kube-reserved` at this time, but the ability
to distinguish the static reservation from observed usage is important for node accounting.

This proposal asserts that `kubernetes.slice` is the default slice associated with
the `kubelet` and `kube-proxy` service units defined in the project.  Keeping it
separate from `system.slice` allows for accounting to be distinguished separately.

The `kubelet` will detect its `cgroup` to track `kube-reserved` observed usage on `systemd`.
If the `kubelet` detects that its a child of the `system-container` based on the observed
`cgroup` hierarchy, it will warn.

If the `kubelet` is launched directly from a terminal, it's most likely destination will
be in a `scope` that is a child of `user.slice` as follows:

`/sys/fs/cgroup/<controller>/user.slice/user-1000.slice/session-1.scope`

In this context, the parent `scope` is what will be used to facilitate local developer
debugging scenarios for tracking `kube-reserved` usage.

The `kubelet` has the following flag:

* `--resource-container="/kubelet":` Absolute name of the resource-only container to create
and run the Kubelet in (Default: /kubelet).

This flag will not be supported on `systemd` environments since the init system has already
spawned the process and placed it in the corresponding container associated with its unit.

### Kubernetes container runtime reserved

This proposal asserts that the reservation of compute resources for any associated
container runtime daemons is tracked by the operator under the `system-reserved` or
`kubernetes-reserved` values and any enforced limits are set by the
operator specific to the container runtime.

**Docker**

If the `kubelet` is configured with the `container-runtime` set to `docker`, the
`kubelet` will detect the `cgroup` associated with the `docker` daemon and use that
to do local node accounting.  If an operator wants to impose runtime limits on the
`docker` daemon to control resource usage, the operator should set those explicitly in
the `service` unit that launches `docker`.  The `kubelet` will not set any limits itself
at this time and will assume whatever budget was set aside for `docker` was included in
either `--kube-reserved` or `--system-reserved` reservations.

Many OS distributions package `docker` by default, and it will often belong to the
`system.slice` hierarchy, and therefore operators will need to budget it for there
by default unless they explicitly move it.

**rkt**

rkt has no client/server daemon, and therefore has no explicit requirements on container-runtime
reservation.

### kubelet cgroup enforcement

The `kubelet` does not enforce the `system-reserved` or `kube-reserved` values by default.

The `kubelet` should support an additional flag to turn on enforcement:

* `--system-reserved-enforce=false` - Optional flag that if true tells the `kubelet`
to enforce the `system-reserved` constraints defined (if any)
* `--kube-reserved-enforce=false` - Optional flag that if true tells the `kubelet`
to enforce the `kube-reserved` constraints defined (if any)

Usage of this flag requires that end-user containers are launched in a separate part
of cgroup hierarchy via `cgroup-root`.

If this flag is enabled, the `kubelet` will continually validate that the configured
resource constraints are applied on the associated `cgroup`.

### kubelet cgroup-root behavior under systemd

The `kubelet` supports a `cgroup-root` flag which is the optional root `cgroup` to use for pods.

This flag should be treated as a pass-through to the underlying configured container runtime.

If `--cgroup-enforce=true`, this flag warrants special consideration by the operator depending
on how the node was configured.  For example, if the container runtime is `docker` and its using
the `systemd` cgroup driver, then `docker` will take the daemon wide default and launch containers
in the same slice associated with the `docker.service`.  By default, this would mean `system.slice`
which could cause end-user pods to be launched in the same part of the cgroup hierarchy as system daemons.

In those environments, it is recommended that `cgroup-root` is configured to be a subtree of `machine.slice`.

### Proposed cgroup hierarchy

```
$ROOT
  |
  +- system.slice 
  |   |
  |   +- sshd.service
  |   +- docker.service (optional)
  |   +- ...
  |
  +- kubernetes.slice
  |   |
  |   +- kubelet.service
  |   +- docker.service (optional)
  |
  +- machine.slice (container runtime specific)
  |   |
  |   +- docker-<container-id>.scope
  |
  +- user.slice
  |   +- ...
```

* `system.slice` corresponds to `--system-reserved`, and contains any services the
operator brought to the node as normal configuration.
* `kubernetes.slice` corresponds to the `--kube-reserved`, and contains kube specific
daemons.
* `machine.slice` should parent all end-user containers on the system and serve as the
root of the end-user cluster workloads run on the system.
* `user.slice` is not explicitly tracked by the `kubelet`, but it is possible that `ssh`
sessions to the node where the user launches actions directly.  Any resource accounting
reserved for those actions should be part of `system-reserved`.

The container runtime daemon, `docker` in this outline, must be accounted for in either
`system.slice` or `kubernetes.slice`.

In the future, the depth of the container hierarchy is not recommended to be rooted
more than 2 layers below the root as it historically has caused issues with node performance
in other `cgroup` aware systems (https://bugzilla.redhat.com/show_bug.cgi?id=850718).  It
is anticipated that the `kubelet` will parent containers based on quality of service
in the future.  In that environment, those changes will be relative to the configured
`cgroup-root`.

### Linux Kernel Parameters

The `kubelet` will set the following:

* `sysctl -w vm.overcommit_memory=1`
* `sysctl -w vm.panic_on_oom=0`
* `sysctl -w kernel/panic=10`
* `sysctl -w kernel/panic_on_oops=1`

### OOM Score Adjustment

The `kubelet` at bootstrapping will set the `oom_score_adj` value for Kubernetes
daemons, and any dependent container-runtime daemons.

If `container-runtime` is set to `docker`, then set its `oom_score_adj=-999`

## Implementation concerns

### kubelet block-level architecture

```
+----------+       +----------+    +----------+
|          |       |          |    | Pod      |
|  Node    <-------+ Container<----+ Lifecycle|
|  Manager |       | Manager  |    | Manager  |
|          +------->          |    |          |
+---+------+       +-----+----+    +----------+
    |                    |
    |                    |
    |  +-----------------+
    |  |                 |
    |  |                 |
+---v--v--+        +-----v----+
| cgroups |        | container|
| library |        | runtimes |
+---+-----+        +-----+----+
    |                    |
    |                    |
    +---------+----------+
              |
              |
  +-----------v-----------+
  |     Linux Kernel      |
  +-----------------------+
```

The `kubelet` should move to an architecture that resembles the above diagram:

* The `kubelet` should not interface directly with the `cgroup` file-system, but instead
should use a common `cgroups library` that has the proper abstraction in place to
work with either `cgroupfs` or `systemd`.  The `kubelet` should just use `libcontainer`
abstractions to facilitate this requirement.  The `libcontainer` abstractions as
currently defined only support an `Apply(pid)` pattern, and we need to separate that
abstraction to allow cgroup to be created and then later joined.
* The existing `ContainerManager` should separate node bootstrapping into a separate
`NodeManager` that is dependent on the configured `cgroup-driver`.
* The `kubelet` flags for cgroup paths will convert internally as part of cgroup library,
i.e. `/foo/bar` will just convert to `foo-bar.slice`

### kubelet accounting for end-user pods

This proposal re-enforces that it is inappropriate at this time to depend on `--cgroup-root` as the
primary mechanism to distinguish and account for end-user pod compute resource usage.

Instead, the `kubelet` can and should sum the usage of each running `pod` on the node to account for
end-user pod usage separate from system-reserved and kubernetes-reserved accounting via `cAdvisor`.

## Known issues

### Docker runtime support for --cgroup-parent

Docker versions <= 1.0.9 did not have proper support for `-cgroup-parent` flag on `systemd`.  This
was fixed in this PR (https://github.com/docker/docker/pull/18612).  As result, it's expected
that containers launched by the `docker` daemon may continue to go in the default `system.slice` and
appear to be counted under system-reserved node usage accounting.

If operators run with later versions of `docker`, they can avoid this issue via the use of `cgroup-root`
flag on the `kubelet`, but this proposal makes no requirement on operators to do that at this time, and
this can be revisited if/when the project adopts docker 1.10.

Some OS distributions will fix this bug in versions of docker <= 1.0.9, so operators should
be aware of how their version of `docker` was packaged when using this feature.





<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubelet-systemd.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
