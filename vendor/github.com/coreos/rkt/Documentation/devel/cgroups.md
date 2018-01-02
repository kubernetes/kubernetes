# Control Groups (cgroups)

## Background

[Control Groups][cgroups] are a Linux feature for organizing processes in hierarchical groups and applying resources limits to them. Each rkt pod is placed in a different cgroup to separate the processes of the pod from the processes of the host. Memory and CPU isolators are also implemented with cgroups.

## What cgroup does rkt use?

Every pod and application within that pod is run within its own cgroup.

### `rkt` started from the command line

When a recent version of systemd is running on the host and `rkt` is not started as a systemd service (typically, from the command line), `rkt` will call `systemd-nspawn` with `--register=true`. This will cause `systemd-nspawn` to call the D-Bus method `CreateMachineWithNetwork` on `systemd-machined` and the cgroup `/machine.slice/machine-rkt...` will be created. This requires systemd v216+ as detected by the [machinedRegister][machinedRegister] function in stage1's `init`.

When systemd is not running on the host, or the systemd version is too old (< v216), `rkt` uses `systemd-nspawn` with the `--register=false` parameter. In this case, `systemd-nspawn` or other systemd components will not create new cgroups for rkt. Instead, `rkt` creates a new cgroup for each pod under the current cgroup, like `$CALLER_CGROUP/machine-some-id.slice`.

### `rkt` started as a systemd service

`rkt` is able to detect if it is started as a systemd service (from a `.service` file or from `systemd-run`).
In that case, `systemd-nspawn` is started with the `--keep-unit` parameter.
This will cause `systemd-nspawn` to use the D-Bus method call `RegisterMachineWithNetwork` instead of `CreateMachineWithNetwork` and the pod will remain in the cgroup of the service.
By default, the slice is `systemd.slice` but [users are advised][rkt-systemd] to select `machine.slice` with `systemd-run --slice=machine` or `Slice=machine.slice` in the `.service` file.
It will result in `/machine.slice/servicename.service` when the user select that slice.

### Summary

1. `/machine.slice/machine-rkt...` when started on the command line with systemd v216+.
2. `/$SLICE.slice/servicename.service` when started from a systemd service.
3. `$CALLER_CGROUP/machine-some-id.slice` without systemd, or with systemd pre-v216

For example, a simple pod run interactively on a system with systemd would look like:

```
├─machine.slice
│ └─machine-rkt\x2df28d074b\x2da8bb\x2d4246\x2d96a5\x2db961e1fe7035.scope
│   ├─init.scope
│   │ └─/usr/lib/systemd/systemd
│   └─system.slice
│     ├─alpine-sh.service
│     │ ├─/bin/sh 
│     └─systemd-journald.service
│       └─/usr/lib/systemd/systemd-journald
```


## What subsystems does rkt use?

Right now, rkt uses the `cpu`, `cpuset`, and `memory` subsystems.

### How are they mounted?

When the stage1 starts, it mounts `/sys` . Then, for every subsystem, it:

1. Mounts the subsystem (on `<rootfs>/sys/fs/cgroup/<subsystem>`)
2. Bind-mounts the subcgroup on top of itself (e.g `<rootfs>/sys/fs/cgroup/memory/machine.slice/machine-rkt-UUID.scope/`)
3. Remounts the subsystem readonly

This is so that the pod itself cannot escape the cgroup. Currently the cgroup filesystems are not accessible to applications within the pod, but that may change.

(N.B. `rkt` prior to v1.23 mounted each individual *knob* read-write. E.g. `.../memory/machine.slice/machine-rkt-UUID.scope/system.slice/etcd.service/{memory.limit_in_bytes, cgroup.procs}`)

## Future work

### Unified hierarchy and cgroup2

Unified hierarchy and cgroup2 is a new feature in Linux that will be available in Linux 4.4.

This is tracked by [#1757][rkt-1757].

### CGroup Namespaces

CGroup Namespaces is a new feature being developed in Linux.

This is tracked by [#1757][rkt-1757].

### Network isolator

Appc/spec defines the [network isolator][network-isolator] `resource/network-bandwidth` to limit the network bandwidth used by each app in the pod.
This is not implemented yet.
This could be implemented with cgroups.

[cgroups]: https://www.kernel.org/doc/Documentation/cgroup-v1/cgroups.txt
[machinedRegister]: https://github.com/coreos/rkt/blob/master/stage1/init/init.go#L153
[network-isolator]: https://github.com/appc/spec/blob/master/spec/ace.md#resourcenetwork-bandwidth
[rkt-1757]: https://github.com/coreos/rkt/issues/1757
[rkt-1844]: https://github.com/coreos/rkt/pull/1844
[rkt-systemd]: ../using-rkt-with-systemd.md
