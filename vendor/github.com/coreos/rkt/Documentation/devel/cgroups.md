# Control Groups (cgroups)

## Background

[Control Groups][cgroups] are a Linux feature for organizing processes in hierarchical groups and applying resources limits to them. Each rkt pod is placed in a different cgroup to separate the processes of the pod from the processes of the host. Memory and CPU isolators are also implemented with cgroups.

[cgroups]: https://www.kernel.org/doc/Documentation/cgroup-v1/cgroups.txt

## Which cgroups are used

### `rkt` started from the command line

When a recent version of systemd is running on the host and `rkt` is not started as a systemd service (typically, from the command line), `rkt` will call `systemd-nspawn` with `--register=true`. This will cause `systemd-nspawn` to call the D-Bus method `CreateMachineWithNetwork` on `systemd-machined` and the cgroup `/machine.slice/machine-rkt...` will be created. This requires systemd v216+ as detected by [machinedRegister](https://github.com/coreos/rkt/blob/master/stage1/init/init.go#L161).

When systemd is not running on the host, or the systemd version is too old (< v216) to support the D-Bus method `CreateMachineWithNetwork`, `rkt` uses `systemd-nspawn` with the `--register=false` parameter. In this case, `systemd-nspawn` or other systemd components will not create new cgroups for rkt. On older `rkt` versions (<= 0.13.0), the pod was staying in the same cgroup as the caller. For example, it could be something like `/user.slice/user-1000.slice/session-1.scope`. Since [#1844](https://github.com/coreos/rkt/pull/1844) (rkt >= 0.14.0), `rkt` creates a new cgroup for each pod, like `$CALLER_CGROUP/machine-some-id.slice`.

### `rkt` started as a systemd service

`rkt` is able to detect if it is started as a systemd service (from a `.service` file or from `systemd-run`).
In that case, `systemd-nspawn` is started with the `--keep-unit` parameter.
This will cause `systemd-nspawn` to use the D-Bus method call `RegisterMachineWithNetwork` instead of `CreateMachineWithNetwork` and the pod will remain in the cgroup of the service.
By default, the slice is `systemd.slice` but [users are advised](https://github.com/coreos/rkt/blob/master/Documentation/using-rkt-with-systemd.md) to select `machine.slice` with `systemd-run --slice=machine` or `Slice=machine.slice` in the `.service` file.
It will result in `/machine.slice/servicename.service` when the user select that slice.

### Summary

1. `/machine.slice/machine-rkt...` when started on the command line with systemd v216+.
2. `/$SLICE.slice/servicename.service` when started from a systemd service.
3. `$CALLER_CGROUP/machine-some-id.slice` without systemd-v216+, since rkt >= 0.14.0.

## Future work

### Unified hierarchy and cgroup2

Unified hierarchy and cgroup2 is a new feature in Linux that will be available in Linux 4.4.

This is tracked by [#1757](https://github.com/coreos/rkt/issues/1757).

### CGroup Namespaces

CGroup Namespaces is a new feature being developed in Linux.

This is tracked by [#1757](https://github.com/coreos/rkt/issues/1757).

### Network isolator

Appc/spec defines a [network isolator](https://github.com/appc/spec/blob/master/spec/ace.md#resourcenetwork-bandwidth) "resource/network-bandwidth" to limit the network bandwidth used by each app in the pod.
This is not implemented yet.
This could be implemented with cgroups.

