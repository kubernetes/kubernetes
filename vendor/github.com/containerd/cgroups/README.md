# cgroups

[![Build Status](https://github.com/containerd/cgroups/workflows/CI/badge.svg)](https://github.com/containerd/cgroups/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/containerd/cgroups/branch/main/graph/badge.svg)](https://codecov.io/gh/containerd/cgroups)
[![GoDoc](https://godoc.org/github.com/containerd/cgroups?status.svg)](https://godoc.org/github.com/containerd/cgroups)
[![Go Report Card](https://goreportcard.com/badge/github.com/containerd/cgroups)](https://goreportcard.com/report/github.com/containerd/cgroups)

Go package for creating, managing, inspecting, and destroying cgroups.
The resources format for settings on the cgroup uses the OCI runtime-spec found
[here](https://github.com/opencontainers/runtime-spec).

## Examples (v1)

### Create a new cgroup

This creates a new cgroup using a static path for all subsystems under `/test`.

* /sys/fs/cgroup/cpu/test
* /sys/fs/cgroup/memory/test
* etc....

It uses a single hierarchy and specifies cpu shares as a resource constraint and
uses the v1 implementation of cgroups.


```go
shares := uint64(100)
control, err := cgroups.New(cgroups.V1, cgroups.StaticPath("/test"), &specs.LinuxResources{
    CPU: &specs.LinuxCPU{
        Shares: &shares,
    },
})
defer control.Delete()
```

### Create with systemd slice support


```go
control, err := cgroups.New(cgroups.Systemd, cgroups.Slice("system.slice", "runc-test"), &specs.LinuxResources{
    CPU: &specs.CPU{
        Shares: &shares,
    },
})

```

### Load an existing cgroup

```go
control, err = cgroups.Load(cgroups.V1, cgroups.StaticPath("/test"))
```

### Add a process to the cgroup

```go
if err := control.Add(cgroups.Process{Pid:1234}); err != nil {
}
```

###  Update the cgroup

To update the resources applied in the cgroup

```go
shares = uint64(200)
if err := control.Update(&specs.LinuxResources{
    CPU: &specs.LinuxCPU{
        Shares: &shares,
    },
}); err != nil {
}
```

### Freeze and Thaw the cgroup

```go
if err := control.Freeze(); err != nil {
}
if err := control.Thaw(); err != nil {
}
```

### List all processes in the cgroup or recursively

```go
processes, err := control.Processes(cgroups.Devices, recursive)
```

### Get Stats on the cgroup

```go
stats, err := control.Stat()
```

By adding `cgroups.IgnoreNotExist` all non-existent files will be ignored, e.g. swap memory stats without swap enabled
```go
stats, err := control.Stat(cgroups.IgnoreNotExist)
```

### Move process across cgroups

This allows you to take processes from one cgroup and move them to another.

```go
err := control.MoveTo(destination)
```

### Create subcgroup

```go
subCgroup, err := control.New("child", resources)
```

### Registering for memory events

This allows you to get notified by an eventfd for v1 memory cgroups events.

```go
event := cgroups.MemoryThresholdEvent(50 * 1024 * 1024, false)
efd, err := control.RegisterMemoryEvent(event)
```

```go
event := cgroups.MemoryPressureEvent(cgroups.MediumPressure, cgroups.DefaultMode)
efd, err := control.RegisterMemoryEvent(event)
```

```go
efd, err := control.OOMEventFD()
// or by using RegisterMemoryEvent
event := cgroups.OOMEvent()
efd, err := control.RegisterMemoryEvent(event)
```

## Examples (v2/unified)

### Check that the current system is running cgroups v2

```go
var cgroupV2 bool
if cgroups.Mode() == cgroups.Unified {
	cgroupV2 = true
}
```

### Create a new cgroup

This creates a new systemd v2 cgroup slice. Systemd slices consider ["-" a special character](https://www.freedesktop.org/software/systemd/man/systemd.slice.html),
so the resulting slice would be located here on disk:

* /sys/fs/cgroup/my.slice/my-cgroup.slice/my-cgroup-abc.slice

```go
import (
    cgroupsv2 "github.com/containerd/cgroups/v2"
    specs "github.com/opencontainers/runtime-spec/specs-go"
)

res := cgroupsv2.Resources{}
// dummy PID of -1 is used for creating a "general slice" to be used as a parent cgroup.
// see https://github.com/containerd/cgroups/blob/1df78138f1e1e6ee593db155c6b369466f577651/v2/manager.go#L732-L735
m, err := cgroupsv2.NewSystemd("/", "my-cgroup-abc.slice", -1, &res)
if err != nil {
	return err
}
```

### Load an existing cgroup

```go
m, err := cgroupsv2.LoadSystemd("/", "my-cgroup-abc.slice")
if err != nil {
	return err
}
```

### Delete a cgroup

```go
m, err := cgroupsv2.LoadSystemd("/", "my-cgroup-abc.slice")
if err != nil {
	return err
}
err = m.DeleteSystemd()
if err != nil {
	return err
}
```

### Attention

All static path should not include `/sys/fs/cgroup/` prefix, it should start with your own cgroups name

## Project details

Cgroups is a containerd sub-project, licensed under the [Apache 2.0 license](./LICENSE).
As a containerd sub-project, you will find the:

 * [Project governance](https://github.com/containerd/project/blob/main/GOVERNANCE.md),
 * [Maintainers](https://github.com/containerd/project/blob/main/MAINTAINERS),
 * and [Contributing guidelines](https://github.com/containerd/project/blob/main/CONTRIBUTING.md)

information in our [`containerd/project`](https://github.com/containerd/project) repository.
