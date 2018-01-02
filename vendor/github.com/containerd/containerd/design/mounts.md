# Mounts

Mounts are the main interaction mechanism in containerd. Container systems of
the past typically end up having several disparate components independently
perform mounts, resulting in complex lifecycle management and buggy behavior
when coordinating large mount stacks.

In containerd, we intend to keep mount syscalls isolated to the container
runtime component, opting to have various components produce a serialized
representation of the mount. This ensures that the mounts are performed as a
unit and unmounted as a unit.

From an architecture perspective, components produce mounts and runtime
executors consume them.

More imaginative use cases include the ability to virtualize a series of mounts
from various components without ever having to create a runtime. This will aid
in testing and implementation of satellite components.

## Structure

The `Mount` type follows the structure of the historic mount syscall:

| Field | Type | Description |
|-------|----|-------------|
| Type | `string` | Specific type of the mount, typically operating system specific |
| Target | `string` | Intended filesystem path for the mount destination. |
| Source | `string` | The object which originates the mount, typically a device or another filesystem path. |
| Options | `[]string` | Zero or more options to apply with the mount, possibly `=`-separated key value pairs. | 

We may want to further parameterize this to support mounts with various
helpers, such as `mount.fuse`, but this is out of scope, for now.
