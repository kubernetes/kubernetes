# Development Report for May 05, 2017

## runc and runtime-spec 1.0 final

This week we have been working on getting Docker integrated with the final RCs for the OCI runtime-spec and runc.  We currently have a [PR open on Docker](https://github.com/moby/moby/pull/33007) that is ready for review.

This is important for the containerd project because we need a stable runtime like runc and have a spec that is not constantly changing so that people can begin integrating with containerd and know that the APIs we expose and APIs/specs that we depend on are stable for the long term.

## New Projects

We finished moving a few of our external dependencies into the containerd organization this week.  There were a few projects that we built outside but wanted to bring these under the project to ensure that our dependencies are held to the same standards as the rest of our codebase.

### go-runc

[go-runc](https://github.com/containerd/go-runc)

This project contains the runc bindings that we consume in containerd to interface with runc and other OCI runtimes.  It lets us interact with the binary and handles many of the common options like working with the console socket and other IO operations.

### fifo

[fifo](https://github.com/containerd/fifo)

This package contains helpers for handling fifos.  Fifos are a little more complex than regular pipes and sometimes requires special handling and blocking semantics depending on the flags used with opening the fifo on either end.  This package helps to handle many of the common use cases that we use in containerd.

### console

[console](https://github.com/containerd/console)

The console package is a refresh of the `term` package from Docker.  It provides a cleaner API for working with the current console for a program or creating new terminals and keeping the flags in sync for proxying reads and writes between the two sides of the console.

### cgroups

[cgroups](https://github.com/containerd/cgroups)

The cgroups package is currently used in containerd for collecting stats from cgroups that are created for a container.  It exposes a package for exporting cgroup level metrics to prometheus for containers.

### btrfs

[btrfs](https://github.com/containerd/btrfs)

The btrfs package handles interfacing with btrfs for our snapshotter.  It binds to the btrfs C library to create subvolumes and handle any other interaction with the filesystem.

### continuity

[continuity](https://github.com/containerd/continuity)

continuity provides a transport agnostic filesystem metadata manifest.  This allows us to work with filesystems at the file level instead of interacting with a "layer". We also intend to concentrate a rich set of file system utility packages for use in containerd.
