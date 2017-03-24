# containerd roadmap

This is a high level roadmap for the project that outlines what is currently being worked on, what comes next, and where you can help.

For a more up to date look please review the milestones on [github](https://github.com/docker/containerd/milestones).

The following are the different status the various phases of development can be in:
* Not Started - no work or thinking has been done towards the goal
* In Design - design work has started for the component and you can find design documents in the `design` folder
* In Progress - design has mostly finished and development has started
* Completed - the development work has been completed
* Stable - the apis for the phase are feature complete and considered stable

We would like to follow the roadmap and develop the components one by one to completion before starting the next phase.  If PRs are opened for another phase before the previous phase has been completed they will be closed as we are not ready for them at that time.

## Phase 1

**Status:** In Progress

### GRPC API

**Documents:**

We are going from a top down design for filling out this missing pieces of containerd and design of the API.

### Design

**Documents:**

The high level design work is needed so that the architecture of containerd stays consistent throughout the development process.

### Build & Test Process

**Documents:**

We need to have a simple build and test process for new developers to bootstrap their environments.
Because containerd will be the base of many high level systems we need to have a simple build process that does
not require high level tooling.

## Phase 2

Phase 2 includes most of the design and development work for the execution and storage layers of containerd.
It will include porting over existing "graph drivers" from Docker Engine and finding a common model for representing snapshots for layered filesystems.

This will also include moving the existing execution code support OCI's Runtime Spec and the existing containerd execution code.

**Status:** In Design

### Runtime

The runtime layer is responsible for the creation of containers and their management, and supervision of the processes inside those containers.

### Storage

**Documents:** https://github.com/docker/containerd/blob/master/design/snapshots.md

The current graph drivers were built when we only had overlay filesystems like aufs.
We forced the model to be designed around overlay filesystems and this introduced a lot of complexity for snapshotting graph drivers like btrfs and devicemapper thin-p.
Our current approach is to model our storage layer after snapshotting drivers instead of overlay drivers as we can get the same results and its cleaner and more robust to have an overlay filesytem model snapshots than it is to have a snapshot filesystem model overlay filesystems.

## Phase 3

This phase includes getting support for the OCI Image spec built into containerd.

**Status:** Not Started

### Distribution

## Phase 4

Phase 4 involves graduating to version 1.0, and shifting the focus from features to maintenance. Graduating to 1.0 implies:

- Completing all of the above phases.
- Covering the functionalities required by a majority of container-centric platforms.
- Offering feature parity, to the extent of technical possibilities, across Linux and Windows.
- Demonstrating that containerd fulfills the requirements of at least one higher-level platforms through its complete integration as an upstream.

**Status:** Not Started
