Docker Engine Roadmap
=====================

### How should I use this document?

This document provides description of items that the project decided to prioritize. This should
serve as a reference point for Docker contributors to understand where the project is going, and
help determine if a contribution could be conflicting with some longer terms plans.

The fact that a feature isn't listed here doesn't mean that a patch for it will automatically be
refused (except for those mentioned as "frozen features" below)! We are always happy to receive
patches for new cool features we haven't thought about, or didn't judge priority. Please however
understand that such patches might take longer for us to review.

### How can I help?

Short term objectives are listed in the [wiki](https://github.com/docker/docker/wiki) and described
in [Issues](https://github.com/docker/docker/issues?q=is%3Aopen+is%3Aissue+label%3Aroadmap). Our
goal is to split down the workload in such way that anybody can jump in and help. Please comment on
issues if you want to take it to avoid duplicating effort! Similarly, if a maintainer is already
assigned on an issue you'd like to participate in, pinging him on IRC or GitHub to offer your help is
the best way to go.

### How can I add something to the roadmap?

The roadmap process is new to the Docker Engine: we are only beginning to structure and document the
project objectives. Our immediate goal is to be more transparent, and work with our community to
focus our efforts on fewer prioritized topics.

We hope to offer in the near future a process allowing anyone to propose a topic to the roadmap, but
we are not quite there yet. For the time being, the BDFL remains the keeper of the roadmap, and we
won't be accepting pull requests adding or removing items from this file.

# 1. Features and refactoring

## 1.1 Runtime improvements

We recently introduced [`runC`](https://runc.io) as a standalone low-level tool for container
execution. The initial goal was to integrate runC as a replacement in the Engine for the traditional
default libcontainer `execdriver`, but the Engine internals were not ready for this.

As runC continued evolving, and the OCI specification along with it, we created
[`containerd`](https://containerd.tools/), a daemon to control and monitor multiple `runC`. This is
the new target for Engine integration, as it can entirely replace the whole `execdriver`
architecture, and container monitoring along with it.

Docker Engine will rely on a long-running `containerd` companion daemon for all container execution
related operations. This could open the door in the future for Engine restarts without interrupting
running containers.

## 1.2 Plugins improvements

Docker Engine 1.7.0 introduced plugin support, initially for the use cases of volumes and networks
extensions. The plugin infrastructure was kept minimal as we were collecting use cases and real
world feedback before optimizing for any particular workflow.

In the future, we'd like plugins to become first class citizens, and encourage an ecosystem of
plugins. This implies in particular making it trivially easy to distribute plugins as containers
through any Registry instance, as well as solving the commonly heard pain points of plugins needing
to be treated as somewhat special (being active at all time, started before any other user
containers, and not as easily dismissed).

## 1.3 Internal decoupling

A lot of work has been done in trying to decouple the Docker Engine's internals. In particular, the
API implementation has been refactored, and the Builder side of the daemon is now
[fully independent](https://github.com/docker/docker/tree/master/builder) while still residing in
the same repository.

We are exploring ways to go further with that decoupling, capitalizing on the work introduced by the
runtime renovation and plugins improvement efforts. Indeed, the combination of `containerd` support
with the concept of "special" containers opens the door for bootstrapping more Engine internals
using the same facilities.

## 1.4 Cluster capable Engine

The community has been pushing for a more cluster capable Docker Engine, and a huge effort was spent
adding features such as multihost networking, and node discovery down at the Engine level. Yet, the
Engine is currently incapable of taking scheduling decisions alone, and continues relying on Swarm
for that.

We plan to complete this effort and make Engine fully cluster capable. Multiple instances of the
Docker Engine being already capable of discovering each other and establish overlay networking for
their container to communicate, the next step is for a given Engine to gain ability to dispatch work
to another node in the cluster. This will be introduced in a backward compatible way, such that a
`docker run` invocation on a particular node remains fully deterministic.

# 2. Frozen features

## 2.1 Docker exec

We won't accept patches expanding the surface of `docker exec`, which we intend to keep as a
*debugging* feature, as well as being strongly dependent on the Runtime ingredient effort.

## 2.2 Remote Registry Operations

A large amount of work is ongoing in the area of image distribution and provenance. This includes
moving to the V2 Registry API and heavily refactoring the code that powers these features. The
desired result is more secure, reliable and easier to use image distribution.

Part of the problem with this part of the code base is the lack of a stable and flexible interface.
If new features are added that access the registry without solidifying these interfaces, achieving
feature parity will continue to be elusive. While we get a handle on this situation, we are imposing
a moratorium on new code that accesses the Registry API in commands that don't already make remote
calls.

Currently, only the following commands cause interaction with a remote registry:

  - push
  - pull
  - run
  - build
  - search
  - login

In the interest of stabilizing the registry access model during this ongoing work, we are not
accepting additions to other commands that will cause remote interaction with the Registry API. This
moratorium will lift when the goals of the distribution project have been met.
