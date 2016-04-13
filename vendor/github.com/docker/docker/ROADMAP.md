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

## 1.1 Security

Security is a top objective for the Docker Engine. The most notable items we intend to provide in
the near future are:

- Trusted distribution of images: the effort is driven by the [distribution](https://github.com/docker/distribution)
group but will have significant impact on the Engine
- [User namespaces](https://github.com/docker/docker/pull/12648)
- [Seccomp support](https://github.com/docker/libcontainer/pull/613)

## 1.2 Plumbing project

We define a plumbing tool as a standalone piece of software usable and meaningful on its own. In
the current state of the Docker Engine, most subsystems provide independent functionalities (such
the builder, pushing and pulling images, running applications in a containerized environment, etc)
but all are coupled in a single binary.  We want to offer the users to flexibility to use only the
pieces they need, and we will also gain in maintainability by splitting the project among multiple
repositories.

As it currently stands, the rough design outlines is to have:
- Low level plumbing tools, each dealing with one responsibility (e.g., [runC](https://runc.io))
- Docker subsystems services, each exposing an elementary concept over an API, and relying on one or
multiple lower level plumbing tools for their implementation (e.g., network management)
- Docker Engine to expose higher level actions (e.g., create a container with volume `V` and network
`N`), while still providing pass-through access to the individual subsystems.

The architectural details are still being worked on, but one thing we know for sure is that we need
to technically decouple the pieces.

### 1.2.1 Runtime

A Runtime tool already exists today in the form of [runC](https://github.com/opencontainers/runc).
We intend to modify the Engine to directly call out to a binary implementing the Open Containers
Specification such as runC rather than relying on libcontainer to set the container runtime up.

This plan will deprecate the existing [`execdriver`](https://github.com/docker/docker/tree/master/daemon/execdriver)
as different runtime backends will be implemented as separated binaries instead of being compiled
into the Engine.

### 1.2.2 Builder

The Builder (i.e., the ability to build an image from a Dockerfile) is already nicely decoupled,
but would benefit from being entirely separated from the Engine, and rely on the standard Engine
API for its operations.

### 1.2.3 Distribution

Distribution already has a [dedicated repository](https://github.com/docker/distribution) which
holds the implementation for Registry v2 and client libraries. We could imagine going further by
having the Engine call out to a binary providing image distribution related functionalities.

There are two short term goals related to image distribution. The first is stabilize and simplify
the push/pull code. Following that is the conversion to the more secure Registry V2 protocol.

### 1.2.4 Networking

Most of networking related code was already decoupled today in [libnetwork](https://github.com/docker/libnetwork).
As with other ingredients, we might want to take it a step further and make it a meaningful utility
that the Engine would call out to instead of a library.

## 1.3 Plugins

An initiative around plugins started with Docker 1.7.0, with the goal of allowing for out of
process extensibility of some Docker functionalities, starting with volumes and networking. The
approach is to provide specific extension points rather than generic hooking facilities. We also
deliberately keep the extensions API the simplest possible, expanding as we discover valid use
cases that cannot be implemented.

At the time of writing:

- Plugin support is merged as an experimental feature: real world use cases and user feedback will
help us refine the UX to make the feature more user friendly.
- There are no immediate plans to expand on the number of pluggable subsystems.
- Golang 1.5 might add language support for [plugins](https://docs.google.com/document/d/1nr-TQHw_er6GOQRsF6T43GGhFDelrAP0NqSS_00RgZQ)
which we consider supporting as an alternative to JSON/HTTP.

## 1.4 Volume management

Volumes are not a first class citizen in the Engine today: we would like better volume management,
similar to the way network are managed in the new [CNM](https://github.com/docker/docker/issues/9983).

## 1.5 Better API implementation

The current Engine API is insufficiently typed, versioned, and ultimately hard to maintain. We
also suffer from the lack of a common implementation with [Swarm](https://github.com/docker/swarm).

## 1.6 Checkpoint/restore

Support for checkpoint/restore was [merged](https://github.com/docker/libcontainer/pull/479) in
[libcontainer](https://github.com/docker/libcontainer) and made available through [runC](https://runc.io):
we intend to take advantage of it in the Engine.

# 2 Frozen features

## 2.1 Docker exec

We won't accept patches expanding the surface of `docker exec`, which we intend to keep as a
*debugging* feature, as well as being strongly dependent on the the Runtime ingredient effort.

## 2.2 Dockerfile syntax

The Dockerfile syntax as we know it is simple, and has proven succesful in supporting all our
[official images](https://github.com/docker-library/official-images). Although this is *not* a
definitive move, we temporarily won't accept more patches to the Dockerfile syntax for several
reasons:

- Long term impact of syntax changes is a sensitive matter that require an amount of attention
the volume of Engine codebase and activity today doesn't allow us to provide.
- Allowing the Builder to be implemented as a separate utility consuming the Engine's API will
open the door for many possibilities, such as offering alternate syntaxes or DSL for existing
languages without cluttering the Engine's codebase.
- A standalone Builder will also offer the opportunity for a better dedicated group of maintainers
to own the Dockerfile syntax and decide collectively on the direction to give it.
- Our experience with official images tend to show that no new instruction or syntax expansion is
*strictly* necessary for the majority of use cases, and although we are aware many things are still
lacking for many, we cannot make it a priority yet for the above reasons.

Again, this is not about saying that the Dockerfile syntax is done, it's about making choices about
what we want to do first!
