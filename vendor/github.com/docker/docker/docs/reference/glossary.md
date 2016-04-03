<!--[metadata]>
+++
title = "Docker Glossary"
description = "Glossary of terms used around Docker"
keywords = ["glossary, docker, terms,  definitions"]
[menu.main]
parent = "mn_about"
weight = "50"
+++
<![end-metadata]-->

# Glossary

A list of terms used around the Docker project.

## aufs

aufs (advanced multi layered unification filesystem) is a Linux [filesystem](#filesystem) that
Docker supports as a storage backend. It implements the
[union mount](http://en.wikipedia.org/wiki/Union_mount) for Linux file systems.

## boot2docker

[boot2docker](http://boot2docker.io/) is a lightweight Linux distribution made
specifically to run Docker containers. It is a common choice for a [VM](#virtual-machine)
to run Docker on Windows and Mac OS X.

boot2docker can also refer to the boot2docker management tool on Windows and
Mac OS X which manages the boot2docker VM.

## btrfs

btrfs (B-tree file system) is a Linux [filesystem](#filesystem) that Docker
supports as a storage backend. It is a [copy-on-write](http://en.wikipedia.org/wiki/Copy-on-write)
filesystem.

## build

build is the process of building Docker images using a [Dockerfile](#dockerfile).
The build uses a Dockerfile and a "context". The context is the set of files in the
directory in which the image is built.

## cgroups

cgroups is a Linux kernel feature that limits, accounts for, and isolates
the resource usage (CPU, memory, disk I/O, network, etc.) of a collection
of processes. Docker relies on cgroups to control and isolate resource limits.

*Also known as : control groups*

## Compose

[Compose](https://github.com/docker/compose) is a tool for defining and
running complex applications with Docker. With compose, you define a
multi-container application in a single file, then spin your
application up in a single command which does everything that needs to
be done to get it running.

*Also known as : docker-compose, fig*

## container

A container is a runtime instance of a [docker image](#image).

A Docker container consists of

- A Docker image
- Execution environment
- A standard set of instructions

The concept is borrowed from Shipping Containers, which define a standard to ship
goods globally. Docker defines a standard to ship software.

## data volume

A data volume is a specially-designated directory within one or more containers
that bypasses the Union File System. Data volumes are designed to persist data,
independent of the container's life cycle. Docker therefore never automatically
delete volumes when you remove a container, nor will it "garbage collect"
volumes that are no longer referenced by a container.


## Docker

The term Docker can refer to

- The Docker project as a whole, which is a platform for developers and sysadmins to
develop, ship, and run applications
- The docker daemon process running on the host which manages images and containers


## Docker Hub

The [Docker Hub](https://hub.docker.com/) is a centralized resource for working with
Docker and its components. It provides the following services:

- Docker image hosting
- User authentication
- Automated image builds and work-flow tools such as build triggers and web hooks
- Integration with GitHub and Bitbucket


## Dockerfile

A Dockerfile is a text document that contains all the commands you would
normally execute manually in order to build a Docker image. Docker can
build images automatically by reading the instructions from a Dockerfile.

## filesystem

A file system is the method an operating system uses to name files
and assign them locations for efficient storage and retrieval.

Examples :

- Linux : ext4, aufs, btrfs, zfs
- Windows : NTFS
- OS X : HFS+

## image

Docker images are the basis of [containers](#container). An Image is an
ordered collection of root filesystem changes and the corresponding
execution parameters for use within a container runtime. An image typically
contains a union of layered filesystems stacked on top of each other. An image
does not have state and it never changes.

## libcontainer

libcontainer provides a native Go implementation for creating containers with
namespaces, cgroups, capabilities, and filesystem access controls. It allows
you to manage the lifecycle of the container performing additional operations
after the container is created.

## link

links provide an interface to connect Docker containers running on the same host
to each other without exposing the hosts' network ports. When you set up a link,
you create a conduit between a source container and a recipient container.
The recipient can then access select data about the source. To create a link,
you can use the `--link` flag.

## Machine

[Machine](https://github.com/docker/machine) is a Docker tool which
makes it really easy to create Docker hosts on  your computer, on
cloud providers and inside your own data center. It creates servers,
installs Docker on them, then configures the Docker client to talk to them.

*Also known as : docker-machine*

## overlay

OverlayFS is a [filesystem](#filesystem) service for Linux which implements a
[union mount](http://en.wikipedia.org/wiki/Union_mount) for other file systems.
It is supported by the Docker daemon as a storage driver.

## registry

A Registry is a hosted service containing [repositories](#repository) of [images](#image)
which responds to the Registry API.

The default registry can be accessed using a browser at [Docker Hub](#docker-hub)
or using the `docker search` command.

## repository

A repository is a set of Docker images. A repository can be shared by pushing it
to a [registry](#registry) server. The different images in the repository can be
labeled using [tags](#tag).

Here is an example of the shared [nginx repository](https://registry.hub.docker.com/_/nginx/)
and its [tags](https://registry.hub.docker.com/_/nginx/tags/manage/)

## Swarm

[Swarm](https://github.com/docker/swarm) is a native clustering tool for Docker.
Swarm pools together several Docker hosts and exposes them as a single virtual
Docker host. It serves the standard Docker API, so any tool that already works
with Docker can now transparently scale up to multiple hosts.

*Also known as : docker-swarm*

## tag

A tag is a label applied to a Docker image in a [repository](#repository).
tags are how various images in a repository are distinguished from each other.

*Note : This label is not related to the key=value labels set for docker daemon*

## Union file system

Union file systems, or UnionFS, are file systems that operate by creating layers, making them
very lightweight and fast. Docker uses union file systems to provide the building
blocks for containers.


## Virtual Machine

A Virtual Machine is a program that emulates a complete computer and imitates dedicated hardware.
It shares physical hardware resources with other users but isolates the operating system. The
end user has the same experience on a Virtual Machine as they would have on dedicated hardware.

Compared to to containers, a Virtual Machine is heavier to run, provides more isolation,
gets its own set of resources and does minimal sharing.

*Also known as : VM*

