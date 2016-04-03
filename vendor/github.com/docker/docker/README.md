Docker: the Linux container engine
==================================

Docker is an open source project to pack, ship and run any application
as a lightweight container.

Docker containers are both *hardware-agnostic* and *platform-agnostic*.
This means they can run anywhere, from your laptop to the largest
EC2 compute instance and everything in between - and they don't require
you to use a particular language, framework or packaging system. That
makes them great building blocks for deploying and scaling web apps,
databases, and backend services without depending on a particular stack
or provider.

Docker began as an open-source implementation of the deployment engine which
powers [dotCloud](https://dotcloud.com), a popular Platform-as-a-Service.
It benefits directly from the experience accumulated over several years
of large-scale operation and support of hundreds of thousands of
applications and databases.

![Docker L](docs/static_files/docker-logo-compressed.png "Docker")

## Security Disclosure

Security is very important to us. If you have any issue regarding security, 
please disclose the information responsibly by sending an email to 
security@docker.com and not by creating a github issue.

## Better than VMs

A common method for distributing applications and sandboxing their
execution is to use virtual machines, or VMs. Typical VM formats are
VMware's vmdk, Oracle VirtualBox's vdi, and Amazon EC2's ami. In theory
these formats should allow every developer to automatically package
their application into a "machine" for easy distribution and deployment.
In practice, that almost never happens, for a few reasons:

  * *Size*: VMs are very large which makes them impractical to store
     and transfer.
  * *Performance*: running VMs consumes significant CPU and memory,
    which makes them impractical in many scenarios, for example local
    development of multi-tier applications, and large-scale deployment
    of cpu and memory-intensive applications on large numbers of
    machines.
  * *Portability*: competing VM environments don't play well with each
     other. Although conversion tools do exist, they are limited and
     add even more overhead.
  * *Hardware-centric*: VMs were designed with machine operators in
    mind, not software developers. As a result, they offer very
    limited tooling for what developers need most: building, testing
    and running their software. For example, VMs offer no facilities
    for application versioning, monitoring, configuration, logging or
    service discovery.

By contrast, Docker relies on a different sandboxing method known as
*containerization*. Unlike traditional virtualization, containerization
takes place at the kernel level. Most modern operating system kernels
now support the primitives necessary for containerization, including
Linux with [openvz](https://openvz.org),
[vserver](http://linux-vserver.org) and more recently
[lxc](https://linuxcontainers.org/), Solaris with
[zones](https://docs.oracle.com/cd/E26502_01/html/E29024/preface-1.html#scrolltoc),
and FreeBSD with
[Jails](https://www.freebsd.org/doc/handbook/jails.html).

Docker builds on top of these low-level primitives to offer developers a
portable format and runtime environment that solves all four problems.
Docker containers are small (and their transfer can be optimized with
layers), they have basically zero memory and cpu overhead, they are
completely portable, and are designed from the ground up with an
application-centric design.

Perhaps best of all, because Docker operates at the OS level, it can still be
run inside a VM!

## Plays well with others

Docker does not require you to buy into a particular programming
language, framework, packaging system, or configuration language.

Is your application a Unix process? Does it use files, tcp connections,
environment variables, standard Unix streams and command-line arguments
as inputs and outputs? Then Docker can run it.

Can your application's build be expressed as a sequence of such
commands? Then Docker can build it.

## Escape dependency hell

A common problem for developers is the difficulty of managing all
their application's dependencies in a simple and automated way.

This is usually difficult for several reasons:

  * *Cross-platform dependencies*. Modern applications often depend on
    a combination of system libraries and binaries, language-specific
    packages, framework-specific modules, internal components
    developed for another project, etc. These dependencies live in
    different "worlds" and require different tools - these tools
    typically don't work well with each other, requiring awkward
    custom integrations.

  * *Conflicting dependencies*. Different applications may depend on
    different versions of the same dependency. Packaging tools handle
    these situations with various degrees of ease - but they all
    handle them in different and incompatible ways, which again forces
    the developer to do extra work.

  * *Custom dependencies*. A developer may need to prepare a custom
    version of their application's dependency. Some packaging systems
    can handle custom versions of a dependency, others can't - and all
    of them handle it differently.


Docker solves the problem of dependency hell by giving the developer a simple
way to express *all* their application's dependencies in one place, while
streamlining the process of assembling them. If this makes you think of
[XKCD 927](https://xkcd.com/927/), don't worry. Docker doesn't
*replace* your favorite packaging systems. It simply orchestrates
their use in a simple and repeatable way. How does it do that? With
layers.

Docker defines a build as running a sequence of Unix commands, one
after the other, in the same container. Build commands modify the
contents of the container (usually by installing new files on the
filesystem), the next command modifies it some more, etc. Since each
build command inherits the result of the previous commands, the
*order* in which the commands are executed expresses *dependencies*.

Here's a typical Docker build process:

```bash
FROM ubuntu:12.04
RUN apt-get update && apt-get install -y python python-pip curl
RUN curl -sSL https://github.com/shykes/helloflask/archive/master.tar.gz | tar -xzv
RUN cd helloflask-master && pip install -r requirements.txt
```

Note that Docker doesn't care *how* dependencies are built - as long
as they can be built by running a Unix command in a container.


Getting started
===============

Docker can be installed on your local machine as well as servers - both
bare metal and virtualized.  It is available as a binary on most modern
Linux systems, or as a VM on Windows, Mac and other systems.

We also offer an [interactive tutorial](https://www.docker.com/tryit/)
for quickly learning the basics of using Docker.

For up-to-date install instructions, see the [Docs](https://docs.docker.com).

Usage examples
==============

Docker can be used to run short-lived commands, long-running daemons
(app servers, databases, etc.), interactive shell sessions, etc.

You can find a [list of real-world
examples](https://docs.docker.com/examples/) in the
documentation.

Under the hood
--------------

Under the hood, Docker is built on the following components:

* The
  [cgroups](https://www.kernel.org/doc/Documentation/cgroups/cgroups.txt)
  and
  [namespaces](http://man7.org/linux/man-pages/man7/namespaces.7.html)
  capabilities of the Linux kernel
* The [Go](https://golang.org) programming language
* The [Docker Image Specification](https://github.com/docker/docker/blob/master/image/spec/v1.md)
* The [Libcontainer Specification](https://github.com/docker/libcontainer/blob/master/SPEC.md)

Contributing to Docker
======================

[![GoDoc](https://godoc.org/github.com/docker/docker?status.svg)](https://godoc.org/github.com/docker/docker)
[![Jenkins Build Status](https://jenkins.dockerproject.org/job/Docker%20Master/badge/icon)](https://jenkins.dockerproject.org/job/Docker%20Master/)

Want to hack on Docker? Awesome! We have [instructions to help you get
started contributing code or documentation](https://docs.docker.com/project/who-written-for/).

These instructions are probably not perfect, please let us know if anything
feels wrong or incomplete. Better yet, submit a PR and improve them yourself.

Getting the development builds
==============================

Want to run Docker from a master build? You can download 
master builds at [master.dockerproject.org](https://master.dockerproject.org). 
They are updated with each commit merged into the master branch.

Don't know how to use that super cool new feature in the master build? Check
out the master docs at
[docs.master.dockerproject.org](http://docs.master.dockerproject.org).

How the project is run
======================

Docker is a very, very active project. If you want to learn more about how it is run,
or want to get more involved, the best place to start is [the project directory](https://github.com/docker/docker/tree/master/project).

We are always open to suggestions on process improvements, and are always looking for more maintainers.

### Talking to other Docker users and contributors

<table class="tg">
  <col width="45%">
  <col width="65%">
  <tr>
    <td>Internet&nbsp;Relay&nbsp;Chat&nbsp;(IRC)</td>
    <td>
      <p>
        IRC a direct line to our most knowledgeable Docker users; we have
        both the  <code>#docker</code> and <code>#docker-dev</code> group on
        <strong>irc.freenode.net</strong>.
        IRC is a rich chat protocol but it can overwhelm new users. You can search
        <a href="https://botbot.me/freenode/docker/#" target="_blank">our chat archives</a>.
      </p>
      Read our <a href="https://docs.docker.com/project/get-help/#irc-quickstart" target="_blank">IRC quickstart guide</a> for an easy way to get started.
    </td>
  </tr>
  <tr>
    <td>Google Groups</td>
    <td>
      There are two groups.
      <a href="https://groups.google.com/forum/#!forum/docker-user" target="_blank">Docker-user</a>
      is for people using Docker containers.
      The <a href="https://groups.google.com/forum/#!forum/docker-dev" target="_blank">docker-dev</a>
      group is for contributors and other people contributing to the Docker
      project.
    </td>
  </tr>
  <tr>
    <td>Twitter</td>
    <td>
      You can follow <a href="https://twitter.com/docker/" target="_blank">Docker's Twitter feed</a>
      to get updates on our products. You can also tweet us questions or just
      share blogs or stories.
    </td>
  </tr>
  <tr>
    <td>Stack Overflow</td>
    <td>
      Stack Overflow has over 7000K Docker questions listed. We regularly
      monitor <a href="https://stackoverflow.com/search?tab=newest&q=docker" target="_blank">Docker questions</a>
      and so do many other knowledgeable Docker users.
    </td>
  </tr>
</table>

### Legal

*Brought to you courtesy of our legal counsel. For more context,
please see the [NOTICE](https://github.com/docker/docker/blob/master/NOTICE) document in this repo.*

Use and transfer of Docker may be subject to certain restrictions by the
United States and other governments.

It is your responsibility to ensure that your use and/or transfer does not
violate applicable laws.

For more information, please see https://www.bis.doc.gov


Licensing
=========
Docker is licensed under the Apache License, Version 2.0. See
[LICENSE](https://github.com/docker/docker/blob/master/LICENSE) for the full
license text.

Other Docker Related Projects
=============================
There are a number of projects under development that are based on Docker's
core technology. These projects expand the tooling built around the
Docker platform to broaden its application and utility.

* [Docker Registry](https://github.com/docker/distribution): Registry 
server for Docker (hosting/delivery of repositories and images)
* [Docker Machine](https://github.com/docker/machine): Machine management 
for a container-centric world
* [Docker Swarm](https://github.com/docker/swarm): A Docker-native clustering 
system
* [Docker Compose](https://github.com/docker/compose) (formerly Fig): 
Define and run multi-container apps
* [Kitematic](https://github.com/kitematic/kitematic): The easiest way to use 
Docker on Mac and Windows

If you know of another project underway that should be listed here, please help 
us keep this list up-to-date by submitting a PR.
