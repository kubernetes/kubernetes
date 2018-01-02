### Docker users, see [Moby and Docker](https://mobyproject.org/#moby-and-docker) to clarify the relationship between the projects

### Docker maintainers and contributors, see [Transitioning to Moby](#transitioning-to-moby) for more details

The Moby Project
================

![Moby Project logo](docs/static_files/moby-project-logo.png "The Moby Project")

Moby is an open-source project created by Docker to advance the software containerization movement.
It provides a “Lego set” of dozens of components, the framework for assembling them into custom container-based systems, and a place for all container enthusiasts to experiment and exchange ideas.

# Moby

## Overview

At the core of Moby is a framework to assemble specialized container systems.
It provides:

- A library of containerized components for all vital aspects of a container system: OS, container runtime, orchestration, infrastructure management, networking, storage, security, build, image distribution, etc.
- Tools to assemble the components into runnable artifacts for a variety of platforms and architectures: bare metal (both x86 and Arm); executables for Linux, Mac and Windows; VM images for popular cloud and virtualization providers.
- A set of reference assemblies which can be used as-is, modified, or used as inspiration to create your own.

All Moby components are containers, so creating new components is as easy as building a new OCI-compatible container.

## Principles

Moby is an open project guided by strong principles, but modular, flexible and without too strong an opinion on user experience, so it is open to the community to help set its direction.
The guiding principles are:

- Batteries included but swappable: Moby includes enough components to build fully featured container system, but its modular architecture ensures that most of the components can be swapped by different implementations.
- Usable security: Moby will provide secure defaults without compromising usability.
- Container centric: Moby is built with containers, for running containers.

With Moby, you should be able to describe all the components of your distributed application, from the high-level configuration files down to the kernel you would like to use and build and deploy it easily.

Moby uses [containerd](https://github.com/containerd/containerd) as the default container runtime.

## Audience

Moby is recommended for anyone who wants to assemble a container-based system. This includes:

- Hackers who want to customize or patch their Docker build
- System engineers or integrators building a container system
- Infrastructure providers looking to adapt existing container systems to their environment
- Container enthusiasts who want to experiment with the latest container tech
- Open-source developers looking to test their project in a variety of different systems
- Anyone curious about Docker internals and how it’s built

Moby is NOT recommended for:

- Application developers looking for an easy way to run their applications in containers. We recommend Docker CE instead.
- Enterprise IT and development teams looking for a ready-to-use, commercially supported container platform. We recommend Docker EE instead.
- Anyone curious about containers and looking for an easy way to learn. We recommend the [docker.com](https://www.docker.com/) website instead.

# Transitioning to Moby

Docker is transitioning all of its open source collaborations to the Moby project going forward.
During the transition, all open source activity should continue as usual.

We are proposing the following list of changes:

- splitting up the engine into more open components
- removing the docker UI, SDK etc to keep them in the Docker org
- clarifying that the project is not limited to the engine, but to the assembly of all the individual components of the Docker platform
- open-source new tools & components which we currently use to assemble the Docker product, but could benefit the community
- defining an open, community-centric governance inspired by the Fedora project (a very successful example of balancing the needs of the community with the constraints of the primary corporate sponsor)

-----

Legal
=====

*Brought to you courtesy of our legal counsel. For more context,
please see the [NOTICE](https://github.com/moby/moby/blob/master/NOTICE) document in this repo.*

Use and transfer of Moby may be subject to certain restrictions by the
United States and other governments.

It is your responsibility to ensure that your use and/or transfer does not
violate applicable laws.

For more information, please see https://www.bis.doc.gov


Licensing
=========
Moby is licensed under the Apache License, Version 2.0. See
[LICENSE](https://github.com/moby/moby/blob/master/LICENSE) for the full
license text.
