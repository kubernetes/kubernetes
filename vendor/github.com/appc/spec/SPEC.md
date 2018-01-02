# App Container Specification

_For version information, see [VERSION](VERSION)_

## Overview

"App Container" (appc) is a specification describing how applications can be packaged, distributed, and executed in a portable and self-contained way.
The specification defines an **image format**, an **image discovery mechanism**, a **deployable grouping**, and an **execution environment**.

The core goals of the specification include:

* Designing for fast downloads and starts of App Containers
* Ensuring images are cryptographically verifiable and highly cacheable
* Designing for composability and independent implementations
* Using common technologies for cryptography, archiving, compression and transport
* Using the DNS namespace to name and discover images

### Requirements

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this and other documents of the specification are to be interpreted as described in [RFC2119](http://tools.ietf.org/html/rfc2119).

## Sections

The specification consists of several key sections; the goal is that each can be implemented independently, but are composable with one another.

1. The **[App Container Image](spec/aci.md)** defines: how files are assembled together into a single image, verified on download and placed onto disk to be run.

2. **[App Container Image Discovery](spec/discovery.md)** defines: how to take a name like `example.com/reduce-worker` and translate that into a downloadable image.

3. The **[App Container Pod](spec/pods.md)** (or "Pod") defines: how one or more App Container Images are grouped into a deployable, executable unit.

4. The **[App Container Executor](spec/ace.md)** defines: how pods are executed and the environment they are run inside (including, for example, filesystem layout, resource constraints, and networking).

    * The [Metadata Service](spec/ace.md#app-container-metadata-service) defines how apps within pods can introspect and get a cryptographically verifiable identity from the execution environment.


## Example Use Case

Here's an example use case demonstrating how the different sections of the specification could be utilized together.

A user wants to launch an "App Container" running three processes.
The three processes the user wants to run are the apps named `example.com/reduce-worker-register`, `example.com/reduce-worker`, and `example.com/reduce-backup`.
First, an executor will make an HTTPS request to example.com and, on inspecting the `<meta>` tags in the returned page, determines that the images can be found at:

	https://storage-mirror.example.com/reduce-worker.aci
	https://storage-mirror.example.com/worker-backup.aci
	https://storage-mirror.example.com/reduce-worker-register.aci

The executor downloads these three images and puts them into its local on-disk cache.
Then the executor extracts three fresh copies of the images to create instances of the "on-disk app format" and reads the three image manifests to figure out what binaries will need to be executed.

Based on user input, the executor now sets up the necessary cgroups, network interfaces, etc. and runs the `pre-start` event handlers for each app.
Next, it forks the `reduce-worker`, `worker-backup`, and `register` processes in their shared namespaces, chrooted into their respective root filesystems.

At some point, the App Container gets some notification that it needs to stop (for example, upon host shutdown).
The executor sends `SIGTERM` to the processes.
After they have exited, the executor runs the `post-stop` event handlers for each app.
