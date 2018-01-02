# Scope and Principles

Having a clearly defined scope of a project is important for ensuring consistency and focus.
These following criteria will be used when reviewing pull requests, features, and changes for the project before being accepted.

### Components

Components should not have tight dependencies on each other so that they are able to be used independently.
The APIs for images and containers should be designed in a way that when used together the components have a natural flow but still be useful independently.

An example for this design can be seen with the overlay filesystems and the container execution layer.
The execution layer and overlay filesystems can be used independently but if you were to use both, they share a common `Mount` struct that the filesystems produce and the execution layer consumes.

### Primitives

containerd should expose primitives to solve problems instead of building high level abstractions in the API.
A common example of this is how build would be implemented.
Instead of having a build API in containerd we should expose the lower level primitives that allow things required in build to work.
Breaking up the filesystem APIs to allow snapshots, copy functionality, and mounts allow people implementing build at the higher levels with more flexibility.

### Extensibility and Defaults

For the various components in containerd there should be defined extension points where implementations can be swapped for alternatives.
The best example of this is that containerd will use `runc` from OCI as the default runtime in the execution layer but other runtimes conforming to the OCI Runtime specification can be easily added to containerd.

containerd will come with a default implementation for the various components.
These defaults will be chosen by the maintainers of the project and should not change unless better tech for that component comes out.
Additional implementations will not be accepted into the core repository and should be developed in a separate repository not maintained by the containerd maintainers.


## Scope

The following table specifies the various components of containerd and general features of container runtimes.
The table specifies whether or not the feature/component is in or out of scope.

| Name | Description | In/Out | Reason |
|------------------------------|--------------------------------------------------------------------------------------------------------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| execution | Provide an extensible execution layer for executing a container | in | Create,start, stop pause, resume exec, signal, delete |
| cow filesystem | Built in functionality for overlay, aufs, and other copy on write filesystems for containers | in |  |
| distribution | Having the ability to push and pull images as well as operations on images as a first class API object | in | containerd will fully support the management and retrieval of images |
| metrics | container-level metrics, cgroup stats, and OOM events | in |
| networking | creation and management of network interfaces | out | Networking will be handled and provided to containerd via higher level systems. |
| build | Building images as a first class API | out | Build is a higher level tooling feature and can be implemented in many different ways on top of containerd |
| volumes | Volume management for external data | out | The API supports mounts, binds, etc where all volumes type systems can be built on top of containerd. |
| logging | Persisting container logs | out | Logging can be build on top of containerd because the container’s STDIO will be provided to the clients and they can persist any way they see fit. There is no io copying of container STDIO in containerd. |


containerd is scoped to a single host and makes assumptions based on that fact.
It can be used to build things like a node agent that launches containers but does not have any concepts of a distributed system.

containerd is designed to be embedded into a larger system, hence it only includes a barebone CLI (`ctr`) specifically for development and debugging purpose, with no mandate to be human-friendly, and no guarantee of interface stability over time.

### How is the scope changed?

The scope of this project is a whitelist.
If it's not mentioned as being in scope, it is out of scope.
For the scope of this project to change it requires a 100% vote from all maintainers of the project.
