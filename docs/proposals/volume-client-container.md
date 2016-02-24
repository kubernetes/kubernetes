# Containerized Volume Clients for Kubernetes
This document presents user stories, design and examples for containerized volume clients.

# Motivation
Many storage systems require specific software to connect.  The current volume architecture requires volume client software installed on each host for the kubelet to access volumes.  Its an common mistake to forget to install the packages on one or more hosts.  In addition, its often difficult to install the packages potentially impossible on some OS.

1. No client installed on host
2. No client configuration on host.


# Roles and Definitions
* **Platform** Host environment running kubernetes (Ex: Fedora-atomic, CentOS)
* **Platform Maintainer** - Maintains host platform.
* **Framework** - Kubernetes or applications packaging Kubernetes
* **Framework Maintainer** - Builds, packages, bundles or distributes framework.
* **Storage Framework Maintainer** - Owns or distributes a specific storage solution.
* **Storage Administrator** - Creates, owns and adminsters a deployed storage system independent of kubernetes.
* **Kubernetes Administrator** - Maintains Kubernetes cluster with elevated permissions compared to User.
* **User** - End user of Kubernetes.  Submits applications to running framework.
* **Tire Kicker** - Single/simple deployer of Kubernetes.  Runs example, non production use cases with the combined responsibilities of Storage, Kube and Platform administrator and end user.
* **VCS** - Volume (Client/Control) Software. Any packages or utilities required to mount/read/write/admin a storage volume. Provided by host platform.
* **Storage Control Plane** - Storage Administrator operation interface for create/delete/modify type functions of the storage subsystem external from Kubernetes. Example: Browser interface to SAN fabric configuration.


# User Stories

* **Story 1**: As a Framework maintainer, I want to provide a no-setup storage experience for non-production environments.

* **Story 2**: As a Framework maintainer, I want to allow custom configuration for production environments.

* **Story  3**: As a Kubernetes Administrator I do not want to install volume drivers.

* **Story  4**: As a Framework Maintainer I want to reduce kube configuration required to use kubernetes storage.

* **Story  5**: As a Framework Maintainer I want to reduce/eliminate platform configuration required to use storage.

* **Story  6**: As a Framework Maintainer I want to provide VCS support for volume types that works out-of-the-box.

* **Story  7**: As a Framework Maintainer I want a workable zero-configuration out-of-the-box experience.

* **Story  8**: As a Framework Maintainer I want applications to be portable across Platforms.

* **Story  9**: As a Framework Maintainer I want supported volumes to work regardless of Platform.

* **Story  10**: As a Storage Framework Maintainer I want to provide enhanced or custom supported VCS.

* **Story  11**: As a Storage Framework Maintainer I want to provide out-of-tree VCS.

* **Story  12**: As a Storage Administrator I want a deliberate upgrade path for VCS.

* **Story  13**: As a Storage Framework Maintainer my storage system requires the VCS to continue running while the storage is being used. (TSR/Daemon requried for client)

* **Story  14**: As a Platform Matinaer I want the VCS containerized

* **Story  15**: As a Platform Maintainer I do not want the VCS containerized or managed by the Framework. The platform will manage the VCS.

* **Story  16**: As a tire-kicker I want to run network storage examples with no additional configuration.

* **Story  17**: As a tire-kicker I want to run network storage examples with no manual VCS configuration

# Design
Some storage volumes are supported by default in the host kernel but others require specific software packages (VCS). The VCS is typically dependent on the host platform, which presents configuration and administration difficulties for the Framework Maintainer.

By containerizing the VCS they become portable across multiple platforms. A containerized VCS can mount the volume then export the mount to other containers eliminating the need for VCS installed on the host.

# Mount Namespace
The platform maintains a root mount namespace and each new user or container instance creates a slave namespace.  Mounts in the root namespace are visible to slaves, but slave namespace mounts are not visible to the root or other slave mount namespaces by default.

With Docker 1.10 its possible to share a slave mount namespace with the root namespace.  By sharing the namespace, one container can mount a filesystem thats visible by another container.

# Configuration
As of Kubernetes 1.2, the volume plugins expect that the VCS is installed on the host.  *Each plugin will have to opt-in to container usage* this may leave a undeseriable "out of the box" experience until the default is changed to opt-out. This design prsents a pattern which will be added to plugins independently.

For greater flexibility each plugin will specify whether to support containerized or host based mount, or both.  If containerized mount is not specified, then host based mount is assumed.  If containerized mount is supported, the container images can either retrieved from ConfigMap like the following:

--nfs_mount_container="k8/nfs_container:latest"


# Container State #
The volume plugin will manage the containers state if necesarry.  If a container must remain running while the mount is available, the container should ensure as such.  Once the volume is cleaned the container may be released.

# Container Design
In many cases its as simple as creating a container with the entire VCS and executing the same mount command that would have been executed on host.  This allows a single mount command to work both on a host VCS or containerized VCS.

With other scenarios it may be desirable to override the mount command, or continue running.  Each in-tree plugin should handle this on a case-by-case basis.

# Limitation

FUSE mount is not supported in the first pass. 

# Design Alternatives


