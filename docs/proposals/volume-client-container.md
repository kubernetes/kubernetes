# Containerized Volume Clients for Kubernetes
This document presents user stories and design for containerized volume clients.

# Motivation
Many storage systems require specific software pakcages to connect.  The current volume architecture requires that volume client software (VCS) is installed on each host for kubelet to access the volumes.  Its a common mistake to forget to install packages on one or more hosts.  And on some platforms the volume client configuration is difficult or impossible.

**Benefits:**

1. No client installed on host.
2. No client configuration on host.

# Roles and Definitions
* **Platform** - Host environment running kubernetes (Ex: Fedora-atomic, CentOS)
* **Platform Maintainer** - Maintains host platform.
* **Framework** - Kubernetes or applications packaging Kubernetes
* **Framework Maintainer** - Builds, packages, bundles or distributes framework.
* **Storage Framework Maintainer** - Owns or distributes a specific storage solution.
* **Storage Administrator** - Creates, owns and adminsters a deployed storage system independent of kubernetes.
* **Kubernetes Administrator** - Maintains Kubernetes cluster with elevated permissions compared to User.
* **User** - End user of Kubernetes.  Submits applications to running framework.
* **Tire Kicker** - Single/simple deployer of Kubernetes.  Runs example, non production use cases with the combined responsibilities of Storage, Kube and Platform administrator and end user.
* **VCS** - Volume (Client/Control) Software. Any packages or utilities required to mount/read/write/admin a storage volume. Provided by host platform.

# Primary User Story
* **Story 0**: As a Kubernetes Administrator I do not want to install VCS.

# Detailed User Stories

* **Story 1**: As a Framework maintainer, I want to provide a no-setup storage experience for non-production environments.

* **Story 2**: As a Framework maintainer, I want to allow custom configuration for production environments.

* **Story 3**: As a Framework Maintainer I want to reduce kube configuration required to use kubernetes storage.

* **Story 4**: As a Framework Maintainer I want to reduce/eliminate platform configuration required to use storage.

* **Story 5**: As a Framework Maintainer I want to provide VCS support for volume types that works out-of-the-box.

* **Story 6**: As a Framework Maintainer I want a workable zero-configuration out-of-the-box experience.

* **Story 7**: As a Framework Maintainer I want applications to be portable across Platforms.

* **Story 8**: As a Framework Maintainer I want supported volumes to work regardless of Platform.

* **Story 9**: As a Storage Framework Maintainer I want to provide enhanced or custom supported VCS.

* **Story  10**: As a Storage Framework Maintainer I want to provide out-of-tree VCS.

* **Story 11**: As a Storage Administrator I want a deliberate upgrade path for VCS.

* **Story 12**: As a Storage Framework Maintainer my storage system requires the VCS to continue running while the storage is being used. (TSR/Daemon requried for client)

* **Story 13**: As a Platform Maintainer I want the VCS containerized

* **Story 14**: As a Platform Maintainer I do not want the VCS containerized or managed by the Framework. The platform will manage the VCS.

* **Story 15**: As a tire-kicker I want to run network storage examples with no additional configuration.

* **Story 16**: As a tire-kicker I want to run network storage examples with no manual VCS configuration

* **Story 17**: As a Framework Maintainer I want to containerized the Framework.

* **Story 18**: As a Framework Maintainer I want the volume end-to-end tests to be portable across platforms.

# Design
Some storage volumes are supported by default in the platforms kernel while others require specific software packages. The exact VCS depends on the platform which presents configuration and administration difficulties for the Framework Maintainer.

By containerizing the VCS they become portable across multiple platforms. A containerized VCS can mount the volume then export the mount to other containers eliminating the need for VCS installed on the host.

# Mount Namespace
The platform maintains a root mount namespace and each new user or container instance creates a slave namespace.  Mounts in the root namespace are visible to slaves but slave namespace mounts are not visible to the root or other slave mount namespaces by default.

With Docker 1.10 its possible to share a slave mount namespace with the root namespace.  By sharing the namespace, one container can mount a filesystem thats visible by another container.

# Container "Sidecar"
When a pod's container(s) needs access to a volume type who's client runs in a container its required that the VCS conatainer runs on the same host.  This can be achieved by enhancing  Kubelet to be aware of the VCS container, or by launching a pod targeted at the specific node.

This design proposes enhancing Kubelet with knowledge of the VCS for these reasons:
1. Pod lifecycle is overly heavy 
2. VCS Pods would show up in status or must be filtered out.
3. Codependency of the pods (VCS and main) would require a controller to manage. 
 
# Configuration
As of Kubernetes 1.2, the volume plugins expect that the VCS is installed on the host.  **Each plugin will have to opt-in to container usage** this may leave a undeseriable "out of the box" experience (User Stories 1, 15, 16) until the default is changed to opt-out. This design prsents a pattern which will be added to plugins independently.

For greater flexibility each plugin will specify whether to support containerized VCS, host installed VCS or both on a case-by-case basis and enhanced as contributed by the community.  If containerized VCS is not specified then host based VCS is assumed for all plugins.  If containerized VCS is supported the container image configuration can either retrieved from ConfigMap or with the following:

--nfs_mount_container="k8/nfs_container:latest"

**ConfigMap:**
``` yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: volume-client
data:
  volume-plugin: "nfs"
  docker-container: "k8io/nfs:latest"
```

# Container State 
The volume plugin will manage its containers state if necesarry.  If a container must remain running while the mount is available the container should ensure as such.  Once the volume is unbound the container may be released.

# Container Design
In many cases its as simple as creating a container with the entire VCS and executing the same mount command that would have been executed on host.  This allows a single mount command to work both on a host VCS or containerized VCS.

With other scenarios it may be desirable to override the mount command or continue running.  Each in-tree plugin should handle this on a case-by-case basis.

# Containerized Kubelet
When the kubelet runs in a container, nsenter is used to escape the container mount namespace, and runs all mounts on the host.  With the addition of shared docker namespace we can remove the nsenter function in kubelet and modify mount.go

1. Delete nsenter_mount.go (eventually, backward compatibility)
2. Eliminate the distinction in the code with regard to mount operations of whether the kubelet is containerized or not

# Limitations
FUSE mount is not supported in the first pass (user story 12). 



