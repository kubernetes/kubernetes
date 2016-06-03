Kubelet OCI runtime integration
===========================================================
Authors: Mrunal Patel (@mrunalp), Vishnu Kannan (@vishh)

## Abstract
This proposal aims to introduce support for OCI Runtime Specification compatible runtimes in Kubelet.

## Motivation
Kubelet should support the OCI runtime-spec to support Open Standards. OCI Runtime Spec has the primitives to support
creating kubernetes pods. Compatibility with OCI Specification will let kubernetes support any OCI compliant runtimes.

## Design aspects
The following subsections will discuss the various design aspects of the oci-runtime integration with kubelet. For the
purposes of this doc, the kubelet runtime that will provide support for OCI compliant container runtimes will be
referred to as `oci-runtime`.

### Runtime API
The implementation will make use of the kubelet Container Runtime Interface
(https://github.com/kubernetes/kubernetes/pull/25899). The oci-runtime will be implemented against the Kubelet Container
Runtime Interface as a standalone daemon.

### Image management
For the first release, oci-runtime will continue to use docker-engine for managing images.  The image management
functionality will be separated from the runtime functionality so each could have different implementations which could
potentially be switched.  It will be ideal to support the OCI image specification for images once it reaches v1.0. In
addition to the docker image format, other packaging formats like tar files and docker image tar files, etc will also be
supported. Image management requires quite a bit of flexibility in various fronts like transport, content verification,
etc., and this proposal will not delve into those aspects. This proposal essentially opens up a door for easily
extending support for alternate image formats.  Image managers are expected to expose the disk usage of each image and
the overall usage of disk for all images.  To provide better security, it is possible to chown images of each pod to a
separate uid and use user namespaces to have containers in that pod use that uid. This added security comes with a cost,
because it will no longer be possible to share images between different pods and layering of images will no longer make
any sense. 

### Lifecycle
While the integration will implement the kubelet runtime API, we will have the oci-runtime manage the lifecycle of
containers. The lifecycle of the container will not be tied to that of the oci-runtime.  Since the oci-runtime can be
restarted while the containers are running, there needs to be a reliable mechanism to collect the exit code of
containers.  One obvious choice is to use systemd as the parent of these containers.  Another option is to run a wrapper
process and have the wrapper collect the exit code of the container and make it available to the oci-runtime via a named
pipe.  The oci-runtime will provide plugins for lifecycle management and both of the options mentioned above will be
made available.  The kubelet will interact with the runtime to control the lifecycle using the runtime API. Instead of
the declarative SyncPods API, the kubelet will have much finer control over the pod/container lifecycles. 

### Logging
Logging will work by using files to redirect stdout / stderr of the containers. Other higher level drivers could build
on top of files. Using files allow using disk quotas, impose Disk IO limits, and also avoid bottlenecks such as a SPOF
daemons. This approach also let’s Kubelet manage the lifecycle of logs with policies. Whenever there is disk pressure,
the kubelet can:
- Rotate log files of existing containers.
- Prioritize logs from the first and last instance of a container and delete the rest.
- Delete logs from dead pods & containers.
- Impose per-container default Disk IO limits to help fairly share local disk.
- Provide a common extension point for logging daemons to collect logs and metadata from Kubernetes deployments.

### Rootfs management
The oci-runtime will be responsible for creating the rootfs for the containers in a pod. Different kinds of root
filesystems will be supported (such as simple file based, union file systems, read-only with mounts, etc.).

### Networking
Networking should work just like it works today with hooks running in the network namespace of the infra container. Once
the pod sandbox has been setup using an infra container, kubelet will let network plugins update networking options in
the pod sandbox.

### Volumes
Kubelet volumes will also be supported similar to how they work today using bind mounts in OCI Runtime configuration.
Security OCI Runtime Specification covers all the security knobs in the linux kernel and is quick to add support for any
new features that are added to the kernel.

### Container Configuration
Kubelet runtime configuration will be translated to OCI Specification using Ocitools.

### Resource Management
OCI Runtime Specification supports cgroups settings as well as joining cgroups specified by a path. That allows the
kubelet to manage the resources of a pod in a fine grained manner.  Also, there is support for systemd driver for
cgroups, which provides compatibility with distros reliant on systemd. Oci-runtime will comply with the resource
specification provided by the kubelet via the runtime API.

### Development
The new oci-runtime can be developed in a new repository outside of kubernetes core, and can have its own set of
maintainers and roadmap.

