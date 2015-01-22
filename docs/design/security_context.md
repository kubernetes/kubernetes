# Security Contexts
## Abstract
A security context is a set of constraints that are applied to a container in order to achieve the following goals (from [security design](security.md)):

1.  Ensure a clear isolation between container and the underlying host it runs on
2.  Limit the ability of the container to negatively impact the infrastructure or other containers

## Background

The problem of securing containers in Kubernetes has come up [before](https://github.com/GoogleCloudPlatform/kubernetes/issues/398) and the potential problems with container security are [well known](http://opensource.com/business/14/7/docker-security-selinux). Although it is not possible to completely isolate Docker containers from their hosts, new features like [user namespaces](https://github.com/docker/libcontainer/pull/304) make it possible to greatly reduce the attack surface.

## Motivation

### Container isolation

In order to improve container isolation from host and other containers running on the host, containers should only be 
granted the access they need to perform their work. To this end it should be possible to take advantage of Docker 
features such as the ability to [add or remove capabilities](https://docs.docker.com/reference/run/#runtime-privilege-linux-capabilities-and-lxc-configuration) and [assign MCS labels](https://docs.docker.com/reference/run/#security-configuration) 
to the container process.

Support for user namespaces has recently been [merged](https://github.com/docker/libcontainer/pull/304) into Docker's libcontainer project and should soon surface in Docker itself. It will make it possible to assign a range of unprivileged uids and gids from the host to each container, improving the isolation between host and container and between containers.

### External integration with shared storage
In order to support external integration with shared storage, processes running in a Kubernetes cluster 
should be able to be uniquely identified by their Unix UID, such that a chain of  ownership can be established. 
Processes in pods will need to have consistent UID/GID/SELinux category labels in order to access shared disks.

## Constraints and Assumptions
* It is out of the scope of this document to prescribe a specific set 
  of constraints to isolate containers from their host. Different use cases need different
  settings.
* The concept of a security context should not be tied to a particular security mechanism or platform 
  (ie. SELinux, AppArmor)
* Applying a different security context to a scope (namespace or pod) requires a solution such as the one proposed for
  [service accounts](https://github.com/GoogleCloudPlatform/kubernetes/pull/2297).

## Use Cases

In order of increasing complexity, following are example use cases that would 
be addressed with security contexts:

1.  Kubernetes is used to run a single cloud application. In order to protect
    nodes from containers:
    * All containers run as a single non-root user
    * Privileged containers are disabled
    * All containers run with a particular MCS label 
    * Kernel capabilities like CHOWN and MKNOD are removed from containers
    
2.  Just like case #1, except that I have more than one application running on
    the Kubernetes cluster.
    * Each application is run in its own namespace to avoid name collisions
    * For each application a different uid and MCS label is used
    
3.  Kubernetes is used as the base for a PAAS with 
    multiple projects, each project represented by a namespace. 
    * Each namespace is associated with a range of uids/gids on the node that
      are mapped to uids/gids on containers using linux user namespaces. 
    * Certain pods in each namespace have special privileges to perform system
      actions such as talking back to the server for deployment, run docker
      builds, etc.
    * External NFS storage is assigned to each namespace and permissions set
      using the range of uids/gids assigned to that namespace. 

## Proposed Design

### Overview
A *security context* consists of a set of constraints that determine how a container
is secured before getting created and run. It has a 1:1 correspondence to a
[service account](https://github.com/GoogleCloudPlatform/kubernetes/pull/2297). A *security context provider* is passed to the Kubelet so it can have a chance
to mutate Docker API calls in order to apply the security context.

It is recommended that this design be implemented in two phases:

1.  Implement the security context provider extension point in the Kubelet 
    so that a default security context can be applied on container run and creation.
2.  Implement a security context structure that is part of a service account. The
    default context provider can then be used to apply a security context based
    on the service account associated with the pod.
    
### Security Context Provider

The Kubelet will have an interface that points to a `SecurityContextProvider`. The `SecurityContextProvider` is invoked before creating and running a given container:

```go
type SecurityContextProvider interface {
    // ModifyContainerConfig is called before the Docker createContainer call.
    // The security context provider can make changes to the Config with which
    // the container is created.
    // An error is returned if it's not possible to secure the container as 
    // requested with a security context. 
	ModifyContainerConfig(pod *api.BoundPod, container *api.Container, config *docker.Config) error
	
	// ModifyHostConfig is called before the Docker runContainer call.
	// The security context provider can make changes to the HostConfig, affecting
	// security options, whether the container is privileged, volume binds, etc.
	// An error is returned if it's not possible to secure the container as requested 
    // with a security context. 
	ModifyHostConfig(pod *api.BoundPod, container *api.Container, hostConfig *docker.HostConfig)
}
```
If the value of the SecurityContextProvider field on the Kubelet is nil, the kubelet will create and run the container as it does today.   

### Security Context

A security context has a 1:1 correspondence to a service account and it can be included as
part of the service account resource. Following is an example of an initial implementation:

```go
type SecurityContext struct {
    // user is the uid to use when running the container
	User int
	
	// allowPrivileged indicates whether this context allows privileged mode containers
	AllowPrivileged bool
	
	// allowedVolumeTypes lists the types of volumes that a container can bind
	AllowedVolumeTypes []string
	
	// addCapabilities is the list of Linux kernel capabilities to add
	AddCapabilities []string
	
	// removeCapabilities is the list of Linux kernel capabilities to remove
	RemoveCapabilities []string
	
	// SELinux specific settings (optional)
	SELinux *SELinuxContext
	
	// AppArmor specific settings (optional)
	AppArmor *AppArmorContext
	
	// FUTURE:
	// With Linux user namespace support, it should be possible to map
	// a range of container uids/gids to arbitrary host uids/gids
	// UserMappings []IDMapping
	// GroupMappings []IDMapping
}

type SELinuxContext struct {
    // MCS label/SELinux level to run the container under
    Level string
    
    // SELinux type label for container processes
    Type  string    
    
    // FUTURE:
    // LabelVolumeMountsExclusive []Volume
    // LabelVolumeMountsShared    []Volume
}

type AppArmorContext struct {
	// AppArmor profile
	Profile string
}
```

#### Security Context Lifecycle
 
The lifecycle of a security context will be tied to that of a service account. It is expected that a service account with a default security context will be created for every Kubernetes namespace (without administrator intervention). If resources need to be allocated when creating a security context (for example, assign a range of host uids/gids), a pattern such as [finalizers](https://github.com/GoogleCloudPlatform/kubernetes/issues/3585) can be used before declaring the security context / service account / namespace ready for use.