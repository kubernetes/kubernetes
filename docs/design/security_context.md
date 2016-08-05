<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/design/security_context.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Security Contexts

## Abstract

A security context is a set of constraints that are applied to a container in
order to achieve the following goals (from [security design](security.md)):

1.  Ensure a clear isolation between container and the underlying host it runs
on
2.  Limit the ability of the container to negatively impact the infrastructure
or other containers

## Background

The problem of securing containers in Kubernetes has come up
[before](http://issue.k8s.io/398) and the potential problems with container
security are [well known](http://opensource.com/business/14/7/docker-security-selinux).
Although it is not possible to completely isolate Docker containers from their
hosts, new features like [user namespaces](https://github.com/docker/libcontainer/pull/304)
make it possible to greatly reduce the attack surface.

## Motivation

### Container isolation

In order to improve container isolation from host and other containers running
on the host, containers should only be granted the access they need to perform
their work. To this end it should be possible to take advantage of Docker
features such as the ability to
[add or remove capabilities](https://docs.docker.com/reference/run/#runtime-privilege-linux-capabilities-and-lxc-configuration)
and [assign MCS labels](https://docs.docker.com/reference/run/#security-configuration)
to the container process.

Support for user namespaces has recently been
[merged](https://github.com/docker/libcontainer/pull/304) into Docker's
libcontainer project and should soon surface in Docker itself. It will make it
possible to assign a range of unprivileged uids and gids from the host to each
container, improving the isolation between host and container and between
containers.

### External integration with shared storage

In order to support external integration with shared storage, processes running
in a Kubernetes cluster should be able to be uniquely identified by their Unix
UID, such that a chain of  ownership can be established. Processes in pods will
need to have consistent UID/GID/SELinux category labels in order to access
shared disks.

## Constraints and Assumptions

* It is out of the scope of this document to prescribe a specific set of
constraints to isolate containers from their host. Different use cases need
different settings.
* The concept of a security context should not be tied to a particular security
mechanism or platform (ie. SELinux, AppArmor)
* Applying a different security context to a scope (namespace or pod) requires
a solution such as the one proposed for [service accounts](service_accounts.md).

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

3.  Kubernetes is used as the base for a PAAS with multiple projects, each
project represented by a namespace.
    * Each namespace is associated with a range of uids/gids on the node that
are mapped to uids/gids on containers using linux user namespaces.
    * Certain pods in each namespace have special privileges to perform system
actions such as talking back to the server for deployment, run docker builds,
etc.
    * External NFS storage is assigned to each namespace and permissions set
using the range of uids/gids assigned to that namespace.

## Proposed Design

### Overview

A *security context* consists of a set of constraints that determine how a
container is secured before getting created and run. A security context resides
on the container and represents the runtime parameters that will be used to
create and run the container via container APIs. A *security context provider*
is passed to the Kubelet so it can have a chance to mutate Docker API calls in
order to apply the security context.

It is recommended that this design be implemented in two phases:

1.  Implement the security context provider extension point in the Kubelet
so that a default security context can be applied on container run and creation.
2.  Implement a security context structure that is part of a service account. The
default context provider can then be used to apply a security context based on
the service account associated with the pod.

### Security Context Provider

The Kubelet will have an interface that points to a `SecurityContextProvider`.
The `SecurityContextProvider` is invoked before creating and running a given
container:

```go
type SecurityContextProvider interface {
	// ModifyContainerConfig is called before the Docker createContainer call.
	// The security context provider can make changes to the Config with which
	// the container is created.
	// An error is returned if it's not possible to secure the container as 
	// requested with a security context. 
	ModifyContainerConfig(pod *api.Pod, container *api.Container, config *docker.Config)
	
	// ModifyHostConfig is called before the Docker runContainer call.
	// The security context provider can make changes to the HostConfig, affecting
	// security options, whether the container is privileged, volume binds, etc.
	// An error is returned if it's not possible to secure the container as requested 
	// with a security context. 
	ModifyHostConfig(pod *api.Pod, container *api.Container, hostConfig *docker.HostConfig)
}
```

If the value of the SecurityContextProvider field on the Kubelet is nil, the
kubelet will create and run the container as it does today.

### Security Context

A security context resides on the container and represents the runtime
parameters that will be used to create and run the container via container APIs.
Following is an example of an initial implementation:

```go
type Container struct {
	... other fields omitted ...
	// Optional: SecurityContext defines the security options the pod should be run with
    SecurityContext *SecurityContext
}

// SecurityContext holds security configuration that will be applied to a container.  SecurityContext
// contains duplication of some existing fields from the Container resource.  These duplicate fields
// will be populated based on the Container configuration if they are not set.  Defining them on
// both the Container AND the SecurityContext will result in an error.
type SecurityContext struct {
	// Capabilities are the capabilities to add/drop when running the container
	Capabilities *Capabilities

	// Run the container in privileged mode
	Privileged *bool

	// SELinuxOptions are the labels to be applied to the container
	// and volumes
	SELinuxOptions *SELinuxOptions

	// RunAsUser is the UID to run the entrypoint of the container process.
	RunAsUser *int64
}

// SELinuxOptions are the labels to be applied to the container.
type SELinuxOptions struct {
	// SELinux user label
	User string

	// SELinux role label
	Role string

	// SELinux type label
	Type string

	// SELinux level label.
	Level string
}
```

### Admission

It is up to an admission plugin to determine if the security context is
acceptable or not. At the time of writing, the admission control plugin for
security contexts will only allow a context that has defined capabilities or
privileged. Contexts that attempt to define a UID or SELinux options will be
denied by default. In the future the admission plugin will base this decision
upon configurable policies that reside within the [service account](http://pr.k8s.io/2297).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/security_context.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
