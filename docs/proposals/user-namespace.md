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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# User Namespaces

## Motivation

Running a container as root is a common practice for many
image authors. However, many clusters operate under a more strict
security profile that prevents usage of the root UID. This means that
either admins must allow exceptions for root access to run many of
the images available or image authors must be aware of the deployment
environment in order to develop a container that works well under
the increased security constraints. This frustrates end-users that
want to run the same images they run on their host as they cannot
run those images on the cluster.

Docker 1.10 supports [user namespace remapping](https://docs.docker.com/v1.10/engine/reference/commandline/daemon/#daemon-user-namespace-options)
in the docker daemon.  This allows you to run as the root user in a
container while running as a non-root user in the host.

Additionally, the `HostConfig` for a container now contains a parameter
that lets you escape the private user namespace and indicate that you
would like to [use the host's user namespace](https://github.com/docker/engine-api/blob/master/types/container/host_config.go#L310).
By using this setting, administrators may run containers outside of the user
namespace while still requiring other containers to run as a remapped
user.

## Current State

In Docker, you may provide the `--userns-remap` option to the daemon
to enable remapping.  This option will read the `/etc/subuid` and
the `/etc/subgid` files to determine [how to map UIDs and GIDs
in the container](https://docs.docker.com/engine/reference/commandline/dockerd/#detailed-information-on-subuidsubgid-ranges).

Currently there are no configuration options available to make
Kubernetes aware of a container runtime running a remapped user
environment.

If the Docker daemon is launched with the remap option in a Kubernetes
cluster today and a pod author would like to run a container that uses
the host user namespace there is not currently a way to express the
configuration.

### Known Issues

These are the known issues (as of writing) that will be hit when
enabling user namespace remapping in a Kubernetes cluster.

1. Negative oomScoreAdj values fail.  In this case the pause container
cannot be started and shows an error `write /proc/self/oom_score_adj: permission denied`
in the daemon log.  Failing in docker 1.10, working in 1.12
1. Cannot share the host or a container's network namespace when user
namespaces are enabled.  [Fixed in 1.11](https://github.com/docker/docker/pull/21383).
1. Permission denied mounting.  Occurs due to ownership permissions
in the `/var/lib/kubelet/pods` directory when trying to mount the
termination logs and secrets.  A workaround (not a solution) is to
change the remapped GID to the root group.

## Design

### Constraints

* If containers share any namespace they must share the user namespace.
This is because [capabilities are checked against the user namespace](http://man7.org/linux/man-pages/man7/user_namespaces.7.html).
This means that if any namespace is set to `host` the pod must
also use the host user namespace.  Since containers in Kubernetes
always share a network namespace they must also share the user
namespace.
* Currently you may only specify a single user namespace remap value
to the docker daemon.  This means you cannot ensure namespaces are
using different UIDs and GIDs for remapping, they will all be remapped
to the same value(s).
* [Known restrictions when running in a remap environment](https://docs.docker.com/engine/reference/commandline/dockerd/#user-namespace-known-restrictions)

### Open Questions

1.  Do we need to expose a `HostUserNamespace` option on the `SecurityContext`
or can we provide defaulting mechanisms that do the right thing?
1.  If we are not defaulting and we are aware of the remap environment
should we validate in the API server for misconfiguration or allow
this to fail in the container runtime?
1.  Is there a valid use case for running some nodes with remapping
turned on and some without?

### Option 1: System Defaulting

In this option, there is no `HostUserNamespace` flag exposed in the API.
This requires that the Kubelet be aware of the remapped environment and
provide defaulting for the host user namespace settings of a container
runtime when launching the containers.  This work has been
[prototyped](https://github.com/kubernetes/kubernetes/pull/31169).

It is debatable whether or not the kubelet should provide defaulting based on
this knowledge since it may mean that a container is run with a
configuration that is not what the pod author expected.  However, if
policy can check that the defaulted configuration is valid for the
user it may be valuable and avoid frustrations with runtime errors.

Pros:

1. reduces the need for Linux specific knowledge
1. system does the right thing

Cons:

1. requires extra flags to make the system aware of the remapped
environment (is this required anyway due to the mount issue?)
1. the user may unexpectedly run a container with the host user namespace.
1. Cluster administrators must be aware that allowing certain capabilities will automatically escape user namespaces.

### Option 2: API Support

Fully exposing `HostUserNamespace` as a flag on the `SecurityContext` of
a pod requires the following changes for a minimally viable feature:

1. updates to the API objects
1. support in the Kubelet to set the correct flags based on the spec

In this option validation may occur in the API server if it is aware
of the remapped environment.  This would allow rejection during the
submission of the pod spec or, optionally, allow for the defaulting
logic to kick in and not reject the pod.  This could be considered
option 3.

The pattern for exposing namespaces in the `SecurityContext` of a pod
is already established.

Pros:

1. allows full flexibility
1. still allows for defaulting and validation to occur (which incurs the
cons mentioned in the defaulting option above)

Cons:

1. requires pod authors to have linux and cluster environment specific
knowledge to know that the pod spec must contain a flag (unless defaulting
is also implemented)

### Option 3: Combination/staged approach

As a way to begin implementation it is possible that both defaulting
and full api support are eventually implemented.  It is useful that
the system do the right thing or at least provide useful messaging
to the user when misconfiguration is detected.

In this option the implementation would begin with defaulting and, based
on real world feedback, move to full API support if necessary.  Once
API support is included a decision could be made to keep defaulting
or start rejecting misconfigured pods in a useful way.

### Kubelet

In all cases, the system must ensure that the remapped UID and GID have
access to the correct directories under `/var/lib/kubelet/pods` in order
to perform mounting.

This may be accomplished by ensuring group access to the correct
directories.  It implies that the Kubelet will be given (or read)
the value of `/etc/subgid` to perform a `chmod`.

### Defaulting Scenarios

If the system is to default host user namespace options it would occur
when:

1.  setting any other namespace flag to host.  Not strictly
required for net=host if not binding to low ports but if the system
provides defaulting it covers the 80% case
1.  using capabilities that aren't tied to any namespace
    * MKNOD
    * SYS_MODULE
    * SYS_TIME
1.  explicitly granted permissions to use a host volume and escape the private user namespace

### Security

In both a defaulting or API based solution administrators may wish to
restrict who is able to use the host user namespace.  To do this the
`PodSecurityPolicy` should be updated to enable controlling the
usage of the host user namespace.  Since `PodSecurityPolicy` is
validated during admission the system must ensure that any defaulting
is still valid against granted policy.

### Use with rkt

[rkt supports user namespaces](https://coreos.com/rkt/docs/latest/devel/user-namespaces.html)
with a systemd-nspawn based implementation.  Since rkt and docker both
have different configuration options for the usage of host namespaces
when user namespaces are enabled it is important that anything the
system does for defaulting is conveyed as a recommendation that is
then implemented by the container runtime shim.

In rkt it would mean running with `--private-users=false`.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/user-namespace.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
