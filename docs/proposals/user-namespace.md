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

# Support Host User Namespaces

## Motivation

Docker 1.10 supports [user namespace remapping](https://docs.docker.com/v1.10/engine/reference/commandline/daemon/#daemon-user-namespace-options)
in the docker daemon.  This allows you to run as the root user in a
container while running as a non-root user in the host.

Additionally, the `HostConfig` for a container now contains a parameter
that lets you escape the private user namespace and indicate that you
would like to [use the host's user namespace](https://github.com/docker/engine-api/blob/master/types/container/host_config.go#L310).

If the Docker daemon is launched with the remap option in a Kubernetes
cluster today and a pod author would like to run a container that uses
the host user namespace there is not currently a way to express the
configuration.

## Assumptions

1.  There is currently a bug in Docker that does not allow you to use
`--net=host`, `--pid=host`, or `--ipc=host` when remapping is enabled.
This is fixed in https://github.com/docker/docker/pull/25771 such that
if you are enabling user namespace remapping you may use `--net=host`, `--pid=host`,
and `--ipc=host` flags if `--userns=host`.

## When is host user namespace required

Kubernetes cluster administrators wish to have a higher level of
default cluster security on Docker 1.10+ by enabling user namespace
remapping, but want to be able to continue to run cluster level actions
for monitoring.  Assuming that the docker daemon is started with
`--userns-remap=default`,
here is when the host user namespace should be set:

1.  When setting any other namespace flag to `host`
    * Not strictly required for `net=host` if not binding to low ports 
     but if the system provides defaulting it covers the 80% case
2.  When using capabilities that aren't tied to any namespace
    * MKNOD
    * SYS_MODULE
    * SYS_TIME
    * others?  (@mrunalp is looking at source to determine this)
3.  When explicitly granted permissions to use a host volume and 
escape the private user namespace


### Concrete Examples

| Use Case  | Support |
|---|---|
| Load a kernel module  | --userns=host --cap-add SYS_ADMIN  |
| Create a device using mknod  |  --userns=host --cap-add MKNOD |
| Access host path as root user  | --userns=host --cap-add SYS_ADMIN  |
| Bind to a low port on a host interface  | --userns=host, --net=host*  |
| Monitor other processes on host looking at /proc  | --userns=host, --pid=host*  |
| Bind to a high port on a host interface | --net=host, --userns=container_private^  |
| Share net,pid namespaces between containers  | --net,--pid=container:id implicitly shares userns as well  |

We can make a couple of generalizations from the table:

1. Certain capabilities that aren't tied to any namespace such as
CAP_MKNOD, CAP_SYS_MODULE or CAP_SYS_TIME are requested, then they
imply userns=host.
2. Setting any other namespace to host could imply userns=host.

[*] = requires docker/docker#25771
[^] = not supported in docker today but could be done using runc



## Options

1.  Provide a `HostUser` flag and require manual setting.

    Pros
    *  Quick and easy
    *  Allows full flexibility

    Cons
    *  Requires linux specific knowledge

2.  Provide a `HostUser` flag and provide auto setting.

    Pros
    *  System does the right thing when it can

    Cons
    *  Requires extra flags/config - this defaulting should only occur
    when remapping is configured on the Docker daemon.
    *  Does not support edge cases like using user namespaces but binding
    to high ports on the host.
    *  The user may unexpectedly run a container with the host user namespace.

## Proposal

For 1.4 we can add the `HostUser` flag to provide immediate support
for anyone who has a use case that is not currently supported and
enable users on Docker 1.10+ to test and enable `userns=remap`.  We can
then iterate on smart defaulting in future releases to ensure that the
system will do the right thing when remapping is known to be configured.


## Additional Concerns

When any new field is added to the `SecurityContext` it should also be
added to the `PodSecurityPolicy` so that it may be controlled by an
administrator.

In this case the field is a boolean and should be controlled by an option
that allows it to be set or not.  PSP should not provide a default.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/user-namespace.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
