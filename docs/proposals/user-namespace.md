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

[Private user namespaces](http://man7.org/linux/man-pages/man7/user_namespaces.7.html) 
provide the abiltity to isolate user IDs and group IDs.  This allows
a process to have different UID and GID values inside and outside of a 
user namespace.  Ie. running as UID 0 inside the namespace and running
as a non-root UID outside the namespace so that a process has full 
privileges inside its isolated namespace but is otherwise unprivileged.

## Container Runtime Support

### Docker

Docker 1.10 supports [user namespace remapping](https://docs.docker.com/v1.10/engine/reference/commandline/daemon/#daemon-user-namespace-options)
in the docker daemon.  This allows you to run as the root user in a
container while running as a non-root user in the host.

In Docker, you may provide the `--userns-remap` option to the daemon
to enable remapping.  This option will read the `/etc/subuid` and
the `/etc/subgid` files to determine [how to map UIDs and GIDs
in the container](https://docs.docker.com/engine/reference/commandline/dockerd/#detailed-information-on-subuidsubgid-ranges).

Additionally, the `HostConfig` for a container now contains a parameter
that lets you escape the private user namespace and indicate that you
would like to [use the host's user namespace](https://github.com/docker/engine-api/blob/4290f40c056686fcaa5c9caf02eac1dde9315adf/types/container/host_config.go#L310).
By using this setting, administrators may run containers outside of the user
namespace while still requiring other containers to run as a remapped
user.

### rkt

[rkt supports user namespaces](https://coreos.com/rkt/docs/latest/devel/user-namespaces.html)
with a systemd-nspawn based implementation.  Each pod started will
be allocated a random UID range. 

To use private user namespaces in rkt you specify the `--private-users`
flag on the rkt run command.


## Kubernetes Support

As of 1.5 [experimental support](https://github.com/kubernetes/kubernetes/pull/31169)
has been added which will default the user namespace to host when 
a pod is using features that require host namespaces (see defaulting scenarios
described in the design below).  This proposal is 
building on the experimental work.

### Known Issues

These are the known issues (as of writing) that will be hit when
enabling user namespace remapping in a Kubernetes cluster.

1. Negative oomScoreAdj values fail.  In this case the pause container
cannot be started and shows an error `write /proc/self/oom_score_adj: permission denied`
in the daemon log.  Failing in docker 1.10, working in 1.11+
1. Cannot share the host or a container's network namespace when user
namespaces are enabled.  [Fixed in 1.11](https://github.com/docker/docker/pull/21383).
1. Permission denied mounting.  Occurs due to ownership permissions
in the `/var/lib/kubelet/pods` directory when trying to mount the
termination logs and secrets.  A workaround (not a solution) is to
change the remapped GID to the root group.

## Design

### Constraints

* All non-user namespaces have an owning user namespace.  Actions in 
the non-user namespace require [capabilities that are checked against 
the user namespace](http://man7.org/linux/man-pages/man7/user_namespaces.7.html).
This means that when containers share a non-user namespace they must also
share a user namespace whether that be a private user namespace or
the host's user namespace.  
* Currently you may only specify a single user namespace remap value
to the docker daemon.  This means you cannot ensure namespaces are
using different UIDs and GIDs for remapping, they will all be remapped
to the same value(s).
* [Known restrictions when running in a remap environment](https://docs.docker.com/engine/reference/commandline/dockerd/#user-namespace-known-restrictions)
* A cluster may be running with some nodes performing remapping and 
some not.  Any runtime behavior should be isolated to the kubelet
and information about remapping should come from the CRI.

### Detecting and Driving Private User Namespaces Behavior

In order to Do The Right Thing in any solution the system must be able 
to detect that private user namespaces are in use.  

In Docker, this is a daemon setting and is [exposed via 
the info endpoint](https://github.com/docker/docker/commit/ae74092e450f1f2665b90257b65513cc0c19702f).
This information is available in Docker 1.13+ only.

However, in rkt this is a run parameter so it is
unknown if private user namespaces will be used until the container launch
command is created.  

As a consistent and backwards compatible solution to both driving any behavior to escape 
user namespaces (see system defaulting below) and to also control
the use of `--private-users` in the rkt `run` command a feature
gate can be added to the Kubelet as was done in the [experimental
support](https://github.com/kubernetes/kubernetes/pull/31169/files#diff-c5e4440d8576a91980ab36fe63556326).

### Volumes

The system must ensure that the remapped UID and GID have
access to the correct directories under `/var/lib/kubelet/pods` in order
to perform mounting.  Currently the Kubelet root directory is created
with 0750 permissions and are owned by root:root.  Since pods are running
with at least the root group (`runAsUser` may be specified as well as 
supplemental groups) the pod has access to read these directories.  In
a remap environment the group itself may change so the pod will no longer
be able to access these directories.  In order to fix this the root 
directory should be given 0755 access in the Kubelet's `setupDataDirs`.

Beneath the Kubelet root directory each pod is given it's own directory
named by a unique identifier containing:

* containers
* plugins
* volumes

Rather than increasing the access on directories under the root directory,
the pod directory (and sub directories) may be owned by the remapped
GID.  This is very similar to the `fsGroup`, however `fsGroup` is not applied
to items like the termination logs.

How this GID is retrieved should be a container runtime detail.  This
leaves the container runtimes free to manage the permissions of the 
directories under the pod directory (available via injecting the
`RuntimeHelper` into the runtime implementation) in a way that suits
their ability to provide the GID.  For Docker this is the value in
`/etc/subgid`.

TODO 

* are GIDs being remapped in the rkt implementation?  If not then
  root group may be enough.
* more specific information on mount types
* can the remap GID be injected as an `fsGroup` for mount types that 
support `chmod` (if `fsGroup` is not already set)
* host mounts do not - and should not - chmod for remap 
environments.  This is up to the administrator to manage.

### Escaping Private User Namespaces

Some containers may use features which prohibit the use of private
user namespaces.  If the Kubelet is aware that private user namespaces
are in use it has the ability to modify the container's runtime 
configuration when it detects a feature that is not compatible with
private user namespaces.

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

This logic was implemented in the [experimental
support](https://github.com/kubernetes/kubernetes/pull/31169) as of
Kubernetes 1.5.  

Open Question: should we allow the usage of host user namespace as a 
flag in addition to defaulting?  Ie. exposing `HostUserNamespace` as 
a flag on the `SecurityContext` like we do with net, ipc, and pid?

### Enforcing Private User Namespaces

In some cases a cluster administrator may wish to ensure that a container
is never defaulted to use the host user namespace.  This falls under
the responsibility of cluster policy with `PodSecurityPolicy`.  

`PodSecurityPolicy` has the ability to control using the host network,
PID, and IPC namespaces, which capabilities can be added, as well as
which types of volumes can be used.  This effectively controls who
will be allowed to trigger the Kubelet defaulting logic that would 
escape private user namespaces.  

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/user-namespace.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
