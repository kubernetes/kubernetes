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
[here](http://releases.k8s.io/release-1.3/docs/design/seccomp.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

A proposal for adding **alpha** support for
[seccomp](https://github.com/seccomp/libseccomp) to Kubernetes.  Seccomp is a
system call filtering facility in the Linux kernel which lets applications
define limits on system calls they may make, and what should happen when
system calls are made.  Seccomp is used to reduce the attack surface available
to applications.

## Motivation

Applications use seccomp to restrict the set of system calls they can make.
Recently, container runtimes have begun adding features to allow the runtime
to interact with seccomp on behalf of the application, which eliminates the
need for applications to link against libseccomp directly.  Adding support in
the Kubernetes API for describing seccomp profiles will allow administrators
greater control over the security of workloads running in Kubernetes.

Goals of this design:

1.  Describe how to reference seccomp profiles in containers that use them

## Constraints and Assumptions

This design should:

*  build upon previous security context work
*  be container-runtime agnostic
*  allow use of custom profiles
*  facilitate containerized applications that link directly to libseccomp

## Use Cases

1.  As an administrator, I want to be able to grant access to a seccomp profile
    to a class of users
2.  As a user, I want to run an application with a seccomp profile similar to
    the default one provided by my container runtime
3.  As a user, I want to run an application which is already libseccomp-aware
    in a container, and for my application to manage interacting with seccomp
    unmediated by Kubernetes
4.  As a user, I want to be able to use a custom seccomp profile and use
    it with my containers

### Use Case: Administrator access control

Controlling access to seccomp profiles is a cluster administrator
concern. It should be possible for an administrator to control which users
have access to which profiles.

The [pod security policy](https://github.com/kubernetes/kubernetes/pull/7893)
API extension governs the ability of users to make requests that affect pod
and container security contexts.  The proposed design should deal with
required changes to control access to new functionality.

### Use Case: Seccomp profiles similar to container runtime defaults

Many users will want to use images that make assumptions about running in the
context of their chosen container runtime.  Such images are likely to
frequently assume that they are running in the context of the container
runtime's default seccomp settings.  Therefore, it should be possible to
express a seccomp profile similar to a container runtime's defaults.

As an example, all dockerhub 'official' images are compatible with the Docker
default seccomp profile.  So, any user who wanted to run one of these images
with seccomp would want the default profile to be accessible.

### Use Case: Applications that link to libseccomp

Some applications already link to libseccomp and control seccomp directly.  It
should be possible to run these applications unmodified in Kubernetes; this
implies there should be a way to disable seccomp control in Kubernetes for
certain containers, or to run with a "no-op" or "unconfined" profile.

Sometimes, applications that link to seccomp can use the default profile for a
container runtime, and restrict further on top of that.  It is important to
note here that in this case, applications can only place _further_
restrictions on themselves.  It is not possible to re-grant the ability of a
process to make a system call once it has been removed with seccomp.

As an example, elasticsearch manages its own seccomp filters in its code.
Currently, elasticsearch is capable of running in the context of the default
Docker profile, but if in the future, elasticsearch needed to be able to call
`ioperm` or `iopr` (both of which are disallowed in the default profile), it
should be possible to run elasticsearch by delegating the seccomp controls to
the pod.

### Use Case: Custom profiles

Different applications have different requirements for seccomp profiles; it
should be possible to specify an arbitrary seccomp profile and use it in a
container.  This is more of a concern for applications which need a higher
level of privilege than what is granted by the default profile for a cluster,
since applications that want to restrict privileges further can always make
additional calls in their own code.

An example of an application that requires the use of a syscall disallowed in
the Docker default profile is Chrome, which needs `clone` to create a new user
namespace.  Another example would be a program which uses `ptrace` to
implement a sandbox for user-provided code, such as
[eval.in](https://eval.in/).

## Community Work

### Container runtime support for seccomp

#### Docker / opencontainers

Docker supports the open container initiative's API for
seccomp, which is very close to the libseccomp API.  It allows full
specification of seccomp filters, with arguments, operators, and actions.

Docker allows the specification of a single seccomp filter.  There are
community requests for:

Issues:

* [docker/22109](https://github.com/docker/docker/issues/22109): composable
  seccomp filters
* [docker/21105](https://github.com/docker/docker/issues/22105): custom
  seccomp filters for builds

#### rkt / appcontainers

The `rkt` runtime delegates to systemd for seccomp support; there is an open
issue to add support once `appc` supports it.  The `appc` project has an open
issue to be able to describe seccomp as an isolator in an appc pod.

The systemd seccomp facility is based on a whitelist of system calls that can
be made, rather than a full filter specification.

Issues:

* [appc/529](https://github.com/appc/spec/issues/529)
* [rkt/1614](https://github.com/coreos/rkt/issues/1614)

#### HyperContainer

[HyperContainer](https://hypercontainer.io) does not support seccomp.

### Other platforms and seccomp-like capabilities

FreeBSD has a seccomp/capability-like facility called
[Capsicum](https://www.freebsd.org/cgi/man.cgi?query=capsicum&sektion=4).

#### lxd

[`lxd`](http://www.ubuntu.com/cloud/lxd) constrains containers using a default profile.

Issues:

* [lxd/1084](https://github.com/lxc/lxd/issues/1084): add knobs for seccomp

## Proposed Design

### Seccomp API Resource?

An earlier draft of this proposal described a new global API resource that
could be used to describe seccomp profiles.  After some discussion, it was
determined that without a feedback signal from users indicating a need to
describe new profiles in the Kubernetes API, it is not possible to know
whether a new API resource is warranted.

That being the case, we will not propose a new API resource at this time.  If
there is strong community desire for such a resource, we may consider it in
the future.

Instead of implementing a new API resource, we propose that pods be able to
reference seccomp profiles by name.  Since this is an alpha feature, we will
use annotations instead of extending the API with new fields.

### API changes?

In the alpha version of this feature we will use annotations to store the
names of seccomp profiles.  The keys will be:

`container.seccomp.security.alpha.kubernetes.io/<container name>`

which will be used to set the seccomp profile of a container, and:

`seccomp.security.alpha.kubernetes.io/pod`

which will set the seccomp profile for the containers of an entire pod.  If a
pod-level annotation is present, and a container-level annotation present for
a container, then the container-level profile takes precedence.

The value of these keys should be container-runtime agnostic. We will
establish a format that expresses the conventions for distinguishing between
an unconfined profile, the container runtime's default, or a custom profile.
Since format of profile is likely to be runtime dependent, we will consider
profiles to be opaque to kubernetes for now.

The following format is scoped as follows:

1.  `runtime/default` - the default profile for the container runtime
2.  `unconfined` - unconfined profile, ie, no seccomp sandboxing
3.  `localhost/<profile-name>` - the profile installed to the node's local seccomp profile root

Since seccomp profile schemes may vary between container runtimes, we will
treat the contents of profiles as opaque for now and avoid attempting to find
a common way to describe them.  It is up to the container runtime to be
sensitive to the annotations proposed here and to interpret instructions about
local profiles.

A new area on disk (which we will call the seccomp profile root) must be
established to hold seccomp profiles.  A field will be added to the Kubelet
for the seccomp profile root and a knob (`--seccomp-profile-root`) exposed to
allow admins to set it. If unset, it should default to the `seccomp`
subdirectory of the kubelet root directory.

### Pod Security Policy annotation

The `PodSecurityPolicy` type should be annotated with the allowed seccomp
profiles using the key
`seccomp.security.alpha.kubernetes.io/allowedProfileNames`.  The value of this
key should be a comma delimited list.

## Examples

### Unconfined profile

Here's an example of a pod that uses the unconfined profile:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: trustworthy-pod
  annotations:
    seccomp.security.alpha.kubernetes.io/pod: unconfined
spec:
  containers:
    - name: trustworthy-container
      image: sotrustworthy:latest
```

### Custom profile

Here's an example of a pod that uses a profile called `example-explorer-
profile` using the container-level annotation:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: explorer
  annotations:
    container.seccomp.security.alpha.kubernetes.io/explorer: localhost/example-explorer-profile
spec:
  containers:
    - name: explorer
      image: gcr.io/google_containers/explorer:1.0
      args: ["-port=8080"]
      ports:
        - containerPort: 8080
          protocol: TCP
      volumeMounts:
        - mountPath: "/mount/test-volume"
          name: test-volume
  volumes:
    - name: test-volume
      emptyDir: {}
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/seccomp.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
