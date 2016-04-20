<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
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

## Abstract

A proposal for adding [seccomp](https://github.com/seccomp/libseccomp) support
to Kubernetes.  Seccomp is a system call filtering facility in the Linux
kernel which lets applications define limits on system calls they may make,
and what should happen when system calls are made.  Seccomp is used to reduce
the attack surface available to applications.

## Motivation

Applications use seccomp to restrict the set of system calls they can make.
Recently, container runtimes have begun adding features to allow the runtime
to interract with seccomp on behalf of the application, which eliminates the
need for applications to link against libseccomp directly.  Adding support in
the Kubernetes API for describing seccomp profiles will allow administrators
greater control over the security of workloads running in Kubernetes.

Goals of this design:

1.  Describe how to model seccomp profiles
2.  Describe how to reference those profiles in the containers that use them

## Constraints and Assumptions

This design should:

*  build upon previous security context work
*  be container-runtime agnostic
*  allow definition of custom profiles
*  facilitate containerized applications that link directly to libseccomp

## Use Cases

1.  As an administrator, I want to be able to control access to seccomp profiles:
    1.  To be able to default the seccomp profile when unspecified
    2.  To be able to grant access to a seccomp profile to a class of users
2.  As a user, I want to run an application with a seccomp profile similar to
    the default one provided by my container runtime
3.  As a user, I want to run an application which is already libseccomp-aware
    in a container, and for my application to manage interacting with seccomp
    unmediated by Kubernetes
4.  As a user, I want to be able to describe a custom seccomp profile and use
    it with my containers

### Use Case: Administrator access control

Controlling to control access to seccomp profiles is a cluster administrator concern.
It should be possible for an administrator to:

1.  Define a default seccomp profile for a class of users
2.  Grant access to a seccomp profile to a class of users

Administrators should be able to control two aspects of defaults for seccomp
profiles:

1.  Defaults for classes of users based on pod security policy
2.  Defaults at the API level

#### Classes of users

The [pod security policy](https://github.com/kubernetes/kubernetes/pull/7893)
API extension governs the ability of users to make requests that affect pod
and container security contexts.  The proposed design should deal with
required changes to control access to new API features added.

#### API Defaults

There should be a minimum API-level guarantee for the seccomp profile to use
when the field is not set when a pod is admitted to the API server.  This
profile should be exposed as an argument to the API server.

### Use Case: Seccomp profiles similar to container runtime defaults

Many users will want to use images that make assumptions about running in the
context of their chosen container runtime.  Such images are likely to
frequently assume that they are running in the context of the container
runtime's default seccomp settings.  Therefore, it should be possible to
express a seccomp profile similar to a container runtime's defaults.

### Use Case: Applications that link to libseccomp

Some applications already link to libseccomp and control seccomp directly.  It
should be possible to run these applications unmodified in Kubernetes; this
implies there should be a way to disable seccomp control in Kubernetes for
certain containers, or to run with a "no-op" or "unconfined" profile.

### Use Case: Custom profiles

Different applications have different requirements for seccomp profiles; it
should be possible to express arbitrary seccomp profiles and use them in
containers.

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
* [docker/21105](https://github.com/docker/docker/issues/22109): custom
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

[Hyper](https://hyper.sh) uses a hypervisor to run containers, so seccomp is
not an applicable concept.

### Other platforms and seccomp-like capabilities

FreeBSD has a seccomp/capability-like facility called
[Capsicum](https://www.freebsd.org/cgi/man.cgi?query=capsicum&sektion=4).

#### lxd

[`lxd`](http://www.ubuntu.com/cloud/lxd) constrains containers using a default profile.

Issues:

* [lxd/1084](https://github.com/lxc/lxd/issues/1084): add knobs for seccomp

## Proposed Design

### Seccomp API Resource

We should have an API resource to describe seccomp profiles.  Since docker and
runc both support a rich seccomp API, in order to capture default profiles,
our own API for seccomp should be close to the libseccomp API as well.

Making this API resource non-namespaced has several advantages:

1.  It fits with the design of seccomp profiles as administrator-governed
2.  It makes it easier to implement an access policy

```go

// Represents a seccomp profile.  A profile is composed of a list of seccomp
// rules and a default action.  Seccomp profiles are unnamespaced.
type SeccompProfile struct {
	unversioned.TypeMeta `json:",inline"`
	ObjectMeta           `json:"metadata,omitempty"`

	Rules         []SeccompRule `json:"syscalls,omitempty"`
	DefaultAction SeccompAction `json:"defaultAction"`
}

// A seccomp syscall filter rule.
type SeccompRule struct {
	// The name of the system call to use
	Name   string `json:"name"`
	// The action to take
	Action Action `json:"action"`
	// The arguments to the filter
	Args   []*Arg `json:"args"`
}

// An action associated with a seccomp rule.
type SeccompAction string

const (
	Kill  SeccompAction = "kill"
	Errno SeccompAction = "errno"
	Trap  SeccompAction = "trap"
	Allow SeccompAction = "allow"
	Trace SeccompAction = "trace"
)

// A comparison operator to be used when matching syscall arguments in Seccomp
type SeccompOperator string

const (
	EqualTo              SeccompOperator = "equalTo"
	NotEqualTo           SeccompOperator = "notEqualTo"
	GreaterThan          SeccompOperator = "greaterThan"
	GreaterThanOrEqualTo SeccompOperator = "greaterThanOrEqualTo"
	LessThan             SeccompOperator = "lessThan"
	LessThanOrEqualTo    SeccompOperator = "lassThanOrEqualTo"
	MaskEqualTo          SeccompOperator = "maskEqualTo"
)

// Represents a specific syscall argument in a seccomp rule.
type SeccompArg struct {
	Index    uint            `json:"index"`
	Value    uint64          `json:"value"`
	ValueTwo uint64          `json:"valueTwo"`
	Op       SeccompOperator `json:"op"`
}
```

Along with this API change, there will be supporting code changes:

1.  New REST API implementation
2.  Client code
2.  Validations/conversions
3.  `kubectl` support (including describer)

### Pod API changes

The container `securityContext` must be changed to allow specifying a seccomp profile:

```go
// SecurityContext holds security configuration that will be applied to a container.
// Some fields are present in both SecurityContext and PodSecurityContext.  When both
// are set, the values in SecurityContext take precedence.
type SecurityContext struct {
  // other fields omitted
  // SeccompProfileName is the name of the seccomp profile this pod should run under.
  SeccompProfileName string `json:"seccompProfileName,omitempty"`
}
```

The kubelet docker runtime implementation in `pkg/kubelet/dockertools` will
require code changes to implement setting the seccomp profile when a container
is run.  This API change should not affect the kubelet itself.

The kubectl describer for pods should change to include the new field.

### Pod Security Policy changes

The `PodSecurityPolicy` type should be changed to control seccomp profiles:

```go
// PodSecurityPolicySpec defines the policy enforced.
type PodSecurityPolicySpec struct {
  // other fields omitted
  Seccomp *SeccompStrategyOptions `json:"seccomp,omitempty"`
}

// Defines the seccomp strategy type and any options used to create the strategy.
type SeccompStrategyOptions struct {
	// Rule is the strategy that will dictate the allowable seccomp profile name values that may be set.
	Rule SeccompStrategy `json:"rule"`
	// Profiles are a list of profile names that may be used.
	Profiles []string `json:"profiles,omitempty"`
}

// SeccompStrategy denotes strategy types for generating SeccompProfileName values for a
// SecurityContext.
type SeccompStrategy string

const (
	// container must run as one of a set of profile names.
	SeccompStrategyMustRunAs SeccompStrategy = "MustRunAs"
	// container may make requests for any profile name.
	SeccompStrategyRunAsAny SeccompStrategy = "RunAsAny"
)
```

## Examples

### Use in a pod

Here's an example of a pod that uses a profile called `example-explorer-profile`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: explorer
spec:
  containers:
    - name: explorer
      image: gcr.io/google_containers/explorer:1.0
      args: ["-port=8080"]
      ports:
        - containerPort: 8080
          protocol: TCP
      securityContext:
        seccompProfileName: example-explorer-profile
      volumeMounts:
        - mountPath: "/mount/test-volume"
          name: test-volume
  volumes:
    - name: test-volume
      emptyDir: {}
```

### Unconfined profile

An unconfined profile can be implemented with a single default action:

```yaml
apiVersion: v1
kind: SeccompProfile
metadata:
  name: unconfined
defaultAction: allow
```

### Docker default profile

We can model the docker default profile.  Profiles such as this should by
convention contain the version number of the runtime they are for:

```yaml
apiVersion: v1
kind: SeccompProfile
metadata:
  name: docker-v1-10
defaultAction: kill
rules:
  - name: personality
    action: allow
    args:
    - index: 0
      value: 0
      op: equalTo
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/seccomp.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
