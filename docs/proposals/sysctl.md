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

This proposal aims at extending the current pod specification with a support
for kernel parameters setting for containers.

In Linux, the sysctl interface allows an administrator to modify kernel parameters at runtime.
Parameters are available via ``/proc/sys/`` virtual process file system.
The parameters covers various subsystems such as:

* kernel (common prefix: ``kernel.``)
* networking (common prefixes: ``net.``)
* virtual memory (common prefix: ``vm.``)
* MDADM (common prefix: ``dev.``)

More subsystems are described in [Kernel docs](https://www.kernel.org/doc/Documentation/sysctl/README).

To get a list of basic prefixes on your system, you can run

```
$ sudo sysctl -a | cut -d' ' -f1 | cut -d'.' -f1 | sort -u
```

To get a list of all parameters, you can run

```
$ sudo sysctl -a
```


## Motivation

Since kernel support for namespaced kernel parameters,
it is possible to set these parameters for each container separately.

In docker version 1.11.1 it is possible to change kernel parameters inside privileged containers.
However, the process is purely manual and the changes are applied across all containers affecting entire host system.
It is not possible to set the parameters within a non-privileged container.

Since docker verison 1.12.0 it is possible to set namespaced parameters during a container creation process.
Parameters set are namespace scoped and applies to a single container only.

In kubernetes we want to allow to set these parameters within a pod specification.

## Constraints and Assumptions

* Only namespaced kernel parameters can be modified
* Built on a top of the existing security context work
* Be container-runtime agnostic
* Kernel parameters can be set during a container creation process only

## Further work

* Update kernel parameters in running container(s)

## Use Cases

1. As an administrator I want to set customizable kernel parameters for a container
    1. To be able to limit consumed kernel resources

### Use Case: Set namespaced kernel parameters

As an administrator I would like to limit available kernel resources for a container
so I can provide more resources to other containers or to restrict system communication.

## Community Work

### Runc support for sysctl

Applied changes:

* https://github.com/opencontainers/runc/pull/73
* https://github.com/opencontainers/runc/pull/303

### Docker support for sysctl

Applied changes:

* https://github.com/docker/docker#19265
* https://github.com/docker/engine-api#38

Issues:

* https://github.com/docker/docker#21126
* https://github.com/ibm-messaging/mq-docker#13

## Proposed Design
	
### Sysctl API Resource

We should have an API resource to describe sysctl parameters.

```go

// SysctlParameter defines a kernel parameter to be set
type SysctlParameter struct {
	// Name of a property to set
	Variable string `json:"variable"`
	// Value of a property to set
	Value string `json:"value"`
}
```
### Pod API changes

Container specification must be changed to allow specifycation of kernel parameters:

```go
// SecurityContext holds security configuration that will be applied to a container.
// Some fields are present in both SecurityContext and PodSecurityContext.  When both
// are set, the values in SecurityContext take precedence.
type SecurityContext struct {
	// List of Namespaced sysctls used for the container
	SysctlParameters []SysctlParameter `json:"sysctlParameters,omitempty"`
}
```
## Examples

### Use in a pod

Here's an example of a pod that has ``net.ipv4.ip_forward`` set to ``2``:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  labels:
    name: nginx
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
    securityContext:
      sysctlParameters:
        - variable: net.ipv4.ip_forward: 
          value: 2
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/seccomp.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
