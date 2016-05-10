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

## Motivation
With kernel support of namespaced kernel parameters, it is possible to set these settings differently on each container.
In docker version 1.11.1 it was possible to change kernel parameters only manualy from privileged containers
and changes would be applied to every container.
It was not possible to set the parameters within a non-privileged container.
Since docker verison 1.12.0 it is possible to set namespaced parameters during container creation.
These modified parameters would be related only to container.
In kubernetes we want to allow to set these parameters within a pod.

Goal:
- possibility to set customizable sysctl records for namespaces within a pod

## Constraints

* Only namespaced sysctl parameters can be changed.
* Should build on top of the existing security context work

## Use Cases

A user wants to set customizable sysctl records for namespaces within a container.

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

Container specification must be changed to allow specifying sysctl parameters:

```go
// SecurityContext holds security configuration that will be applied to a container.
// Some fields are present in both SecurityContext and PodSecurityContext.  When both
// are set, the values in SecurityContext take precedence.
type SecurityContext struct {
...
	// List of Namespaced sysctls used for the container
	SysctlParameters []SysctlParameter `json:"sysctlOptions,omitempty"`
...
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
    sysctlParameters:
        - variable: net.ipv4.ip_forward: 
          value: 2
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/seccomp.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
