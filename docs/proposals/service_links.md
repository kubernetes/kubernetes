# Explicit service links

## Abstract

Proposes an API for explicit links to services in the pod spec.

## Motivation

Currently, all pods are automatically injected with docker-link-style
environment variables for each service in their namespaces.  Making service
dependencies explicit in the PodSpec has a number of advantages:

1.  Reduce accidental coupling
2.  Reduce the number of IP tables rules that have to be created by the
    kube-proxy
3.  Identify relationships between pods and services, allowing user interfaces
    to display additional information for end users.

## Constraints and Assumptions

1.  The old behavior must remain the default (for now)
2.  It must be possible to migrate off the default to using explicit links

## Use Cases

1.  As a user, I want to be able to explicitly define which pods are injected
    with information about a service so that I can ensure my pods receive
    information only about services they actually need to use
2.  As a user, I want to control how services are injected into pods and
    containers

### Explicit links between pods and services

Currently in Kubernetes a pod is injected with docker- link style environment
variables and simpler `{SERVICE_NAME}_SERVICE_HOST}` and
`{SERVICE_NAME}_SERVICE_PORT`
[environment variables](http://kubernetes.io/docs/user-guide/services/#environment-variables)
for every service in its namespace.  This is disadvantageous for a number of
reasons:

1.  It is not possible to tell from the pod's spec which services it consumes
2.  The proxy must generate rules on a node for every service in every namespace
3.  There is no way to prevent accidental coupling

Adding explicit links to services addresses all three of the above concerns,
once the existing automatic injection behavior is turned off:

1.  Explicit links will appear in the pod's spec
2.  The proxy can be modified to generate rules only for the services which are
    explicitly consumed on a host

### Controlling how pods are injected with information

It is common for legacy applications to expect custom environment variable names
that do not fit the docker-link style that is generated today. Custom
environment variables for service links are possible, but they are cumbersome
and lead to duplication in the environment.  To get custom environment vars for
service links, a pod must define the custom variables and use variable expansion
to inject the value of the default variable.

Explicit links offer an opportunity to ease the experience and make custom
variables easier to reason about.  Instead of referencing variables that will be
present at runtime, but are not present otherwise in the pod spec, a pod author
could simply specify a format or prefix for the variables to be created.

## Analysis

Determining the right API is a complex topic that warrants some analysis.

### What degree of specificity is desired?

The desired degree of specificity for an explicit service link is important to
consider.  There is a continuum of options of different complexities.  On the
simple end of the spectrum is an API that manifests environment variables in the
exact same way as the current implicit mechanism, but allows a service to be
explicitly referenced.  On the complex end of the spectrum is an API that allows
individual aspects of a service to be projected into individual environment
variables.

The right API is probably not either of those options, but it is useful to
consider the pros and cons of both as a way to discover the right API.

#### Pro/con: Simple API

Pros:

1.  Very small practical and conceptual delta from current behavior
2.  Users can employ the same strategies for custom env vars
3.  Easiest to explain
4.  Easiest to support

Cons:

1.  Inflexible
2.  Users have the same usability issues when custom env var names are required
3.  Does not extend well to services in other namespaces

#### Pro/con: Most complex API

Pros:

1.  Offers highest degree of flexibility
2.  Custom env var names are cheap at fine-grained level of control

Cons:

1.  Most difficult to explain and reason about
2.  Most difficult to support
3.  Higher implementation cost
4.  Large conceptual delta
5.  Likely to add significant complexity to pod spec
6.  Likely to be very verbose

#### Lessons Learned from Prior APIs

One fundamental lesson of Kubernetes API design is that we must make backward
compatible API changes except in cases of special exception.  This means that we
need to consider backward compatibility when making new API changes.  As a
requirement, backward compatibility imposes a burden that you have to live with
the APIs you release.  This means that we should attempt to be conservative with
new API features and avoid unnecessary complexity.  Therefore, a simpler API is
advantageous because complexity can be added later **if desired and warranted**.

#### Middleground

A conservative middleground might be an API that has behavior extremely close to
the current implicit behavior but allows specifying custom variable names.  This
approach has the advantage of being close to what users already expect, and can
be implemented with a fairly simple API.

### Non-existent Services

**TODO:** we need to consider what the behavior is when a service that does not
exist is referenced via this API.

## Proposed Changes

### API changes

Pull request [#37295](https://github.com/kubernetes/kubernetes/pull/37295)
proposes an `EnvFrom` addition to the PodSpec to handle creating multiple
environment variables from a single subject.  This API surface is ideal for the
service use-cases, so we will extend it here:

```go
type Container struct {
	// other fields omitted

	EnvFrom []EnvFromSource `json:"envFrom,omitempty"`
}

type EnvVarFromSource struct {
	// other fields omitted

	// Prefix is an optional prefix for generated environment variables
	Prefix string `json:"prefix,omitempty"`

	// Can be refactored to ObjectReference at a later time while
	// maintaining backward compatibility.
	Service *LocalObjectReference `json:"service,omitempty"`
}
```

### Kubelet Changes

The Kubelet must be modified to support creation of environment variables via
the new API.  Additionally, the Kubelet should have a flag added which disables
the current implicit behavior to allow users to experiment with explicit-only
service links.

## Examples

### No custom variable names

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: simple-service-reference-example
spec:
  containers:
  - image: busybox
    name: example-container
    envFrom:
    - service:
        name: redis-master
```

Yields the environment variables:

```
REDIS_MASTER_SERVICE_HOST=10.0.0.11
REDIS_MASTER_SERVICE_PORT=6379
REDIS_MASTER_PORT=tcp://10.0.0.11:6379
REDIS_MASTER_PORT_6379_TCP=tcp://10.0.0.11:6379
REDIS_MASTER_PORT_6379_TCP_PROTO=tcp
REDIS_MASTER_PORT_6379_TCP_PORT=6379
REDIS_MASTER_PORT_6379_TCP_ADDR=10.0.0.11
```

### Custom variable names

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: simple-service-reference-example
spec:
  containers:
  - image: busybox
    name: example-container
    envFrom:
    - prefix: "CUSTOM_NAME"
      service:
        name: redis-master
```

Yields the environment variables:

```
CUSTOM_NAME_SERVICE_HOST=10.0.0.11
CUSTOM_NAME_SERVICE_PORT=6379
CUSTOM_NAME_PORT=tcp://10.0.0.11:6379
CUSTOM_NAME_PORT_6379_TCP=tcp://10.0.0.11:6379
CUSTOM_NAME_PORT_6379_TCP_PROTO=tcp
CUSTOM_NAME_PORT_6379_TCP_PORT=6379
CUSTOM_NAME_PORT_6379_TCP_ADDR=10.0.0.11
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/service_links.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
