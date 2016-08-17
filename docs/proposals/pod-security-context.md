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
[here](http://releases.k8s.io/release-1.3/docs/proposals/pod-security-context.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

A proposal for refactoring `SecurityContext` to have pod-level and container-level attributes in
order to correctly model pod- and container-level security concerns.

## Motivation

Currently, containers have a `SecurityContext` attribute which contains information about the
security settings the container uses.  In practice, many of these attributes are uniform across all
containers in a pod.  Simultaneously, there is also a need to apply the security context pattern
at the pod level to correctly model security attributes that apply only at a pod level.

Users should be able to:

1.  Express security settings that are applicable to the entire pod
2.  Express base security settings that apply to all containers
3.  Override only the settings that need to be differentiated from the base in individual
    containers

This proposal is a dependency for other changes related to security context:

1.  [Volume ownership management in the Kubelet](https://github.com/kubernetes/kubernetes/pull/12944)
2.  [Generic SELinux label management in the Kubelet](https://github.com/kubernetes/kubernetes/pull/14192)

Goals of this design:

1.  Describe the use cases for which a pod-level security context is necessary
2.  Thoroughly describe the API backward compatibility issues that arise from the introduction of
    a pod-level security context
3.  Describe all implementation changes necessary for the feature

## Constraints and assumptions

1.  We will not design for intra-pod security; we are not currently concerned about isolating
    containers in the same pod from one another
1.  We will design for backward compatibility with the current V1 API

## Use Cases

1.  As a developer, I want to correctly model security attributes which belong to an entire pod
2.  As a user, I want to be able to specify container attributes that apply to all containers
    without repeating myself
3.  As an existing user, I want to be able to use the existing container-level security API

### Use Case: Pod level security attributes

Some security attributes make sense only to model at the pod level.  For example, it is a
fundamental property of pods that all containers in a pod share the same network namespace.
Therefore, using the host namespace makes sense to model at the pod level only, and indeed, today
it is part of the `PodSpec`.  Other host namespace support is currently being added and these will
also be pod-level settings; it makes sense to model them as a pod-level collection of security
attributes.

## Use Case: Override pod security context for container

Some use cases require the containers in a pod to run with different security settings.  As an
example, a user may want to have a pod with two containers, one of which runs as root with the
privileged setting, and one that runs as a non-root UID.  To support use cases like this, it should
be possible to override appropriate (ie, not intrinsically pod-level) security settings for
individual containers.

## Proposed Design

### SecurityContext

For posterity and ease of reading, note the current state of `SecurityContext`:

```go
package api

type Container struct {
    // Other fields omitted

    // Optional: SecurityContext defines the security options the pod should be run with
    SecurityContext *SecurityContext `json:"securityContext,omitempty"`
}

type SecurityContext struct {
    // Capabilities are the capabilities to add/drop when running the container
    Capabilities *Capabilities `json:"capabilities,omitempty"`

    // Run the container in privileged mode
    Privileged *bool `json:"privileged,omitempty"`

    // SELinuxOptions are the labels to be applied to the container
    // and volumes
    SELinuxOptions *SELinuxOptions `json:"seLinuxOptions,omitempty"`

    // RunAsUser is the UID to run the entrypoint of the container process.
    RunAsUser *int64 `json:"runAsUser,omitempty"`

    // RunAsNonRoot indicates that the container should be run as a non-root user.  If the RunAsUser
    // field is not explicitly set then the kubelet may check the image for a specified user or
    // perform defaulting to specify a user.
    RunAsNonRoot bool `json:"runAsNonRoot,omitempty"`
}

// SELinuxOptions contains the fields that make up the SELinux context of a container.
type SELinuxOptions struct {
    // SELinux user label
    User string `json:"user,omitempty"`

    // SELinux role label
    Role string `json:"role,omitempty"`

    // SELinux type label
    Type string `json:"type,omitempty"`

    // SELinux level label.
    Level string `json:"level,omitempty"`
}
```

### PodSecurityContext

`PodSecurityContext` specifies two types of security attributes:

1.  Attributes that apply to the pod itself
2.  Attributes that apply to the containers of the pod

In the internal API, fields of the `PodSpec` controlling the use of the host PID, IPC, and network
namespaces are relocated to this type:

```go
package api

type PodSpec struct {
    // Other fields omitted

    // Optional: SecurityContext specifies pod-level attributes and container security attributes
    // that apply to all containers.
    SecurityContext *PodSecurityContext `json:"securityContext,omitempty"`
}

// PodSecurityContext specifies security attributes of the pod and container attributes that apply
// to all containers of the pod.
type PodSecurityContext struct {
    // Use the host's network namespace. If this option is set, the ports that will be
    // used must be specified.
    // Optional: Default to false.
    HostNetwork bool
    // Use the host's IPC namespace
    HostIPC bool

    // Use the host's PID namespace
    HostPID bool

    // Capabilities are the capabilities to add/drop when running containers
    Capabilities *Capabilities `json:"capabilities,omitempty"`

    // Run the container in privileged mode
    Privileged *bool `json:"privileged,omitempty"`

    // SELinuxOptions are the labels to be applied to the container
    // and volumes
    SELinuxOptions *SELinuxOptions `json:"seLinuxOptions,omitempty"`

    // RunAsUser is the UID to run the entrypoint of the container process.
    RunAsUser *int64 `json:"runAsUser,omitempty"`

    // RunAsNonRoot indicates that the container should be run as a non-root user.  If the RunAsUser
    // field is not explicitly set then the kubelet may check the image for a specified user or
    // perform defaulting to specify a user.
    RunAsNonRoot bool
}

// Comments and generated docs will change for the container.SecurityContext field to indicate
// the precedence of these fields over the pod-level ones.

type Container struct {
    // Other fields omitted

    // Optional: SecurityContext defines the security options the pod should be run with.
    // Settings specified in this field take precedence over the settings defined in
    // pod.Spec.SecurityContext.
    SecurityContext *SecurityContext `json:"securityContext,omitempty"`
}
```

In the V1 API, the pod-level security attributes which are currently fields of the `PodSpec` are
retained on the `PodSpec` for backward compatibility purposes:

```go
package v1

type PodSpec struct {
    // Other fields omitted

    // Use the host's network namespace. If this option is set, the ports that will be
    // used must be specified.
    // Optional: Default to false.
    HostNetwork bool `json:"hostNetwork,omitempty"`
    // Use the host's pid namespace.
    // Optional: Default to false.
    HostPID bool `json:"hostPID,omitempty"`
    // Use the host's ipc namespace.
    // Optional: Default to false.
    HostIPC bool `json:"hostIPC,omitempty"`

    // Optional: SecurityContext specifies pod-level attributes and container security attributes
    // that apply to all containers.
    SecurityContext *PodSecurityContext `json:"securityContext,omitempty"`
}
```

The `pod.Spec.SecurityContext` specifies the security context of all containers in the pod.
The containers' `securityContext` field is overlaid on the base security context to determine the
effective security context for the container.

The new V1 API should be backward compatible with the existing API.  Backward compatibility is
defined as:

> 1.  Any API call (e.g. a structure POSTed to a REST endpoint) that worked before your change must
>     work the same after your change.
> 2.  Any API call that uses your change must not cause problems (e.g. crash or degrade behavior) when
>     issued against servers that do not include your change.
> 3.  It must be possible to round-trip your change (convert to different API versions and back) with
>     no loss of information.

Previous versions of this proposal attempted to deal with backward compatibility by defining
the affect of setting the pod-level fields on the container-level fields.  While trying to find
consensus on this design, it became apparent that this approach was going to be extremely complex
to implement, explain, and support.  Instead, we will approach backward compatibility as follows:

1.  Pod-level and container-level settings will not affect one another
2.  Old clients will be able to use container-level settings in the exact same way
3.  Container level settings always override pod-level settings if they are set

#### Examples

1.  Old client using `pod.Spec.Containers[x].SecurityContext`

    An old client creates a pod:

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      containers:
      - name: a
        securityContext:
          runAsUser: 1001
      - name: b
        securityContext:
          runAsUser: 1002
    ```

    looks to old clients like:

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      containers:
      - name: a
        securityContext:
          runAsUser: 1001
      - name: b
        securityContext:
          runAsUser: 1002
    ```

    looks to new clients like:

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      containers:
      - name: a
        securityContext:
          runAsUser: 1001
      - name: b
        securityContext:
          runAsUser: 1002
    ```

2.  New client using `pod.Spec.SecurityContext`

    A new client creates a pod using a field of `pod.Spec.SecurityContext`:

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      securityContext:
        runAsUser: 1001
      containers:
      - name: a
      - name: b
    ```

    appears to new clients as:

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      securityContext:
        runAsUser: 1001
      containers:
      - name: a
      - name: b
    ```

    old clients will see:

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      containers:
      - name: a
      - name: b
    ```

3.  Pods created using `pod.Spec.SecurityContext` and `pod.Spec.Containers[x].SecurityContext`

    If a field is set in both `pod.Spec.SecurityContext` and
    `pod.Spec.Containers[x].SecurityContext`, the value in `pod.Spec.Containers[x].SecurityContext`
    wins.  In the following pod:

    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: test-pod
    spec:
      securityContext:
        runAsUser: 1001
      containers:
      - name: a
        securityContext:
          runAsUser: 1002
      - name: b
    ```

    The effective setting for `runAsUser` for container A is `1002`.

#### Testing

A backward compatibility test suite will be established for the v1 API.  The test suite will
verify compatibility by converting objects into the internal API and back to the version API and
examining the results.

All of the examples here will be used as test-cases.  As more test cases are added, the proposal will
be updated.

An example of a test like this can be found in the
[OpenShift API package](https://github.com/openshift/origin/blob/master/pkg/api/compatibility_test.go)

E2E test cases will be added to test the correct determination of the security context for containers.

### Kubelet changes

1.  The Kubelet will use the new fields on the `PodSecurityContext` for host namespace control
2.  The Kubelet will be modified to correctly implement the backward compatibility and effective
    security context determination defined here

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/pod-security-context.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
