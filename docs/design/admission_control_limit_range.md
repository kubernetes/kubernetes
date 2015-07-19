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

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/design/admission_control_limit_range.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Admission control plugin: LimitRanger

## Background

This document proposes a system for enforcing min/max limits per resource as part of admission control.

## Model Changes

A new resource, **LimitRange**, is introduced to enumerate min/max limits for a resource type scoped to a
Kubernetes namespace.

```go
const (
  // Limit that applies to all pods in a namespace
  LimitTypePod string = "Pod"
  // Limit that applies to all containers in a namespace
  LimitTypeContainer string = "Container"
)

// LimitRangeItem defines a min/max usage limit for any resource that matches on kind
type LimitRangeItem struct {
  // Type of resource that this limit applies to
  Type string `json:"type,omitempty"`
  // Max usage constraints on this kind by resource name
  Max ResourceList `json:"max,omitempty"`
  // Min usage constraints on this kind by resource name
  Min ResourceList `json:"min,omitempty"`
  // Default usage constraints on this kind by resource name
  Default ResourceList `json:"default,omitempty"`
}

// LimitRangeSpec defines a min/max usage limit for resources that match on kind
type LimitRangeSpec struct {
  // Limits is the list of LimitRangeItem objects that are enforced
  Limits []LimitRangeItem `json:"limits"`
}

// LimitRange sets resource usage limits for each kind of resource in a Namespace
type LimitRange struct {
  TypeMeta   `json:",inline"`
  ObjectMeta `json:"metadata,omitempty"`

  // Spec defines the limits enforced
  Spec LimitRangeSpec `json:"spec,omitempty"`
}

// LimitRangeList is a list of LimitRange items.
type LimitRangeList struct {
  TypeMeta `json:",inline"`
  ListMeta `json:"metadata,omitempty"`

  // Items is a list of LimitRange objects
  Items []LimitRange `json:"items"`
}
```

## AdmissionControl plugin: LimitRanger

The **LimitRanger** plug-in introspects all incoming admission requests.

It makes decisions by evaluating the incoming object against all defined **LimitRange** objects in the request context namespace.

The following min/max limits are imposed:

**Type: Container**

| ResourceName | Description |
| ------------ | ----------- |
| cpu | Min/Max amount of cpu per container |
| memory | Min/Max amount of memory per container |

**Type: Pod**

| ResourceName | Description |
| ------------ | ----------- |
| cpu | Min/Max amount of cpu per pod |
| memory | Min/Max amount of memory per pod |

If a resource specifies a default value, it may get applied on the incoming resource.  For example, if a default
value is provided for container cpu, it is set on the incoming container if and only if the incoming container
does not specify a resource requirements limit field.

If a resource specifies a min value, it may get applied on the incoming resource.  For example, if a min
value is provided for container cpu, it is set on the incoming container if and only if the incoming container does
not specify a resource requirements requests field.

If the incoming object would cause a violation of the enumerated constraints, the request is denied with a set of
messages explaining what constraints were the source of the denial.

If a constraint is not enumerated by a **LimitRange** it is not tracked.

## kube-apiserver

The server is updated to be aware of **LimitRange** objects.

The constraints are only enforced if the kube-apiserver is started as follows:

```console
$ kube-apiserver -admission_control=LimitRanger
```

## kubectl

kubectl is modified to support the **LimitRange** resource.

`kubectl describe` provides a human-readable output of limits.

For example,

```console
$ kubectl namespace myspace
$ kubectl create -f docs/user-guide/limitrange/limits.yaml
$ kubectl get limits
NAME
limits
$ kubectl describe limits limits
Name:           limits
Type            Resource        Min     Max     Default
----            --------        ---     ---     ---
Pod             memory          1Mi     1Gi     -
Pod             cpu             250m    2       -
Container       memory          1Mi     1Gi     1Mi
Container       cpu             250m    250m    250m
```

## Future Enhancements: Define limits for a particular pod or container.

In the current proposal, the **LimitRangeItem** matches purely on **LimitRangeItem.Type**

It is expected we will want to define limits for particular pods or containers by name/uid and label/field selector.

To make a **LimitRangeItem** more restrictive, we will intend to add these additional restrictions at a future point in time.

## Example

See the [example of Limit Range](../user-guide/limitrange/) for more information.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/admission_control_limit_range.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
