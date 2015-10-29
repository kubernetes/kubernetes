<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Admission control plugin: LimitRanger

## Background

This document proposes a system for enforcing resource requirements constraints as part of admission control.

## Use cases

1. Ability to enumerate resource requirement constraints per namespace
2. Ability to enumerate min/max resource constraints for a pod
3. Ability to enumerate min/max resource constraints for a container
4. Ability to specify default resource limits for a container
5. Ability to specify default resource requests for a container
6. Ability to enforce a ratio between request and limit for a resource.

## Data Model

The **LimitRange** resource is scoped to a **Namespace**.

### Type

```go
// LimitType is a type of object that is limited
type LimitType string

const (
  // Limit that applies to all pods in a namespace
  LimitTypePod LimitType = "Pod"
  // Limit that applies to all containers in a namespace
  LimitTypeContainer LimitType = "Container"
)

// LimitRangeItem defines a min/max usage limit for any resource that matches on kind.
type LimitRangeItem struct {
  // Type of resource that this limit applies to.
  Type LimitType `json:"type,omitempty"`
  // Max usage constraints on this kind by resource name.
  Max ResourceList `json:"max,omitempty"`
  // Min usage constraints on this kind by resource name.
  Min ResourceList `json:"min,omitempty"`
  // Default resource requirement limit value by resource name if resource limit is omitted.
  Default ResourceList `json:"default,omitempty"`
  // DefaultRequest is the default resource requirement request value by resource name if resource request is omitted.
  DefaultRequest ResourceList `json:"defaultRequest,omitempty"`
  // MaxLimitRequestRatio if specified, the named resource must have a request and limit that are both non-zero where limit divided by request is less than or equal to the enumerated value; this represents the max burst for the named resource.
  MaxLimitRequestRatio ResourceList `json:"maxLimitRequestRatio,omitempty"`
}

// LimitRangeSpec defines a min/max usage limit for resources that match on kind.
type LimitRangeSpec struct {
  // Limits is the list of LimitRangeItem objects that are enforced.
  Limits []LimitRangeItem `json:"limits"`
}

// LimitRange sets resource usage limits for each kind of resource in a Namespace.
type LimitRange struct {
  TypeMeta `json:",inline"`
  // Standard object's metadata.
  // More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#metadata
  ObjectMeta `json:"metadata,omitempty"`

  // Spec defines the limits enforced.
  // More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#spec-and-status
  Spec LimitRangeSpec `json:"spec,omitempty"`
}

// LimitRangeList is a list of LimitRange items.
type LimitRangeList struct {
  TypeMeta `json:",inline"`
  // Standard list metadata.
  // More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#types-kinds
  ListMeta `json:"metadata,omitempty"`

  // Items is a list of LimitRange objects.
  // More info: http://releases.k8s.io/release-1.1/docs/design/admission_control_limit_range.md
  Items []LimitRange `json:"items"`
}
```

### Validation

Validation of a **LimitRange** enforces that for a given named resource the following rules apply:

Min (if specified) <= DefaultRequest (if specified) <= Default (if specified) <= Max (if specified)

### Default Value Behavior

The following default value behaviors are applied to a LimitRange for a given named resource.

```
if LimitRangeItem.Default[resourceName] is undefined 
  if LimitRangeItem.Max[resourceName] is defined
    LimitRangeItem.Default[resourceName] = LimitRangeItem.Max[resourceName]
```

```
if LimitRangeItem.DefaultRequest[resourceName] is undefined
  if LimitRangeItem.Default[resourceName] is defined
    LimitRangeItem.DefaultRequest[resourceName] = LimitRangeItem.Default[resourceName]
  else if LimitRangeItem.Min[resourceName] is defined
    LimitRangeItem.DefaultRequest[resourceName] = LimitRangeItem.Min[resourceName]
```

## AdmissionControl plugin: LimitRanger

The **LimitRanger** plug-in introspects all incoming pod requests and evaluates the constraints defined on a LimitRange.

If a constraint is not specified for an enumerated resource, it is not enforced or tracked.

To enable the plug-in and support for LimitRange, the kube-apiserver must be configured as follows:

```console
$ kube-apiserver --admission-control=LimitRanger
```

### Enforcement of constraints

**Type: Container**

Supported Resources:

1. memory
2. cpu

Supported Constraints:

Per container, the following must hold true

| Constraint | Behavior |
| ---------- | -------- |
| Min | Min <= Request (required) <= Limit (optional) |
| Max | Limit (required) <= Max |
| LimitRequestRatio | LimitRequestRatio <= ( Limit (required, non-zero) / Request (required, non-zero)) |

Supported Defaults:

1. Default - if the named resource has no enumerated value, the Limit is equal to the Default
2. DefaultRequest - if the named resource has no enumerated value, the Request is equal to the DefaultRequest

**Type: Pod**

Supported Resources:

1. memory
2. cpu

Supported Constraints:

Across all containers in pod, the following must hold true

| Constraint | Behavior |
| ---------- | -------- |
| Min | Min <= Request (required) <= Limit (optional) |
| Max | Limit (required) <= Max |
| LimitRequestRatio | LimitRequestRatio <= ( Limit (required, non-zero) / Request (non-zero) ) |

## Run-time configuration

The default ```LimitRange``` that is applied via Salt configuration will be updated as follows:

```
apiVersion: "v1"
kind: "LimitRange"
metadata:
  name: "limits"
  namespace: default
spec:
  limits:
    - type: "Container"
      defaultRequests:
        cpu: "100m"
```

## Example

An example LimitRange configuration:

| Type | Resource | Min | Max | Default | DefaultRequest | LimitRequestRatio |
| ---- | -------- | --- | --- | ------- | -------------- | ----------------- |
| Container | cpu | .1 | 1 | 500m | 250m | 4 |
| Container | memory | 250Mi | 1Gi | 500Mi | 250Mi | |

Assuming an incoming container that specified no incoming resource requirements,
the following would happen.

1. The incoming container cpu would request 250m with a limit of 500m.
2. The incoming container memory would request 250Mi with a limit of 500Mi
3. If the container is later resized, it's cpu would be constrained to between .1 and 1 and the ratio of limit to request could not exceed 4.



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/admission_control_limit_range.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
