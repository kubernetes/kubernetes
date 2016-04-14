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

# NetworkPolicy

## Abstract

A proposal for implementing a new resource - NetworkPolicy - which
will enable definition of ingress policies for selections of pods.

The design for this proposal has been created by, and discussed
extensively within the Kubernetes networking SIG.  It has been implemented
and tested using Kubernetes API extensions by various networking solutions already.

In this design, users can create various NetworkPolicy objects which select groups of pods and
define how those pods should be allowed to communicate with each other.  The
implementation of that policy at the network layer is left up to the
chosen networking solution.

> Note that this proposal does not yet include egress / cidr-based policy, which is still actively undergoing discussion in the SIG.  In general,
egress is expected to augment this proposal rather than modify it, with one exception (noted below).

## Implementation

The implmentation in Kubernetes consists of:
- A v1beta NetworkPolicy API object
- A field on the `Namespace` object to control isolation.

### Namespace changes

> Note that this section is subject to some change with the addition of egress policy as mentioned above.

The `Namespace` object will be augmented with the field `networkIsolation` which will take one of the following values:

When `networkIsolation=true` is set on a namespace:
- Pods in that Namespace will not be accessible from any other source, even sources within that namespace, unless explicitly allowed by a NetworkPolicy object.
- Pods in that Namespace will be able to access any other source (egress is uninhibited).

When `networkIsolation=false` is set on a namespace:
- Pods in that Namespace will be accessible from any other source.
- Pods in that Namespace will be able to access any other source.

The `networkIsolation` field will default to `false` if not specified.

### NetworkPolicy YAML Definition and Behavior

Once the namespace is isolated, a mechanism to selectively allow traffic into the namespace and between pods within
the namespace is required.  That is accomplished through ingress rules on `NetworkPolicy`
objects (of which there can be multiple in a single namespace).  Pods selected by
one or more NetworkPolicy objects should allow any incoming connections that match any
ingress rule on those NetworkPolicy objects, per the network plugin’s capabilities.

If `networkIsolation=false` on a namespace, then all traffic is allowed to / from pods in that namespace independent of any 
NetworkPolicy objects what may have selected them.

```yaml
kind: NetworkPolicy
apiVersion: v1beta 
metadata:
  name:
  namespace:
spec:
  podSelector:            // Standard label selector - selects pods.  
  ingress:                // List of ingress rules (optional).             
    - ports:              // List of allowed ports / protocols (optional).          
        - port:           // Port on the specified protocol (optional). 
          protocol:       // Protocol (TCP, UDP) 
      from:               // List of allowed sources (optional).    
        - pods:           // Label selector - selects Pods (optional). 
          namespaces:     // Label selector - selects Namespaces (optional).
```

Each NetworkPolicy object supports the following fields:
- `podSelector`: A label selector which selects which pods this `NetworkPolicy` applies to.
- `ingress`: A list of ingress rules.  If not present, this `NetworkPolicy` has no effect on ingress to the selected pods.

Each ingress rule supports the following fields:
- `ports`: A list of protocol / ports which should be accessible on the selected pods. If not defined, then access to the selected pods will not be restricted by port/protocol.
- `from`: A list of source criteria which dictate which sources should be able to access the pods selected. If no `from` key is provided, then access will not be restricted based on source.

Each item in the `ports` list supports the following fields.  Must match `port` AND `protocol` if both are defined.
- `protocol`: The protocol for the given port - TCP or UDP.
- `port`: The port on the specified protocol.

Each item in the `from` list may declare at most ONE of the following:
- `pods`: A label selector which selects pods (in the same namespace as the NetworkPolicy).
- `namespaces`: A label selector which selects namespaces.

Note:
- For a given pod, any ingress traffic which does not match one of the ingress rules from any of the NetworkPolicy objects which select it will be dropped via a default
drop behavior.  This only occurs when `networkIsolation=false` on the namespace.
- All ingress rules are whitelist rules, meaning that it should be easy to resolve the case where multiple
NetworkPolicy objects select the same set of pods, as there can be no conflicting rules.
- All pods will always be accessible from the host that they are running on.  This is required to allow for kubelet health checks.

### Go Representation

```go
type NetworkPolicy struct {
  TypeMeta
  ObjectMeta

  Spec NetworkPolicySpec 
}

type NetworkPolicySpec struct {
  // Selects the pods which this NetworkPolicy object
  // applies to.
  PodSelector *unversioned.LabelSelector `json:"podSelector,omitempty"`

  // List of ingress rules to be applied to the 
  // selected pods.
  Ingress []NetworkPolicyIngressRule `json:"ingress,omitempty"`
}

type NetworkPolicyIngressRule struct {
  // List of allowed ports. 
  Ports []NetworkPolicyPort `json:"ports,omitempty"`

  // List of allowed sources.
  From []NetworkPolicySource `json:"from,omitempty"`
}

type NetworkPolicyPort struct {
  // If specified, the port on the given protocol.
  Port int32 `json:"port,omitempty"`

  // The protocol - TCP or UDP.
  Protocol string `json:"protocol"`
}

type NetworkPolicySource struct {
  // Label selector - selects pods in this namespace.
  // If 'Namespaces' is defined, this must not be.
  Pods *unversioned.LabelSelector `json:"pods,omitempty"`

  // Label selector - selects namespaces.
  // If 'Pods' is defined, this must not be.
  Namespaces *unversioned.LabelSelector `json:"namespaces,omitempty"`
}
```


## References

- https://github.com/kubernetes/kubernetes/issues/22469 tracks network policy in kubernetes.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/network-policy.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
