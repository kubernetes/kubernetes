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

> Note that this proposal does not yet include egress / cidr-based policy, which is still actively undergoing discussion in the SIG. These are expected to augment this proposal in a backwards compatible way.

## Implementation

The implmentation in Kubernetes consists of:
- A v1beta1 NetworkPolicy API object
- A structure on the `Namespace` object to control policy, to be developed as an annotation for now.

### Namespace changes

The following objects will be defined on a Namespace Spec.
>NOTE: In v1beta1 these objects will be implemented as an annotation.

```go
type IngressIsolationPolicy string

const (
	// Deny all ingress traffic to pods in this namespace.
	DefaultDeny IngressIsolationPolicy = "DefaultDeny"
) 

// Standard NamespaceSpec object, modified to include a new
// NamespaceNetworkPolicy field.
type NamespaceSpec struct {
	// This is a pointer so that it can be left undefined.
	NetworkPolicy *NamespaceNetworkPolicy `json:"networkPolicy,omitempty"`
}

type NamespaceNetworkPolicy struct {
	// Ingress configuration for this namespace.  This config is 
	// applied to all pods within this namespace. For now, only 
	// ingress is supported.  This field is optional - if not 
	// defined, then the cluster default for ingress is applied.
	Ingress *NamespaceIngressPolicy `json:"ingress,omitempty"`
}

// Configuration for ingress to pods within this namespace.
// For now, this only supports specifying an isolation policy.
type NamespaceIngressPolicy struct {
	// The isolation policy to apply to pods in this namespace.
	// Currently this field only supports "DefaultDeny", but could 
	// be extended to support other policies in the future.  When set to DefaultDeny,
	// pods in this namespace are denied ingress traffic.  When not defined,
	// the cluster default ingress policy is applied (currently allow all). 
	Isolation *IngressIsolationPolicy `json:"isolation,omitempty"` 
}
```

```yaml
kind: Namespace
apiVersion: v1
spec:
  networkPolicy:
    ingress:
      isolation: DefaultDeny
```

The above structures will be represented in v1beta1 as a json encoded annotation like so:

```yaml
kind: Namespace
apiVersion: v1
metadata:
  annotations:
    net.beta.kubernetes.io/network-policy: "{\"ingress\": {\"isolation\": \"DefaultDeny\"}" 
```

### NetworkPolicy Go Definition

Once a Namespace is isolated, a mechanism to selectively allow traffic into the namespace and between pods within
the namespace is required.  That is accomplished through ingress rules on `NetworkPolicy`
objects (of which there can be multiple in a single namespace).  Pods selected by
one or more NetworkPolicy objects should allow any incoming connections that match any
ingress rule on those NetworkPolicy objects, per the network plugin’s capabilities.

If ingress `isolation` is not specified on a namespace, then all traffic is allowed to pods in that namespace.

NetworkPolicy objects and the above namespace isolation both act on _connections_ rather than packets.  That is to say that if traffic from pod A to pod B is allowed by the configured
policy, then the return packets for that connection from B -> A are also allowed, even if policy is in place that would prevent 
B from initiating a connection to A.  If new network policy is applied that would block an existing connection between two endpoints, the enforcer of policy 
should terminate and block the existing connection as soon as can be expected by the implementation.

We propose adding a new `/apis/networking` API group for the NetworkPolicy object.

```go
type NetworkPolicy struct {
  TypeMeta
  ObjectMeta

  // Specification of the desired behavior for this NetworkPolicy.
  Spec NetworkPolicySpec 
}

type NetworkPolicySpec struct {
  // Selects the pods to which this NetworkPolicy object applies.  The array of NetworkPolicyIngressRules below 
  // is applied to any pods selected by this field. Multiple NetworkPolicy objects can select the 
  // same set of pods.  In this case, the NetworkPolicyRules for each are combined additively.
  // This field is NOT optional and follows standard unversioned.LabelSelector semantics.  
  // An empty PodSelector matches all pods in this namespace.
  PodSelector unversioned.LabelSelector `json:"podSelector"`

  // List of ingress rules to be applied to the selected pods.
  // Traffic is allowed to a pod if Namespace.NetworkPolicy.Ingress.Isolation is undefined, 
  // OR if the traffic source is the pod's local kubelet (for health checks), 
  // OR if Namespace.NetworkPolicy.Ingress.Isolation=DefaultDeny and the traffic matches at least 
  // one NetworkPolicyIngressRule across all of the NetworkPolicy 
  // objects whose podSelector matches the pod.  
  // If this field is nil, this NetworkPolicy allows all traffic.
  // If this field is non-nil but contains no rules, this NetworkPolicy allows no traffic.
  // If this field is non-nil and contains at least one rule, this NetworkPolicy allows any traffic
  // which matches at least one of the NetworkPolicyIngressRules in this list.
  Ingress []NetworkPolicyIngressRule `json:"ingress,omitempty"`
}

// This NetworkPolicyIngressRule matches traffic if and only if the traffic matches both Ports AND From. 
type NetworkPolicyIngressRule struct {
  // List of ports which should be made accessible on the pods selected by PodSelector.  
  // Each item in this list is combined using a logical OR.  
  // If this field is nil, this NetworkPolicyIngressRule matches all ports (traffic not restricted by Port). 
  // If this field is non-nil but contains no items, this NetworkPolicyIngressRule matches no ports (no traffic matches).
  // If this field is non-nil and contains at least one item, then this NetworkPolicyIngressRule allows traffic 
  // only if the traffic matches at least one NetworkPolicyPort in the Ports list. 
  Ports []NetworkPolicyPort `json:"ports,omitempty"`

  // List of sources which should be able to access the pods selected by PodSelector.
  // Items in this list are combined using a logical OR operation.
  // If this field nil, this NetworkPolicyIngressRule matches all sources (traffic not restricted by source).
  // If this field is non-nil but contains no items, this NetworkPolicyIngressRule matches no sources (no traffic matches). 
  // If this field is non-nil and contains at least on item, this NetworkPolicyIngressRule allows traffic only if the 
  // traffic matches at least one NetworkPolicyPeer in the From list. 
  From []NetworkPolicyPeer `json:"from,omitempty"`
}

type NetworkPolicyPort struct {
  // The protocol (TCP or UDP) which traffic must match.
  // If not defined, this field defaults to TCP. 
  Protocol api.Protocol `json:"protocol"`

  // If specified, the port on the given protocol.  This can 
  // either be a numerical or named port.  If this field is nil,
  // this NetworkPolicyPort matches all port names and numbers.
  // If non-nil, only traffic on the specified protocol AND port
  // will be matched by this NetworkPolicyPort.
  Port *intstr.IntOrString `json:"port,omitempty"`
}

type NetworkPolicyPeer struct {
  // If 'Namespaces' is defined, 'Pods' must not be.
  // This is a label selector which selects Pods in this namespace.
  // This NetworkPolicyPeer matches any pods selected by this selector.
  // This field follows standard unversioned.LabelSelector semantics.
  // If nil, this selector selects no pods.
  // If non-nil but empty, this selector selects all pods in this namespace.
  Pods *unversioned.LabelSelector `json:"pods,omitempty"`

  // If 'Pods' is defined, 'Namespaces' must not be.
  // Selects Kubernetes Namespaces.  This NetworkPolicyPeer matches 
  // all pods in all namespaces selected by this label selector. 
  // This field follows standard unversioned.LabelSelector semantics.
  // If nil, this selector selects no namespaces.
  // If non-nil but empty, this selector selects all namespaces.
  Namespaces *unversioned.LabelSelector `json:"namespaces,omitempty"`
}
```

### Behavior

The following pseudo-code attempts to define when traffic is allowed to a given pod when using this API.

```python
def is_traffic_allowed(traffic, pod):
  """
  Returns True if traffic is allowed to this pod, False otherwise.
  """
  if not namespace.networkPolicy.ingress.isolation:
    # If ingress isolation is disabled on the Namespace, all traffic is allowed.
    return True 
  elif traffic.source == pod.node.kubelet:
    # Traffic is from kubelet health checks.
    return True
  else:
    # If namespace ingress isolation is enabled, only allow traffic 
    # that matches a network policy which selects this pod.
    for network_policy in network_policies(pod.namespace):
      if not network_policy.podSelector.selects(pod):
        # This policy doesn't select this pod. Try the next one. 
        continue

      # A null ingress list means allow all traffic.
      if network_policy.ingress == null:
        return True

      # This policy selects this pod.  Check each ingress rule 
      # defined on this policy to see if it allows the traffic.
      # If at least one does, then the traffic is allowed.
      for ingress_rule in network_policy.ingress:
        if ingress_rule.matches(traffic): 
          return True 

  # Ingress isolation is DefaultDeny and no policies match the given pod and traffic.
  return False
```

### Open Questions

- A single podSelector per NetworkPolicy may lead to managing a large number of NetworkPolicy objects, each of which is small and easy to understand on its own. However, this may lead for a policy change to require touching several policy objects. Allowing an optional podSelector per ingress rule additionally to the podSelector per NetworkPolicy object would allow the user to group rules into logical segments and define size/complexity ratio where it makes sense. This may lead to a smaller number of objects with more complexity if the user opts in to the additional podSelector.  This increases the complexity of the NetworkPolicy object itself. This proposal has opted to favor a larger number of smaller objects that are easier to understand, with the understanding that additional podSelectors could be added to this design in the future should the requirement become apparent.

- Is the `Namespaces` selector in the `NetworkPolicyPeer` struct too coarse? Do we need to support the AND combination of `Namespaces` and `Pods`?

### Examples

1) Only allow traffic from frontend pods on TCP port 6379 to backend pods in the same namespace.

```yaml
kind: Namespace
apiVersion: v1
metadata:
  name: myns
  annotations:
    net.beta.kubernetes.io/network-policy: "{\"ingress\": {\"isolation\": \"DefaultDeny\"}"
---
kind: NetworkPolicy
apiVersion: v1beta1 
metadata:
  name: allow-frontend
  namespace: myns
spec:
  podSelector:            
    role: backend
  ingress:                
    - from:              
        - pods:
            role: frontend
      ports:
        - protocol: TCP
          port: 6379
```

2) Allow TCP 443 from any source in Bob's namespaces.

```yaml
kind: NetworkPolicy
apiVersion: v1beta1 
metadata:
  name: allow-tcp-443
spec:
  podSelector:            
    role: frontend 
  ingress:
    - ports:
        - protocol: TCP
          port: 443 
      from:
        - namespaces:
            user: bob 
```

3) Allow all traffic to all pods in this namespace.

```yaml
kind: NetworkPolicy
apiVersion: v1beta1 
metadata:
  name: allow-all
spec:
  podSelector:            
```

## References

- https://github.com/kubernetes/kubernetes/issues/22469 tracks network policy in kubernetes.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/network-policy.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
