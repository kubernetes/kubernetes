<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


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
>NOTE: In v1beta1 the Namespace changes will be implemented as an annotation.

```go
type IngressIsolationPolicy string

const (
	// Deny all ingress traffic to pods in this namespace. Ingress means 
	// any incoming traffic to pods, whether that be from other pods within this namespace
	// or any source outside of this namespace.
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
	// pods in this namespace are denied ingress traffic by default.  When not defined,
	// the cluster default ingress isolation policy is applied (currently allow all). 
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
    net.beta.kubernetes.io/network-policy: |
      {
        "ingress": {
          "isolation": "DefaultDeny"
        }
      }
```

### NetworkPolicy Go Definition

For a namespace with ingress isolation, connections to pods in that namespace (from any source) are prevented.
The user needs a way to explicitly declare which connections are allowed into pods of that namespace.

This is accomplished through ingress rules on `NetworkPolicy`
objects (of which there can be multiple in a single namespace).  Pods selected by
one or more NetworkPolicy objects should allow any incoming connections that match any
ingress rule on those NetworkPolicy objects, per the network plugin’s capabilities.

NetworkPolicy objects and the above namespace isolation both act on _connections_ rather than individual packets.  That is to say that if traffic from pod A to pod B is allowed by the configured
policy, then the return packets for that connection from B -> A are also allowed, even if the policy in place would not allow B to initiate a connection to A.  NetworkPolicy objects act on a broad definition of _connection_ which includes both TCP and UDP streams.   If new network policy is applied that would block an existing connection between two endpoints, the enforcer of policy
should terminate and block the existing connection as soon as can be expected by the implementation.

We propose adding the new NetworkPolicy object to the `extensions/v1beta1` API group for now.

The SIG also considered the following while developing the proposed NetworkPolicy object:
- A per-pod policy field.  We discounted this in favor of the loose coupling that labels provide, similar to Services.
- Per-Service policy.  We chose not to attach network policy to services to avoid semantic overloading of a single object, and conflating the existing semantics of load-balancing and service discovery with those of network policy.

```go
type NetworkPolicy struct {
	TypeMeta
	ObjectMeta
	
	// Specification of the desired behavior for this NetworkPolicy.
	Spec NetworkPolicySpec 
}

type NetworkPolicySpec struct {
	// Selects the pods to which this NetworkPolicy object applies.  The array of ingress rules 
	// is applied to any pods selected by this field. Multiple network policies can select the 
	// same set of pods.  In this case, the ingress rules for each are combined additively.
	// This field is NOT optional and follows standard unversioned.LabelSelector semantics.  
	// An empty podSelector matches all pods in this namespace.
	PodSelector unversioned.LabelSelector `json:"podSelector"`
	
	// List of ingress rules to be applied to the selected pods.
	// Traffic is allowed to a pod if namespace.networkPolicy.ingress.isolation is undefined and cluster policy allows it, 
	// OR if the traffic source is the pod's local node, 
	// OR if the traffic matches at least one ingress rule across all of the NetworkPolicy 
	// objects whose podSelector matches the pod.  
	// If this field is empty then this NetworkPolicy does not affect ingress isolation.
	// If this field is present and contains at least one rule, this policy allows any traffic
	// which matches at least one of the ingress rules in this list.
	Ingress []NetworkPolicyIngressRule `json:"ingress,omitempty"`
}

// This NetworkPolicyIngressRule matches traffic if and only if the traffic matches both ports AND from. 
type NetworkPolicyIngressRule struct {
	// List of ports which should be made accessible on the pods selected for this rule. 
	// Each item in this list is combined using a logical OR.  
	// If this field is not provided, this rule matches all ports (traffic not restricted by port). 
	// If this field is empty, this rule matches no ports (no traffic matches).
	// If this field is present and contains at least one item, then this rule allows traffic 
	// only if the traffic matches at least one port in the ports list. 
	Ports *[]NetworkPolicyPort `json:"ports,omitempty"`
	
	// List of sources which should be able to access the pods selected for this rule.
	// Items in this list are combined using a logical OR operation.
	// If this field is not provided, this rule matches all sources (traffic not restricted by source).
	// If this field is empty, this rule matches no sources (no traffic matches). 
	// If this field is present and contains at least on item, this rule allows traffic only if the 
	// traffic matches at least one item in the from list. 
	From *[]NetworkPolicyPeer `json:"from,omitempty"`
}

type NetworkPolicyPort struct {
	// Optional.  The protocol (TCP or UDP) which traffic must match.
	// If not specified, this field defaults to TCP. 
	Protocol *api.Protocol `json:"protocol,omitempty"`
	
	// If specified, the port on the given protocol.  This can 
	// either be a numerical or named port.  If this field is not provided,
	// this matches all port names and numbers.
	// If present, only traffic on the specified protocol AND port
	// will be matched.
	Port *intstr.IntOrString `json:"port,omitempty"`
}

type NetworkPolicyPeer struct {
	// Exactly one of the following must be specified.

	// This is a label selector which selects Pods in this namespace.
	// This field follows standard unversioned.LabelSelector semantics.
	// If present but empty, this selector selects all pods in this namespace.
	PodSelector *unversioned.LabelSelector `json:"podSelector,omitempty"`
	
	// Selects Namespaces using cluster scoped-labels.  This 
	// matches all pods in all namespaces selected by this label selector. 
	// This field follows standard unversioned.LabelSelector semantics.
	// If present but empty, this selector selects all namespaces.
	NamespaceSelector *unversioned.LabelSelector `json:"namespaceSelector,omitempty"`
}
```

### Behavior

The following pseudo-code attempts to define when traffic is allowed to a given pod when using this API.

```python
def is_traffic_allowed(traffic, pod):
  """
  Returns True if traffic is allowed to this pod, False otherwise.
  """
  if not pod.Namespace.Spec.NetworkPolicy.Ingress.Isolation:
    # If ingress isolation is disabled on the Namespace, use cluster default.
    return clusterDefault(traffic, pod)
  elif traffic.source == pod.node.kubelet:
    # Traffic is from kubelet health checks.
    return True
  else:
    # If namespace ingress isolation is enabled, only allow traffic 
    # that matches a network policy which selects this pod.
    for network_policy in network_policies(pod.Namespace):
      if not network_policy.Spec.PodSelector.selects(pod):
        # This policy doesn't select this pod. Try the next one. 
        continue

      # This policy selects this pod.  Check each ingress rule 
      # defined on this policy to see if it allows the traffic.
      # If at least one does, then the traffic is allowed.
      for ingress_rule in network_policy.Ingress or []:
        if ingress_rule.matches(traffic): 
          return True 

  # Ingress isolation is DefaultDeny and no policies match the given pod and traffic.
  return false 
```

### Potential Future Work / Questions

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
    net.beta.kubernetes.io/network-policy: |
      {
        "ingress": {
          "isolation": "DefaultDeny"
        }
      }
---
kind: NetworkPolicy
apiVersion: extensions/v1beta1 
metadata:
  name: allow-frontend
  namespace: myns
spec:
  podSelector:            
    matchLabels:
      role: backend
  ingress:                
    - from:              
        - podSelector:
            matchLabels:
              role: frontend
      ports:
        - protocol: TCP
          port: 6379
```

2) Allow TCP 443 from any source in Bob's namespaces.

```yaml
kind: NetworkPolicy
apiVersion: extensions/v1beta1 
metadata:
  name: allow-tcp-443
spec:
  podSelector:            
    matchLabels:
      role: frontend 
  ingress:
    - ports:
        - protocol: TCP
          port: 443 
      from:
        - namespaceSelector:
            matchLabels:
              user: bob 
```

3) Allow all traffic to all pods in this namespace.

```yaml
kind: NetworkPolicy
apiVersion: extensions/v1beta1
metadata:
  name: allow-all
spec:
  podSelector:
  ingress:
  - {}
```

## References

- https://github.com/kubernetes/kubernetes/issues/22469 tracks network policy in kubernetes.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/network-policy.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
