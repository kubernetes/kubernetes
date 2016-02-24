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

# Network Policy in Kubernetes

#### NOTE: The API described below is in alpha, and is subject to significant future change.

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Network Policy in Kubernetes](#network-policy-in-kubernetes)
      - [NOTE: The API described below is in alpha, and is subject to significant future change.](#note-the-api-described-below-is-in-alpha-and-is-subject-to-significant-future-change)
  - [Summary](#summary)
  - [v1alpha1 API](#v1alpha1-api)
    - [Behavior](#behavior)
    - [Enabling network policy](#enabling-network-policy)
    - [Namespace-level isolation](#namespace-level-isolation)
    - [NetworkPolicy objects](#networkpolicy-objects)
  - [Future Discussion](#future-discussion)

<!-- END MUNGE: GENERATED_TOC -->

## Summary

The [Kubernetes networking model](networking.md) assumes that each pod is
assigned an IP address and that all pods in the cluster are reachable by
all other pods and hosts in the cluster.

There are a number of use cases where you may want to restrict
communications between sets of pods on your cluster (e.g security,
multi-tenancy).  To allow this, Kubernetes provides an API for
network policy.

## v1alpha1 API

The `v1alpha1` network policy API is implemented using Kubernetes
[API extensions](../design/extending-api.md).

### Behavior

The default Kubernetes behavior is to allow all traffic from all sources
inside or outside the cluster to all pods within the cluster.

Using the v1alpha1 network policy API, network isolation can be defined
to limit connectivity from an optional set of traffic sources to an optional set of destination
TCP/UDP ports.

Even with network policy specified, pods will still be accessible by the host on which they are running.  This
is required for Kubelet health checks.

### Enabling network policy

To enable the necessary API extensions, your Kubernetes `apiserver`
must be started with the following option:

```
--runtime-config=extensions/v1beta1=true,extensions/v1beta1/thirdpartyresources=true
```

You can then enable the `NetworkPolicy` API extension by creating the following
`ThirdPartyResource` via kubectl.

```
kind: ThirdPartyResource
apiVersion: extensions/v1beta1
metadata:
  name: network-policy.net.alpha.kubernetes.io
description: "Specification for a network isolation policy"
versions:
- name: v1alpha1
```

### Namespace-level isolation

Pod isolation can be enabled at the namespace level using an annotation on
`Namespace` objects.

When isolation is enabled on a namespace, all incoming
connections to pods in that namespace from any source inside or outside
of the Kubernetes cluster will be denied unless otherwise
allowed by a `NetworkPolicy`.  When isolation is disabled on a namespace,
all incoming connections to pods in that namespace will be allowed.

By default (if no annotation is specified), network isolation
is disabled.

To enable isolation on a namespace:

```
kubectl annotate ns <namespace> "net.alpha.kubernetes.io/network-isolation=yes" --overwrite=true
```

To disable isolation on a namespace:

```
kubectl annotate ns <namespace> "net.alpha.kubernetes.io/network-isolation=no" --overwrite=true
```

### NetworkPolicy objects

The network policy API allows for fine grained access control using
`NetworkPolicy` third party resources. `NetworkPolicy` objects can be used
to override the default `net.alpha.kubernetes.io/network-isolation` behavior
defined at the namespace level.

`NetworkPolicy` objects are namespaced, and multiple `NetworkPolicy` objects
can be defined in a single namespace.  If multiple network policies select the
same pod, the rules from those policies will be combined additively.

`NetworkPolicy` objects have the following schema:

```
kind: NetworkPolicy
metadata:               # Standard Kubernetes metadata
  name: 
  namespace:
spec:
  podSelector:          # Selects pods to which this policy should be applied. 
  ingress:              # (Optional) List of allow rules.
    - ports:            # (Optional) List of dest ports to open.
      - port:           # (Optional) Numeric or named port 
        protocol:       # [ TCP | UDP]
      from:             # (Optional) List of sources.
       - pods:          # (Optional) Standard label selector.
         namespaces:    # (Optional) Standard label selector.
```

- `podSelector`: A standard label selector.  Selects which pods this
NetworkPolicy will apply to.
- `ingress`: A list of whitelist rules which define the set of traffic
which should be allowed.  Any traffic which does not match an `ingress`
rule will be denied access to any pods selected by `podSelector`.
  - `ports`: A list of destination port/protocol combinations to which
traffic is allowed.
  - `from`: List of allowed traffic sources.  Each `from` list item can select
either a set of pods or a set of namespaces using labels (but not both).

Note that while `kubectl` does not yet support creation and deletion of
`NetworkPolicy` objects, they can be managed directly using REST API calls.
e.g.

- `POST /apis/net.alpha.kubernetes.io/v1alpha1/namespaces/<namespace>/networkpolicys`
- `GET /apis/net.alpha.kubernetes.io/v1alpha1/networkpolicys`

>Note: `NetworkPolicy` specifications must be defined in `json` on the REST API.

## Future Discussion

There are a number of outstanding discussion points and features to be considered for this API as it progresses.

- **Egress policy**: Currently, the v1alph1 API makes no claims regarding policy on outgoing connections (all outgoing connections from a pod are allowed).  In the future,
it may make sense to augment this API with egress policy.

- **CIDR-based policy**: The current API does not allow limiting traffic based on source CIDR.  This is useful for
use-cases like applying policy on traffic from outside of the Kubernetes cluster.

This issue is tracking Kubernetes network policy: [#22469](https://github.com/kubernetes/kubernetes/issues/22469)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/network-policy.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
