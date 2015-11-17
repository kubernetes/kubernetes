# Native API Support for Network Policy
#### Abstract
A proposal for adding the API primitives to describe flexible network policy and tenant isolation.

#### Goals
- Add first-class support for flexible network policy and tenant isolation to the API.
- Provide an intent-based API which uses native Kubernetes concepts (labels, namespaces, services).
- Abstract policy and multi-tenancy from the underlying networking technology / implementation.
- Back-compatible with existing network plugins.  Network plugins can choose which (if any) of this API is supported.

#### Proposed Design
- A new namespaced object called `NetworkProfile` will be added.  A `NetworkProfile` represents a communication grouping of pods, meaning that pods within a `NetworkProfile` can communicate with each other.  Multiple `NetworkProfile`s can be applied to a single pod, allowing that pod to reach and be reached by multiple communication groups.

- Labels and selectors are a key feature of Kubernetes.  As such, a new object called `NetworkProfileSelector` is introduced which uses labels to apply `NetworkProfile`s to a selection of pods.  This allows pods to dynamically enter and exit a `NetworkProfile` based on their labels.

- A default `NetworkProfile` will exist on a per-namespace basis.  Each pod created in a given namespace that is not selected by a `NetworkProfileSelector` will be assigned to the default `NetworkProfile` for its namespace.  This can be overridden if a `NetworkProfileSelector` later selects this pod.  The default `NetworkProfile` for a namespace can be configured, but by default is the reserved `OpenProfile`.

- The above points allow for tenant isolation using Kubernetes Namespaces as the isolation boundary.  Additional `NetworkProfiles` can futher subdivide a given Kubernetes namespace.

- To allow for communication between Namespaces as well as from outside of the cluster, Kubernetes Services can be used to expose applications outside of their `NetworkProfile`(s).  A new field on the Service will allow application developers to specify that the selected pods should be part of the `OpenProfile`, thus allowing access to the service.  It may be useful to extend this to allow arbitrary `NetworkProfile` objects to be applied. 

##### New API Objects
The following first-class API objects will be added.
```

const (
  // The default OpenProfile.
  NetworkProfileOpen NetworkProfile = "OpenProfile"
)

type NetworkProfileSelectorSpec struct {
	// Pod label selector.
	Selector map[string]string `json:"selector,omitempty"`

	// List of NetworkProfiles to apply.
	Profiles []NetworkProfile `json:"profiles,omitempty"`
}

type NetworkProfileSelector struct {
	// Standard object's metadata.
	ObjectMeta `json:"metadata,omitempty"`

	// Spec
	Spec NetworkProfileSelectorSpec `json:"metadata,omitempty"`
}

// Represents a grouping of pods that can communicate.
type NetworkProfile struct {
	// Standard object's metadata.
	ObjectMeta `json:"metadata,omitempty"`
}
```

##### Network Plugin API
To implement the above API, the network plugin must be aware of which `NetworkProfile`s have been applied to which pods.  To avoid requiring that each network plugin watch the Kubernetes API to construct this information, a new hook `SetProfiles` will be added to the network plugin API, which informs the plugin of all `NetworkProfile` objects applied to the given pod.

##### Example 
The following set of objects configures two isolated tenants using Namespaces and exposes a public service in one.

Namespaces:
```
apiVersion: v1
kind: Namespace
defaultProfile: tenantProfileA
metadata:
  name: tenant-A 
--------------------
apiVersion: v1
kind: Namespace
defaultProfile: tenantProfileB
metadata:
  name: tenant-B 
```
Because of the default profiles configured, no pods in `tenant-A` can communicate with pods in `tenant-B`.

Tenant A has intra-namespace isolation using NetworkProfiles. Application-1 is isolated from the rest of the Namespace.
```
apiVersion: v1
kind: NetworkProfileSelector 
namespace: tenant-A 
metadata:
  name: application-1 
spec:
  selector:
    application: one 
  profiles:
    - metadata:
        name: application-1 
```

Tenant A also exposes application-1's frontend Service, so it is accessible outside of the namespace.
```
kind: Service
profile: OpenProfile 
namespace: tenant-A 
metadata:
  name: frontend 
spec:
  ports:
  - port: 80 
    targetPort: 80
    protocol: TCP
  selector:
    appplication: one
    role: frontend 
```
