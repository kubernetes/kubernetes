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
[here](http://releases.k8s.io/release-1.0/docs/proposals/multi-tenant-networking.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Multi-tenant networking and Network Providers

## Abstract

A proposal for multi-tenant networking through network providers.

## Motivation

The current kubernetes network design works well for connecting simple containers and building services, but it doesn’t work on many advanced use-cases, such as network isolation, overlapping IP addresses, etc.  To address these issues, this proposal adds "network providers" which enable multi-tenant networking capabilities for kubernetes.

Goals of this design:

* Support multiple networks with isolation between pods in those networks
* Optionally support overlapping IP addresses between isolated networks
* Support network information delivered by external entities through plugins
* Accommodates the current kubernetes single network model

## Use cases

Multi-tenant networking is extremely useful for following use cases:

* Network isolation between pod networks
* Multiple networks and/or IP addresses for Pods

## Proposed design

**Network model changes**

By default pods are not assigned to any network and the network provider is free to configure the pod how it wishes.  This is the current network model.

If the pod is assigned one or more networks, then the network provider is expected to set up the pod according to the network's NetworkSpec.  This may optionally include:

* Isolation: pods may or may not be isolated from each other depending on the configuration of each network assigned to the pod.  For example, the network provider may isolate on a per-network basis, ensuring that pods belonging to different networks cannot communicate with each other.  Or, the network provider could decide that networks A and B can communicate with each other, but only network B can communicate with network C and pods would be thus isolated.
* Overlapping IP addresses: the network provider for a given network controls the IP Address Management (IPAM) for the network, and may allow overlapping IP addresses between networks, or may allocate IP addresses for isolated pods from the same IP subnet and isolate through filtering or other means.
* Service IP addresses: these may or may not be handled by the network provider depending on its IP Address Management (IPAM) scheme
* kube-proxy: may be enabled or disabled by the network provider

**Implementation**

The Network Provider will expand on the existing NetworkPlugin with these additional integration points:

* Provides a list of NetworkSpecs which the Kubernetes core creates Networks with
* Notifies when new networks are added
* Notifies when an existing network is changed
* Notifies when an existing network is deleted
* Provides Init/Teardown methods that are called on startup/shutdown of the Kubernetes process

Since the Network Provider provided the networks to Kubernetes, it can clearly retrieve any additional data it needs (tenant ID, etc) required to actually configure the pod's networks when the pod is instantiated.  The Network Provider's pod setup hook is called once for each network assigned to the pod.

## API changes

New first-class objects will be added to support multi-tenant operation.

**A new `network` resource will be added**

```go
// NetworkStatus is information about the current status of a Network.
type NetworkStatus struct {
    // Phase is the current lifecycle phase of the network.
    Phase NetworkPhase `json:"phase,omitempty"`
}

type NetworkPhase string

// These are the valid phases of a network.
const (
    // NetworkInitializing means the network is just accepted by system
    NetworkInitializing NetworkPhase = "Initializing"
    // NetworkActive means the network is available for use in the system
    NetworkActive NetworkPhase = "Active"
    // NetworkPending means the network is accepted by system, but it is still
    // processing by network provider
    NetworkPending NetworkPhase = "Pending"
    // NetworkFailed means the network is not available
    NetworkFailed NetworkPhase = "Failed"
    // NetworkTerminating means the network is undergoing graceful termination
    NetworkTerminating NetworkPhase = "Terminating"
)

// Subnet is a description of a subnet
type Subnet struct {
    // CIDR of this subnet
    CIDR string `json:"cidr"`
    // Gateway of this subnet
    Gateway string `json:"gateway"`
}

// NetworkSpec is a description of a network
type NetworkSpec struct {
    // Which network provider/plugin provides this network and which will handle pod setup
    Provider string `json:"provider,omitempty"`

    // UUID specified by the network provider
    Uuid string `json:"uuid,omitempty"`

    // The network may contain zero or more subnets
    Subnets map[string]Subnet `json:"subnets,omitempty"`
}

// Network describes an instantiated network
type Network struct {
    unversioned.TypeMeta `json:",inline"`

    // Standard object's metadata.
    // More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
    ObjectMeta `json:"metadata,omitempty"`

    // Spec defines the behavior of the Network.
    Spec NetworkSpec `json:"spec,omitempty"`

    // Status describes the current status of a Network
    Status NetworkStatus `json:"status,omitempty"`
}

// NetworkList is a list of Networks
type NetworkList struct {
    unversioned.TypeMeta `json:",inline"`
    // Standard list metadata.
    // More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
    unversioned.ListMeta `json:"metadata,omitempty"`

    // Items is the list of Network objects in the list
    Items []Network `json:"items"`
}
```

**Network will be added to pod spec**

```go
// PodSpec is a description of a pod
type PodSpec struct {
  ...
  // Networks the pod is assigned to
  Networks []string `json:"networks,omitempty"`
  ...
}
```

## Examples

### Create a network with overlapping 

```yaml
apiVersion: v1
kind: Network
metadata:
  name: net1
spec:
  provider: mycorp/fancy-network-provider
  uuid: 2a9aafcc-6041-4fb8-9006-1f6095102850
  subnets:
    subnet1:
      cidr: 192.168.0.0/24
      gateway: 192.168.0.1
    subnet2:
      cidr: 192.168.1.0/24
      gateway: 192.168.1.1
```

## Community Discussion

* [kubernetes/3350](https://github.com/kubernetes/kubernetes/issues/3350)
* [kubernetes/13622](https://github.com/kubernetes/kubernetes/pull/13622)



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/multi-tenant-networking.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
