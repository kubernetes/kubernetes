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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/proposals/multi-tenant-networking.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Multi-tenant networking

## Abstract

A proposal of multi-tenant networking by network providers.

## Motivation

The current kubernetes network design works great for connecting containers and building services, but it doesn’t work on many use-cases, such as container network isolation, IP address overlapping and so on. So this proposal adds network provider which adds multi-tenant networking for kubernetes.

Goals of this design:

* Support multi-tenant networking isolation between pods in those networks
* Support integration with SDN applications and external IPAMs by network provider
* Support [libnetwork](https://github.com/docker/libnetwork) integration
* Works together with kubelet network plugin

## Use cases

Multi-tenant networking is extremely useful for following use cases:

* Network isolation between pod networks
* Multiple IP addresses for Pods
* Manage kubernetes pod IPs by SDN applications
* Integration with current IPAM tools

## Proposed design

**Network model**

By default, namespaces and pods are not assigned to any networks. It is the current network model.

A new first-class `network` resource will be added to support multi-tenant networking.

If a namespace is assigned with a network, then the network provider is expected to setup all the pods and services within this namespace with the specified network. This network will be primary network for all pods of the namespace. Note that kube-proxy may be replaced with other services, which depends on the implementation of network provider.

If a pod is assigned with networks, then the network provider is expected to setup non-primary networks for the pod togather with network plugin.

To facilitate kubelet network plugin configuring pod's network properly, a new 'networks' param will be added to `NetworkPlugin` interface as described below.

**Implementation**

Network controller and network provider are introduced to manage kubernetes networks. Network controller listens to creation/deletion/update events for networks and services, and tell network provider to configure them.

## API changes

**A new first-class `network` resource will be added to support multi-tenant networking**

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

// Network describes a network
type Network struct {
    unversioned.TypeMeta `json:",inline"`

    // Standard object's metadata.
    // More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
    ObjectMeta `json:"metadata,omitempty"`

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

**Network will be added to namespace spec**

```go
// NamespaceSpec describes the attributes on a Namespace
type NamespaceSpec struct {
    // Finalizers is an opaque list of values that must be empty to permanently remove object from storage
    Finalizers []FinalizerName

    // DefaultNetwork descibes the default network of this namespace
    DefaultNetwork string `json:"defaultNetwork,omitempty"`
}
```

**ExtraNetworks will be added to pod spec**

```go
// PodSpec is description of a pod
type PodSpec struct {
  ...
  // Extra networks to receive non-default IP addresses
  ExtraNetworks []string `json:"extraNetworks,omitempty"`
  ...
}
```

**`networks` will be added to `NetworkPlugin` interface**

```go
// Plugin is an interface to network plugins for the kubelet
type NetworkPlugin interface {
	...
	// SetUpPod is the method called after the infra container of
	// the pod has been created but before the other containers of the
	// pod are launched.
	SetUpPod(namespace string, name string, podInfraContainerID kubetypes.DockerID, networks []string) error

	// TearDownPod is the method called before a pod's infra container will be deleted
	TearDownPod(namespace string, name string, podInfraContainerID kubetypes.DockerID, networks []string) error

	// Status is the method called to obtain the ipv4 or ipv6 addresses of the container
	Status(namespace string, name string, podInfraContainerID kubetypes.DockerID, networks []string) (*PodNetworkStatus, error)
}
```

## Examples

### Create a network

```yaml
apiVersion: v1
kind: Network
metadata:
  name: net1
```

### Create a namespace with network

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: testing
spec:
  network: net1
```

### Create a pod with an extra network

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: webserver
  namespace: testing
spec:
  extraNetworks:
  - net2
```

## Community Discussion

* [kubernetes/3350](https://github.com/kubernetes/kubernetes/issues/3350)
* [kubernetes/13622](https://github.com/kubernetes/kubernetes/pull/13622)



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/multi-tenant-networking.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
