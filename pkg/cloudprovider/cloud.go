/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cloudprovider

import (
	"errors"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/controller"
)

// Interface is an abstract, pluggable interface for cloud providers.
type Interface interface {
	// Initialize provides the cloud with a kubernetes client builder and may spawn goroutines
	// to perform housekeeping activities within the cloud provider.
	Initialize(clientBuilder controller.ControllerClientBuilder)
	// LoadBalancer returns a balancer interface. Also returns true if the interface is supported, false otherwise.
	LoadBalancer() (LoadBalancer, bool)
	// Instances returns an instances interface. Also returns true if the interface is supported, false otherwise.
	Instances() (Instances, bool)
	// Zones returns a zones interface. Also returns true if the interface is supported, false otherwise.
	Zones() (Zones, bool)
	// Clusters returns a clusters interface.  Also returns true if the interface is supported, false otherwise.
	Clusters() (Clusters, bool)
	// Routes returns a routes interface along with whether the interface is supported.
	Routes() (Routes, bool)
	// ProviderName returns the cloud provider ID.
	ProviderName() string
	// ScrubDNS provides an opportunity for cloud-provider-specific code to process DNS settings for pods.
	ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string)
}

// Clusters is an abstract, pluggable interface for clusters of containers.
type Clusters interface {
	// ListClusters lists the names of the available clusters.
	ListClusters() ([]string, error)
	// Master gets back the address (either DNS name or IP address) of the master node for the cluster.
	Master(clusterName string) (string, error)
}

// TODO(#6812): Use a shorter name that's less likely to be longer than cloud
// providers' name length limits.
func GetLoadBalancerName(service *v1.Service) string {
	//GCE requires that the name of a load balancer starts with a lower case letter.
	ret := "a" + string(service.UID)
	ret = strings.Replace(ret, "-", "", -1)
	//AWS requires that the name of a load balancer is shorter than 32 bytes.
	if len(ret) > 32 {
		ret = ret[:32]
	}
	return ret
}

// GetInstanceProviderID builds a ProviderID for a node in a cloud.
func GetInstanceProviderID(cloud Interface, nodeName types.NodeName) (string, error) {
	instances, ok := cloud.Instances()
	if !ok {
		return "", fmt.Errorf("failed to get instances from cloud provider")
	}
	instanceID, err := instances.InstanceID(nodeName)
	if err != nil {
		return "", fmt.Errorf("failed to get instance ID from cloud provider: %v", err)
	}
	return cloud.ProviderName() + "://" + instanceID, nil
}

// LoadBalancer is an abstract, pluggable interface for load balancers.
type LoadBalancer interface {
	// TODO: Break this up into different interfaces (LB, etc) when we have more than one type of service
	// GetLoadBalancer returns whether the specified load balancer exists, and
	// if so, what its status is.
	// Implementations must treat the *v1.Service parameter as read-only and not modify it.
	// Parameter 'clusterName' is the name of the cluster as presented to kube-controller-manager
	GetLoadBalancer(clusterName string, service *v1.Service) (status *v1.LoadBalancerStatus, exists bool, err error)
	// EnsureLoadBalancer creates a new load balancer 'name', or updates the existing one. Returns the status of the balancer
	// Implementations must treat the *v1.Service and *v1.Node
	// parameters as read-only and not modify them.
	// Parameter 'clusterName' is the name of the cluster as presented to kube-controller-manager
	EnsureLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error)
	// UpdateLoadBalancer updates hosts under the specified load balancer.
	// Implementations must treat the *v1.Service and *v1.Node
	// parameters as read-only and not modify them.
	// Parameter 'clusterName' is the name of the cluster as presented to kube-controller-manager
	UpdateLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node) error
	// EnsureLoadBalancerDeleted deletes the specified load balancer if it
	// exists, returning nil if the load balancer specified either didn't exist or
	// was successfully deleted.
	// This construction is useful because many cloud providers' load balancers
	// have multiple underlying components, meaning a Get could say that the LB
	// doesn't exist even if some part of it is still laying around.
	// Implementations must treat the *v1.Service parameter as read-only and not modify it.
	// Parameter 'clusterName' is the name of the cluster as presented to kube-controller-manager
	EnsureLoadBalancerDeleted(clusterName string, service *v1.Service) error
}

// Instances is an abstract, pluggable interface for sets of instances.
type Instances interface {
	// NodeAddresses returns the addresses of the specified instance.
	// TODO(roberthbailey): This currently is only used in such a way that it
	// returns the address of the calling instance. We should do a rename to
	// make this clearer.
	NodeAddresses(name types.NodeName) ([]v1.NodeAddress, error)
	// NodeAddressesByProviderID returns the addresses of the specified instance.
	// The instance is specified using the providerID of the node. The
	// ProviderID is a unique identifier of the node. This will not be called
	// from the node whose nodeaddresses are being queried. i.e. local metadata
	// services cannot be used in this method to obtain nodeaddresses
	NodeAddressesByProviderID(providerId string) ([]v1.NodeAddress, error)
	// ExternalID returns the cloud provider ID of the node with the specified NodeName.
	// Note that if the instance does not exist or is no longer running, we must return ("", cloudprovider.InstanceNotFound)
	ExternalID(nodeName types.NodeName) (string, error)
	// InstanceID returns the cloud provider ID of the node with the specified NodeName.
	InstanceID(nodeName types.NodeName) (string, error)
	// InstanceType returns the type of the specified instance.
	InstanceType(name types.NodeName) (string, error)
	// InstanceTypeByProviderID returns the type of the specified instance.
	InstanceTypeByProviderID(providerID string) (string, error)
	// AddSSHKeyToAllInstances adds an SSH public key as a legal identity for all instances
	// expected format for the key is standard ssh-keygen format: <protocol> <blob>
	AddSSHKeyToAllInstances(user string, keyData []byte) error
	// CurrentNodeName returns the name of the node we are currently running on
	// On most clouds (e.g. GCE) this is the hostname, so we provide the hostname
	CurrentNodeName(hostname string) (types.NodeName, error)
}

// Route is a representation of an advanced routing rule.
type Route struct {
	// Name is the name of the routing rule in the cloud-provider.
	// It will be ignored in a Create (although nameHint may influence it)
	Name string
	// TargetNode is the NodeName of the target instance.
	TargetNode types.NodeName
	// DestinationCIDR is the CIDR format IP range that this routing rule
	// applies to.
	DestinationCIDR string
	// Blackhole is set to true if this is a blackhole route
	// The node controller will delete the route if it is in the managed range.
	Blackhole bool
}

// Routes is an abstract, pluggable interface for advanced routing rules.
type Routes interface {
	// ListRoutes lists all managed routes that belong to the specified clusterName
	ListRoutes(clusterName string) ([]*Route, error)
	// CreateRoute creates the described managed route
	// route.Name will be ignored, although the cloud-provider may use nameHint
	// to create a more user-meaningful name.
	CreateRoute(clusterName string, nameHint string, route *Route) error
	// DeleteRoute deletes the specified managed route
	// Route should be as returned by ListRoutes
	DeleteRoute(clusterName string, route *Route) error
}

var (
	InstanceNotFound = errors.New("instance not found")
	DiskNotFound     = errors.New("disk is not found")
)

// Zone represents the location of a particular machine.
type Zone struct {
	FailureDomain string
	Region        string
}

// Zones is an abstract, pluggable interface for zone enumeration.
type Zones interface {
	// GetZone returns the Zone containing the current failure zone and locality region that the program is running in
	GetZone() (Zone, error)
}
