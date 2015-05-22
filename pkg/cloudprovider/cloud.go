/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"net"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// Interface is an abstract, pluggable interface for cloud providers.
type Interface interface {
	// TCPLoadBalancer returns a balancer interface. Also returns true if the interface is supported, false otherwise.
	TCPLoadBalancer() (TCPLoadBalancer, bool)
	// Instances returns an instances interface. Also returns true if the interface is supported, false otherwise.
	Instances() (Instances, bool)
	// Zones returns a zones interface. Also returns true if the interface is supported, false otherwise.
	Zones() (Zones, bool)
	// Clusters returns a clusters interface.  Also returns true if the interface is supported, false otherwise.
	Clusters() (Clusters, bool)
	// Routes returns a routes interface along with whether the interface is supported.
	Routes() (Routes, bool)
}

// Clusters is an abstract, pluggable interface for clusters of containers.
type Clusters interface {
	// List lists the names of the available clusters.
	ListClusters() ([]string, error)
	// Master gets back the address (either DNS name or IP address) of the master node for the cluster.
	Master(clusterName string) (string, error)
}

// TODO(#6812): Use a shorter name that's less likely to be longer than cloud
// providers' name length limits.
func GetLoadBalancerName(service *api.Service) string {
	//GCE requires that the name of a load balancer starts with a lower case letter.
	ret := "a" + string(service.UID)
	ret = strings.Replace(ret, "-", "", -1)
	//AWS requires that the name of a load balancer is shorter than 32 bytes.
	if len(ret) > 32 {
		ret = ret[:32]
	}
	return ret
}

// TCPLoadBalancer is an abstract, pluggable interface for TCP load balancers.
type TCPLoadBalancer interface {
	// TODO: Break this up into different interfaces (LB, etc) when we have more than one type of service
	// GetTCPLoadBalancer returns whether the specified load balancer exists, and
	// if so, what its status is.
	GetTCPLoadBalancer(name, region string) (status *api.LoadBalancerStatus, exists bool, err error)
	// CreateTCPLoadBalancer creates a new tcp load balancer. Returns the status of the balancer
	CreateTCPLoadBalancer(name, region string, externalIP net.IP, ports []int, hosts []string, affinityType api.ServiceAffinity) (*api.LoadBalancerStatus, error)
	// UpdateTCPLoadBalancer updates hosts under the specified load balancer.
	UpdateTCPLoadBalancer(name, region string, hosts []string) error
	// EnsureTCPLoadBalancerDeleted deletes the specified load balancer if it
	// exists, returning nil if the load balancer specified either didn't exist or
	// was successfully deleted.
	// This construction is useful because many cloud providers' load balancers
	// have multiple underlying components, meaning a Get could say that the LB
	// doesn't exist even if some part of it is still laying around.
	EnsureTCPLoadBalancerDeleted(name, region string) error
}

// Instances is an abstract, pluggable interface for sets of instances.
type Instances interface {
	// NodeAddresses returns the addresses of the specified instance.
	NodeAddresses(name string) ([]api.NodeAddress, error)
	// ExternalID returns the cloud provider ID of the specified instance.
	ExternalID(name string) (string, error)
	// List lists instances that match 'filter' which is a regular expression which must match the entire instance name (fqdn)
	List(filter string) ([]string, error)
	// GetNodeResources gets the resources for a particular node
	GetNodeResources(name string) (*api.NodeResources, error)
}

// Route is a representation of an advanced routing rule.
type Route struct {
	// Name is the name of the routing rule in the cloud-provider.
	Name string
	// TargetInstance is the name of the instance as specified in routing rules
	// for the cloud-provider (in gce: the Instance Name).
	TargetInstance string
	// Destination CIDR is the CIDR format IP range that this routing rule
	// applies to.
	DestinationCIDR string
	// Description is a free-form string. It can be useful for tagging Routes.
	Description string
}

// Routes is an abstract, pluggable interface for advanced routing rules.
type Routes interface {
	// List all routes that match the filter
	ListRoutes(filter string) ([]*Route, error)
	// Create the described route
	CreateRoute(route *Route) error
	// Delete the specified route
	DeleteRoute(name string) error
}

var InstanceNotFound = errors.New("instance not found")

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
