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

package fake_cloud

import (
	"errors"
	"fmt"
	"net"
	"regexp"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const ProviderName = "fake"

// FakeBalancer is a fake storage of balancer information
type FakeBalancer struct {
	Name       string
	Region     string
	ExternalIP net.IP
	Ports      []*api.ServicePort
	Hosts      []string
}

type FakeUpdateBalancerCall struct {
	Name   string
	Region string
	Hosts  []string
}

// FakeCloud is a test-double implementation of Interface, TCPLoadBalancer, Instances, and Routes. It is useful for testing.
type FakeCloud struct {
	Exists        bool
	Err           error
	Calls         []string
	Addresses     []api.NodeAddress
	ExtID         map[string]string
	Machines      []string
	NodeResources *api.NodeResources
	ClusterList   []string
	MasterName    string
	ExternalIP    net.IP
	Balancers     map[string]FakeBalancer
	UpdateCalls   []FakeUpdateBalancerCall
	RouteMap      map[string]*FakeRoute
	Lock          sync.Mutex
	cloudprovider.Zone
}

type FakeRoute struct {
	ClusterName string
	Route       cloudprovider.Route
}

func (f *FakeCloud) addCall(desc string) {
	f.Calls = append(f.Calls, desc)
}

// ClearCalls clears internal record of method calls to this FakeCloud.
func (f *FakeCloud) ClearCalls() {
	f.Calls = []string{}
}

func (f *FakeCloud) ListClusters() ([]string, error) {
	return f.ClusterList, f.Err
}

func (f *FakeCloud) Master(name string) (string, error) {
	return f.MasterName, f.Err
}

func (f *FakeCloud) Clusters() (cloudprovider.Clusters, bool) {
	return f, true
}

// ProviderName returns the cloud provider ID.
func (f *FakeCloud) ProviderName() string {
	return ProviderName
}

// TCPLoadBalancer returns a fake implementation of TCPLoadBalancer.
// Actually it just returns f itself.
func (f *FakeCloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return f, true
}

// Instances returns a fake implementation of Instances.
//
// Actually it just returns f itself.
func (f *FakeCloud) Instances() (cloudprovider.Instances, bool) {
	return f, true
}

func (f *FakeCloud) Zones() (cloudprovider.Zones, bool) {
	return f, true
}

func (f *FakeCloud) Routes() (cloudprovider.Routes, bool) {
	return f, true
}

// GetTCPLoadBalancer is a stub implementation of TCPLoadBalancer.GetTCPLoadBalancer.
func (f *FakeCloud) GetTCPLoadBalancer(name, region string) (*api.LoadBalancerStatus, bool, error) {
	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: f.ExternalIP.String()}}

	return status, f.Exists, f.Err
}

// EnsureTCPLoadBalancer is a test-spy implementation of TCPLoadBalancer.EnsureTCPLoadBalancer.
// It adds an entry "create" into the internal method call record.
func (f *FakeCloud) EnsureTCPLoadBalancer(name, region string, externalIP net.IP, ports []*api.ServicePort, hosts []string, affinityType api.ServiceAffinity) (*api.LoadBalancerStatus, error) {
	f.addCall("create")
	if f.Balancers == nil {
		f.Balancers = make(map[string]FakeBalancer)
	}
	f.Balancers[name] = FakeBalancer{name, region, externalIP, ports, hosts}

	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: f.ExternalIP.String()}}

	return status, f.Err
}

// UpdateTCPLoadBalancer is a test-spy implementation of TCPLoadBalancer.UpdateTCPLoadBalancer.
// It adds an entry "update" into the internal method call record.
func (f *FakeCloud) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
	f.addCall("update")
	f.UpdateCalls = append(f.UpdateCalls, FakeUpdateBalancerCall{name, region, hosts})
	return f.Err
}

// EnsureTCPLoadBalancerDeleted is a test-spy implementation of TCPLoadBalancer.EnsureTCPLoadBalancerDeleted.
// It adds an entry "delete" into the internal method call record.
func (f *FakeCloud) EnsureTCPLoadBalancerDeleted(name, region string) error {
	f.addCall("delete")
	return f.Err
}

func (f *FakeCloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

// Implementation of Instances.CurrentNodeName
func (f *FakeCloud) CurrentNodeName(hostname string) (string, error) {
	return hostname, nil
}

// NodeAddresses is a test-spy implementation of Instances.NodeAddresses.
// It adds an entry "node-addresses" into the internal method call record.
func (f *FakeCloud) NodeAddresses(instance string) ([]api.NodeAddress, error) {
	f.addCall("node-addresses")
	return f.Addresses, f.Err
}

// ExternalID is a test-spy implementation of Instances.ExternalID.
// It adds an entry "external-id" into the internal method call record.
// It returns an external id to the mapped instance name, if not found, it will return "ext-{instance}"
func (f *FakeCloud) ExternalID(instance string) (string, error) {
	f.addCall("external-id")
	return f.ExtID[instance], f.Err
}

// InstanceID returns the cloud provider ID of the specified instance.
func (f *FakeCloud) InstanceID(instance string) (string, error) {
	f.addCall("instance-id")
	return f.ExtID[instance], nil
}

// List is a test-spy implementation of Instances.List.
// It adds an entry "list" into the internal method call record.
func (f *FakeCloud) List(filter string) ([]string, error) {
	f.addCall("list")
	result := []string{}
	for _, machine := range f.Machines {
		if match, _ := regexp.MatchString(filter, machine); match {
			result = append(result, machine)
		}
	}
	return result, f.Err
}

func (f *FakeCloud) GetZone() (cloudprovider.Zone, error) {
	f.addCall("get-zone")
	return f.Zone, f.Err
}

func (f *FakeCloud) ListRoutes(clusterName string) ([]*cloudprovider.Route, error) {
	f.Lock.Lock()
	defer f.Lock.Unlock()
	f.addCall("list-routes")
	var routes []*cloudprovider.Route
	for _, fakeRoute := range f.RouteMap {
		if clusterName == fakeRoute.ClusterName {
			routeCopy := fakeRoute.Route
			routes = append(routes, &routeCopy)
		}
	}
	return routes, f.Err
}

func (f *FakeCloud) CreateRoute(clusterName string, nameHint string, route *cloudprovider.Route) error {
	f.Lock.Lock()
	defer f.Lock.Unlock()
	f.addCall("create-route")
	name := clusterName + "-" + nameHint
	if _, exists := f.RouteMap[name]; exists {
		f.Err = fmt.Errorf("route %q already exists", name)
		return f.Err
	}
	fakeRoute := FakeRoute{}
	fakeRoute.Route = *route
	fakeRoute.Route.Name = name
	fakeRoute.ClusterName = clusterName
	f.RouteMap[name] = &fakeRoute
	return nil
}

func (f *FakeCloud) DeleteRoute(clusterName string, route *cloudprovider.Route) error {
	f.Lock.Lock()
	defer f.Lock.Unlock()
	f.addCall("delete-route")
	name := route.Name
	if _, exists := f.RouteMap[name]; !exists {
		f.Err = fmt.Errorf("no route found with name %q", name)
		return f.Err
	}
	delete(f.RouteMap, name)
	return nil
}
