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
	"net"
	"regexp"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
)

// FakeBalancer is a fake storage of balancer information
type FakeBalancer struct {
	Name       string
	Region     string
	ExternalIP net.IP
	Ports      []int
	Hosts      []string
}

type FakeUpdateBalancerCall struct {
	Name   string
	Region string
	Hosts  []string
}

// FakeCloud is a test-double implementation of Interface, TCPLoadBalancer and Instances. It is useful for testing.
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
	Balancers     []FakeBalancer
	UpdateCalls   []FakeUpdateBalancerCall
	cloudprovider.Zone
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

// GetTCPLoadBalancer is a stub implementation of TCPLoadBalancer.GetTCPLoadBalancer.
func (f *FakeCloud) GetTCPLoadBalancer(name, region string) (endpoint string, exists bool, err error) {
	return f.ExternalIP.String(), f.Exists, f.Err
}

// CreateTCPLoadBalancer is a test-spy implementation of TCPLoadBalancer.CreateTCPLoadBalancer.
// It adds an entry "create" into the internal method call record.
func (f *FakeCloud) CreateTCPLoadBalancer(name, region string, externalIP net.IP, ports []int, hosts []string, affinityType api.AffinityType) (string, error) {
	f.addCall("create")
	f.Balancers = append(f.Balancers, FakeBalancer{name, region, externalIP, ports, hosts})
	return f.ExternalIP.String(), f.Err
}

// UpdateTCPLoadBalancer is a test-spy implementation of TCPLoadBalancer.UpdateTCPLoadBalancer.
// It adds an entry "update" into the internal method call record.
func (f *FakeCloud) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
	f.addCall("update")
	f.UpdateCalls = append(f.UpdateCalls, FakeUpdateBalancerCall{name, region, hosts})
	return f.Err
}

// DeleteTCPLoadBalancer is a test-spy implementation of TCPLoadBalancer.DeleteTCPLoadBalancer.
// It adds an entry "delete" into the internal method call record.
func (f *FakeCloud) DeleteTCPLoadBalancer(name, region string) error {
	f.addCall("delete")
	return f.Err
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

func (f *FakeCloud) GetNodeResources(name string) (*api.NodeResources, error) {
	f.addCall("get-node-resources")
	return f.NodeResources, f.Err
}

func (f *FakeCloud) Configure(name string, spec *api.NodeSpec) error {
	f.addCall("configure")
	return f.Err
}

func (f *FakeCloud) Release(name string) error {
	f.addCall("release")
	return f.Err
}
