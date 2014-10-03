/*
Copyright 2014 Google Inc. All rights reserved.

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

// FakeCloud is a test-double implementation of Interface, TCPLoadBalancer and Instances. It is useful for testing.
type FakeCloud struct {
	Exists        bool
	Err           error
	Calls         []string
	IP            net.IP
	Machines      []string
	NodeResources *api.NodeResources

	cloudprovider.Zone
}

func (f *FakeCloud) addCall(desc string) {
	f.Calls = append(f.Calls, desc)
}

// ClearCalls clears internal record of method calls to this FakeCloud.
func (f *FakeCloud) ClearCalls() {
	f.Calls = []string{}
}

// TCPLoadBalancer returns a fake implementation of TCPLoadBalancer.
//
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

// TCPLoadBalancerExists is a stub implementation of TCPLoadBalancer.TCPLoadBalancerExists.
func (f *FakeCloud) TCPLoadBalancerExists(name, region string) (bool, error) {
	return f.Exists, f.Err
}

// CreateTCPLoadBalancer is a test-spy implementation of TCPLoadBalancer.CreateTCPLoadBalancer.
// It adds an entry "create" into the internal method call record.
func (f *FakeCloud) CreateTCPLoadBalancer(name, region string, port int, hosts []string) error {
	f.addCall("create")
	return f.Err
}

// UpdateTCPLoadBalancer is a test-spy implementation of TCPLoadBalancer.UpdateTCPLoadBalancer.
// It adds an entry "update" into the internal method call record.
func (f *FakeCloud) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
	f.addCall("update")
	return f.Err
}

// DeleteTCPLoadBalancer is a test-spy implementation of TCPLoadBalancer.DeleteTCPLoadBalancer.
// It adds an entry "delete" into the internal method call record.
func (f *FakeCloud) DeleteTCPLoadBalancer(name, region string) error {
	f.addCall("delete")
	return f.Err
}

// IPAddress is a test-spy implementation of Instances.IPAddress.
// It adds an entry "ip-address" into the internal method call record.
func (f *FakeCloud) IPAddress(instance string) (net.IP, error) {
	f.addCall("ip-address")
	return f.IP, f.Err
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
