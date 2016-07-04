/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package concerto_cloud

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

func TestAddSSHKeyToAllInstancesNotSupported(t *testing.T) {
	concerto := &ConcertoCloud{}
	err := concerto.AddSSHKeyToAllInstances("user", nil)
	if err == nil {
		t.Errorf("Expected 'AddSSHKeyToAllInstances' to not be supported")
	}
}

func TestNodeAddresses(t *testing.T) {
	concerto := &ConcertoCloud{
		service: &ConcertoAPIServiceMock{
			instances: []ConcertoInstance{
				{
					Name:     "myinstance",
					Id:       "11235813",
					PublicIP: "123.123.123.123"}}}}
	na, err := concerto.NodeAddresses("myinstance") // ([]api.NodeAddress, error)
	if err != nil {
		t.Errorf("NodeAddresses: Should have found instance 'myinstance'")
	}
	if len(na) != 1 {
		t.Errorf("NodeAddresses: Should have found just one address")
		t.FailNow()
	}
	// api.NodeAddress = {Type: api.NodeExternalIP, Address: externalIP}
	// ip := net.ParseIP(ipAddress)
	if na[0].Type != api.NodeExternalIP {
		t.Errorf("NodeAddresses: Should have found an external address")
	}
	if na[0].Address != "123.123.123.123" {
		t.Errorf("NodeAddresses: Should have found the correct address")
	}
}

func TestNodeAddresses_InstanceNotFound(t *testing.T) {
	concerto := &ConcertoCloud{
		service: &ConcertoAPIServiceMock{
			instances: []ConcertoInstance{
				{
					Name:     "someinstance",
					Id:       "11235813",
					PublicIP: "123.123.123.123"}}}}
	na, err := concerto.NodeAddresses("myinstance") // ([]api.NodeAddress, error)
	if err == nil {
		t.Errorf("NodeAddresses: Should have returned error")
	}
	if len(na) != 0 {
		t.Errorf("NodeAddresses: Should have not found any address")
		t.FailNow()
	}
}

func TestInstanceID(t *testing.T) {
	concerto := &ConcertoCloud{
		service: &ConcertoAPIServiceMock{
			instances: []ConcertoInstance{
				{
					Name:     "myinstance",
					Id:       "11235813",
					PublicIP: "123.123.123.123"}}}}
	id, err := concerto.InstanceID("myinstance")
	if err != nil {
		t.Errorf("InstanceID: Should have found instance 'myinstance'")
	}
	if id != "11235813" {
		t.Errorf("InstanceID: Should have returned the correct ID")
	}
}

func TestInstanceID_InstanceNotFound(t *testing.T) {
	concerto := &ConcertoCloud{
		service: &ConcertoAPIServiceMock{
			instances: []ConcertoInstance{
				{
					Name:     "someinstance",
					Id:       "11235813",
					PublicIP: "123.123.123.123"}}}}
	id, err := concerto.InstanceID("myinstance")
	if err != cloudprovider.InstanceNotFound {
		t.Errorf("InstanceID: Should have returned error")
	}
	if id != "" {
		t.Errorf("InstanceID: Should not have returned any ID")
	}
}

func TestExternalID(t *testing.T) {
	concerto := &ConcertoCloud{
		service: &ConcertoAPIServiceMock{
			instances: []ConcertoInstance{
				{
					Name:     "myinstance",
					Id:       "11235813",
					PublicIP: "123.123.123.123"}}}}
	id, err := concerto.ExternalID("myinstance")
	if err != nil {
		t.Errorf("ExternalID: Should have found instance 'myinstance'")
	}
	if id != "11235813" {
		t.Errorf("ExternalID: Should have returned the correct ID")
	}
}

func TestExternalID_InstanceNotFound(t *testing.T) {
	concerto := &ConcertoCloud{
		service: &ConcertoAPIServiceMock{
			instances: []ConcertoInstance{
				{
					Name:     "someinstance",
					Id:       "11235813",
					PublicIP: "123.123.123.123"}}}}
	id, err := concerto.ExternalID("myinstance")
	if err != cloudprovider.InstanceNotFound {
		t.Errorf("ExternalID: Should have returned error")
	}
	if id != "" {
		t.Errorf("ExternalID: Should not have returned any ID")
	}
}

func TestList(t *testing.T) {
	concerto := &ConcertoCloud{
		service: &ConcertoAPIServiceMock{
			instances: []ConcertoInstance{
				{
					Name:     "myinstance.acme.concerto.io",
					Id:       "11235813",
					PublicIP: "123.123.123.123",
					CPUs:     2,
					Memory:   4},
				{
					Name:     "otherinstance.acme.concerto.io",
					Id:       "9876543",
					PublicIP: "1.1.1.1",
					CPUs:     2,
					Memory:   4},
			}}}
	list, err := concerto.List(".*my.*")
	if err != nil {
		t.Errorf("List: Should not have returned error: %v", err)
	}
	if len(list) != 1 {
		t.Errorf("List: Should have found exactly one instance")
		t.FailNow()
	}
	if list[0] != "myinstance.acme.concerto.io" {
		t.Errorf("List: Should have found instance 'myinstance'")
	}
}

func TestGetNodeResources(t *testing.T) {
	concerto := &ConcertoCloud{
		service: &ConcertoAPIServiceMock{
			instances: []ConcertoInstance{
				{
					Name:     "myinstance",
					Id:       "11235813",
					PublicIP: "123.123.123.123",
					CPUs:     2,
					Memory:   4 * 1024,
				}}}}
	nr, err := concerto.GetNodeResources("myinstance") // (*api.NodeResources, error)
	if err != nil {
		t.Errorf("GetNodeResources: Should have found instance 'myinstance'")
	}
	aux := nr.Capacity[api.ResourceCPU]
	returnedCPUs := (&aux).Value()
	expectedCPUs := int64(2)
	if returnedCPUs != expectedCPUs {
		t.Errorf("GetNodeResources: Should have returned the correct CPU resources")
		t.Errorf("GetNodeResources: Expected %v but was %v", expectedCPUs, returnedCPUs)
	}
	aux = nr.Capacity[api.ResourceMemory]
	returnedMemory := (&aux).Value()
	expectedMemory := int64(4 * 1024 * 1024 * 1024)
	if returnedMemory != expectedMemory {
		t.Errorf("GetNodeResources: Should have returned the correct Memory resources")
		t.Errorf("GetNodeResources: Expected %v but was %v", expectedMemory, returnedMemory)
	}
}

func TestGetNodeResources_InstanceNotFound(t *testing.T) {
	concerto := &ConcertoCloud{
		service: &ConcertoAPIServiceMock{
			instances: []ConcertoInstance{
				{
					Name:     "someinstance",
					Id:       "11235813",
					PublicIP: "123.123.123.123",
					CPUs:     2,
					Memory:   4,
				}}}}
	nr, err := concerto.GetNodeResources("myinstance")
	if err == nil {
		t.Errorf("GetNodeResources: Should have returned error")
	}
	if nr != nil {
		t.Errorf("GetNodeResources: Should not have returned any resources")
	}
}

func TestCurrentNodeName(t *testing.T) {
	concerto := &ConcertoCloud{}
	nodename, err := concerto.CurrentNodeName("something")
	if err != nil {
		t.Errorf("GetNodeName: Should not have returned error: %v", err)
		t.FailNow()
	}
	if nodename != "something" {
		t.Errorf("GetNodeName: Should have returned correct name")
	}
}
