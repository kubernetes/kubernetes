/*
Copyright 2018 The Kubernetes Authors.

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

package azure

import (
	"context"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestNodeAddresses(t *testing.T) {
	fakeVM := &fakeVMSet{}

	cloud := &Cloud{
		vmSet: fakeVM,
		Config: Config{
			ResourceGroup:  "foo",
			RouteTableName: "bar",
			Location:       "location",
		},
	}

	nodeIP := "2.4.6.8"
	fakeVM.NodeToIP = map[string]map[string]string{
		"": {
			"node": nodeIP,
		},
	}

	addr, err := cloud.NodeAddresses(context.TODO(), types.NodeName("node"))
	if err != nil {
		t.Errorf("unexpected error getting node addresses: %v", err)
		t.FailNow()
	}
	if len(addr) != 2 {
		t.Errorf("Expected 2 addresses, saw: %v", addr)
		t.FailNow()
	}

	if addr[0].Type != v1.NodeInternalIP || addr[0].Address != nodeIP {
		t.Errorf("Unexpected address at index 0: %#v", addr[0])
	}

	if addr[1].Type != v1.NodeHostName || addr[1].Address != "node" {
		t.Errorf("Unexpected address at index 1: %#v", addr[1])
	}
}

func TestNodeAddressesByProviderID(t *testing.T) {
	fakeVM := &fakeVMSet{}

	cloud := &Cloud{
		vmSet: fakeVM,
		Config: Config{
			ResourceGroup:  "foo",
			RouteTableName: "bar",
			Location:       "location",
		},
	}

	nodeIP := "2.4.6.8"
	nodeName := "node"
	providerID := "providerID"
	fakeVM.NodeToIP = map[string]map[string]string{
		"": {
			nodeName: nodeIP,
		},
	}
	fakeVM.NodeNames = map[string]string{
		providerID: nodeName,
	}

	addr, err := cloud.NodeAddressesByProviderID(context.TODO(), providerID)
	if err != nil {
		t.Errorf("unexpected error getting node addresses: %v", err)
		t.FailNow()
	}
	if len(addr) != 2 {
		t.Errorf("Expected 2 addresses, saw: %v", addr)
		t.FailNow()
	}

	if addr[0].Type != v1.NodeInternalIP || addr[0].Address != nodeIP {
		t.Errorf("Unexpected address at index 0: %#v", addr[0])
	}

	if addr[1].Type != v1.NodeHostName || addr[1].Address != "node" {
		t.Errorf("Unexpected address at index 1: %#v", addr[1])
	}
}

func TestInstanceExistsByProviderID(t *testing.T) {
	fakeVM := &fakeVMSet{}

	cloud := &Cloud{
		vmSet: fakeVM,
		Config: Config{
			ResourceGroup:  "foo",
			RouteTableName: "bar",
			Location:       "location",
		},
	}
	providerID := "providerID"
	nodeName := "node"
	instanceID := "instance"
	fakeVM.NodeNames = map[string]string{
		providerID: nodeName,
	}
	fakeVM.InstanceIDs = map[string]string{
		nodeName: instanceID,
	}

	exists, err := cloud.InstanceExistsByProviderID(context.TODO(), providerID)
	if !exists {
		t.Errorf("Expected node to exist for %s", providerID)
	}
	if err != nil {
		t.Errorf("Unexpected error for node exists: %v", err)
	}

	exists, err = cloud.InstanceExistsByProviderID(context.TODO(), "non-existent")
	if exists {
		t.Error("Expected node to not exist!")
	}
	if err == nil {
		t.Error("Unexpected non-error for node exists")
	}

	delete(fakeVM.InstanceIDs, nodeName)
	exists, err = cloud.InstanceExistsByProviderID(context.TODO(), providerID)
	if exists {
		t.Error("Expected node to not exist")
	}
	if err != nil {
		t.Errorf("Unexpected error for node exists: %v", err)
	}
}

func TestInstanceTypeByProviderID(t *testing.T) {
	fakeVM := &fakeVMSet{}

	cloud := &Cloud{
		vmSet: fakeVM,
		Config: Config{
			ResourceGroup:  "foo",
			RouteTableName: "bar",
			Location:       "location",
		},
	}
	providerID := "providerID"
	nodeName := "node"
	typeName := "type"
	fakeVM.NodeNames = map[string]string{
		providerID: nodeName,
	}
	fakeVM.InstanceTypes = map[string]string{
		nodeName: typeName,
	}

	typeOut, err := cloud.InstanceTypeByProviderID(context.TODO(), providerID)
	if typeName != typeOut {
		t.Errorf("Expected node type %s saw %s for %s", typeName, typeOut, providerID)
	}
	if err != nil {
		t.Errorf("Unexpected error for node exists: %v", err)
	}

	_, err = cloud.InstanceExistsByProviderID(context.TODO(), "non-existent")
	if err == nil {
		t.Error("Unexpected non-error for node exists")
	}
}
