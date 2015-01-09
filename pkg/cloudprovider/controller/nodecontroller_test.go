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

package controller

import (
	"fmt"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	fake_cloud "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
)

func newNode(name string) *api.Node {
	return &api.Node{ObjectMeta: api.ObjectMeta{Name: name}}
}

type FakeNodeHandler struct {
	client.Fake
	client.FakeNodes

	// Input: Hooks determine if request is valid or not
	CreateHook func(*FakeNodeHandler, *api.Node) bool
	Existing   []*api.Node

	// Output
	CreatedNodes []*api.Node
	DeletedNodes []*api.Node
	RequestCount int
}

func (c *FakeNodeHandler) Nodes() client.NodeInterface {
	return c
}

func (m *FakeNodeHandler) Create(node *api.Node) (*api.Node, error) {
	defer func() { m.RequestCount++ }()
	if m.CreateHook == nil || m.CreateHook(m, node) {
		m.CreatedNodes = append(m.CreatedNodes, node)
		return node, nil
	} else {
		return nil, fmt.Errorf("Create error.")
	}
}

func (m *FakeNodeHandler) List() (*api.NodeList, error) {
	defer func() { m.RequestCount++ }()
	nodes := []api.Node{}
	for i := 0; i < len(m.Existing); i++ {
		if !contains(m.Existing[i], m.DeletedNodes) {
			nodes = append(nodes, *m.Existing[i])
		}
	}
	for i := 0; i < len(m.CreatedNodes); i++ {
		if !contains(m.Existing[i], m.DeletedNodes) {
			nodes = append(nodes, *m.CreatedNodes[i])
		}
	}
	return &api.NodeList{Items: nodes}, nil
}

func (m *FakeNodeHandler) Delete(id string) error {
	m.DeletedNodes = append(m.DeletedNodes, newNode(id))
	m.RequestCount++
	return nil
}

func TestSyncStaticCreateNode(t *testing.T) {
	fakeNodeHandler := &FakeNodeHandler{
		CreateHook: func(fake *FakeNodeHandler, node *api.Node) bool {
			return true
		},
	}
	nodeController := NewNodeController(nil, ".*", []string{"node0"}, &api.NodeResources{}, fakeNodeHandler)
	if err := nodeController.SyncStatic(time.Millisecond); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fakeNodeHandler.RequestCount != 1 {
		t.Errorf("Expected 1 call, but got %v.", fakeNodeHandler.RequestCount)
	}
	if len(fakeNodeHandler.CreatedNodes) != 1 {
		t.Errorf("expect only 1 node created, got %v", len(fakeNodeHandler.CreatedNodes))
	}
	if fakeNodeHandler.CreatedNodes[0].Name != "node0" {
		t.Errorf("unexpect node %v created", fakeNodeHandler.CreatedNodes[0].Name)
	}
}

func TestSyncStaticCreateNodeWithHostIP(t *testing.T) {
	fakeNodeHandler := &FakeNodeHandler{
		CreateHook: func(fake *FakeNodeHandler, node *api.Node) bool {
			return true
		},
	}
	nodeController := NewNodeController(nil, ".*", []string{"10.0.0.1"}, &api.NodeResources{}, fakeNodeHandler)
	if err := nodeController.SyncStatic(time.Millisecond); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fakeNodeHandler.CreatedNodes[0].Name != "10.0.0.1" {
		t.Errorf("unexpect node %v created", fakeNodeHandler.CreatedNodes[0].Name)
	}
	if fakeNodeHandler.CreatedNodes[0].Status.HostIP != "10.0.0.1" {
		t.Errorf("unexpect nil node HostIP for node %v", fakeNodeHandler.CreatedNodes[0].Name)
	}
}

func TestSyncStaticCreateNodeWithError(t *testing.T) {
	fakeNodeHandler := &FakeNodeHandler{
		CreateHook: func(fake *FakeNodeHandler, node *api.Node) bool {
			if fake.RequestCount == 0 {
				return false
			}
			return true
		},
	}
	nodeController := NewNodeController(nil, ".*", []string{"node0"}, &api.NodeResources{}, fakeNodeHandler)
	if err := nodeController.SyncStatic(time.Millisecond); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fakeNodeHandler.RequestCount != 2 {
		t.Errorf("Expected 2 call, but got %v.", fakeNodeHandler.RequestCount)
	}
	if len(fakeNodeHandler.CreatedNodes) != 1 {
		t.Errorf("expect only 1 node created, got %v", len(fakeNodeHandler.CreatedNodes))
	}
	if fakeNodeHandler.CreatedNodes[0].Name != "node0" {
		t.Errorf("unexpect node %v created", fakeNodeHandler.CreatedNodes[0].Name)
	}
}

func TestSyncCloudCreateNode(t *testing.T) {
	fakeNodeHandler := &FakeNodeHandler{
		Existing: []*api.Node{newNode("node0")},
	}
	instances := []string{"node0", "node1"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	nodeController := NewNodeController(&fakeCloud, ".*", nil, nil, fakeNodeHandler)
	if err := nodeController.SyncCloud(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fakeNodeHandler.RequestCount != 2 {
		t.Errorf("Expected 2 call, but got %v.", fakeNodeHandler.RequestCount)
	}
	if len(fakeNodeHandler.CreatedNodes) != 1 {
		t.Errorf("expect only 1 node created, got %v", len(fakeNodeHandler.CreatedNodes))
	}
	if fakeNodeHandler.CreatedNodes[0].Name != "node1" {
		t.Errorf("unexpect node %v created", fakeNodeHandler.CreatedNodes[0].Name)
	}
}

func TestSyncCloudDeleteNode(t *testing.T) {
	fakeNodeHandler := &FakeNodeHandler{
		Existing: []*api.Node{newNode("node0"), newNode("node1")},
	}
	instances := []string{"node0"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	nodeController := NewNodeController(&fakeCloud, ".*", nil, nil, fakeNodeHandler)
	if err := nodeController.SyncCloud(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fakeNodeHandler.RequestCount != 2 {
		t.Errorf("Expected 2 call, but got %v.", fakeNodeHandler.RequestCount)
	}
	if len(fakeNodeHandler.DeletedNodes) != 1 {
		t.Errorf("expect only 1 node deleted, got %v", len(fakeNodeHandler.DeletedNodes))
	}
	if fakeNodeHandler.DeletedNodes[0].Name != "node1" {
		t.Errorf("unexpect node %v created", fakeNodeHandler.DeletedNodes[0].Name)
	}
}

func TestSyncCloudRegexp(t *testing.T) {
	fakeNodeHandler := &FakeNodeHandler{
		Existing: []*api.Node{newNode("node0")},
	}
	instances := []string{"node0", "node1", "fake"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	nodeController := NewNodeController(&fakeCloud, "node[0-9]+", nil, nil, fakeNodeHandler)
	if err := nodeController.SyncCloud(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fakeNodeHandler.RequestCount != 2 {
		t.Errorf("Expected 2 call, but got %v.", fakeNodeHandler.RequestCount)
	}
	if len(fakeNodeHandler.CreatedNodes) != 1 {
		t.Errorf("expect only 1 node created, got %v", len(fakeNodeHandler.CreatedNodes))
	}
	if fakeNodeHandler.CreatedNodes[0].Name != "node1" {
		t.Errorf("unexpect node %v created", fakeNodeHandler.CreatedNodes[0].Name)
	}
}

func contains(node *api.Node, nodes []*api.Node) bool {
	for i := 0; i < len(nodes); i++ {
		if node.Name == nodes[i].Name {
			return true
		}
	}
	return false
}
