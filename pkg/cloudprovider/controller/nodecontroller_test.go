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
	"errors"
	"net"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	fake_cloud "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// FakeNodeHandler is a fake implementation of NodesInterface and NodeInterface.
type FakeNodeHandler struct {
	client.Fake
	client.FakeNodes

	// Input: Hooks determine if request is valid or not
	CreateHook func(*FakeNodeHandler, *api.Node) bool
	Existing   []*api.Node

	// Output
	CreatedNodes []*api.Node
	DeletedNodes []*api.Node
	UpdatedNodes []*api.Node
	RequestCount int
}

func (c *FakeNodeHandler) Nodes() client.NodeInterface {
	return c
}

func (m *FakeNodeHandler) Create(node *api.Node) (*api.Node, error) {
	defer func() { m.RequestCount++ }()
	for _, n := range m.Existing {
		if n.Name == node.Name {
			return nil, apierrors.NewAlreadyExists("Minion", node.Name)
		}
	}
	if m.CreateHook == nil || m.CreateHook(m, node) {
		nodeCopy := *node
		m.CreatedNodes = append(m.CreatedNodes, &nodeCopy)
		return node, nil
	} else {
		return nil, errors.New("Create error.")
	}
}

func (m *FakeNodeHandler) List() (*api.NodeList, error) {
	defer func() { m.RequestCount++ }()
	var nodes []*api.Node
	for i := 0; i < len(m.UpdatedNodes); i++ {
		if !contains(m.UpdatedNodes[i], m.DeletedNodes) {
			nodes = append(nodes, m.UpdatedNodes[i])
		}
	}
	for i := 0; i < len(m.Existing); i++ {
		if !contains(m.Existing[i], m.DeletedNodes) && !contains(m.Existing[i], nodes) {
			nodes = append(nodes, m.Existing[i])
		}
	}
	for i := 0; i < len(m.CreatedNodes); i++ {
		if !contains(m.Existing[i], m.DeletedNodes) && !contains(m.CreatedNodes[i], nodes) {
			nodes = append(nodes, m.CreatedNodes[i])
		}
	}
	nodeList := &api.NodeList{}
	for _, node := range nodes {
		nodeList.Items = append(nodeList.Items, *node)
	}
	return nodeList, nil
}

func (m *FakeNodeHandler) Delete(id string) error {
	m.DeletedNodes = append(m.DeletedNodes, newNode(id))
	m.RequestCount++
	return nil
}

func (m *FakeNodeHandler) Update(node *api.Node) (*api.Node, error) {
	nodeCopy := *node
	m.UpdatedNodes = append(m.UpdatedNodes, &nodeCopy)
	m.RequestCount++
	return node, nil
}

// FakeKubeletClient is a fake implementation of KubeletClient.
type FakeKubeletClient struct {
	Status probe.Result
	Err    error
}

func (c *FakeKubeletClient) GetPodStatus(host, podNamespace, podID string) (api.PodStatusResult, error) {
	return api.PodStatusResult{}, errors.New("Not Implemented")
}

func (c *FakeKubeletClient) HealthCheck(host string) (probe.Result, error) {
	return c.Status, c.Err
}

func TestRegisterNodes(t *testing.T) {
	table := []struct {
		fakeNodeHandler      *FakeNodeHandler
		machines             []string
		retryCount           int
		expectedRequestCount int
		expectedCreateCount  int
		expectedFail         bool
	}{
		{
			// Register two nodes normally.
			machines: []string{"node0", "node1"},
			fakeNodeHandler: &FakeNodeHandler{
				CreateHook: func(fake *FakeNodeHandler, node *api.Node) bool { return true },
			},
			retryCount:           1,
			expectedRequestCount: 2,
			expectedCreateCount:  2,
			expectedFail:         false,
		},
		{
			// Canonicalize node names.
			machines: []string{"NODE0", "node1"},
			fakeNodeHandler: &FakeNodeHandler{
				CreateHook: func(fake *FakeNodeHandler, node *api.Node) bool {
					if node.Name == "NODE0" {
						return false
					}
					return true
				},
			},
			retryCount:           1,
			expectedRequestCount: 2,
			expectedCreateCount:  2,
			expectedFail:         false,
		},
		{
			// No machine to register.
			machines: []string{},
			fakeNodeHandler: &FakeNodeHandler{
				CreateHook: func(fake *FakeNodeHandler, node *api.Node) bool { return true },
			},
			retryCount:           1,
			expectedRequestCount: 0,
			expectedCreateCount:  0,
			expectedFail:         false,
		},
		{
			// Fail the first two requests.
			machines: []string{"node0", "node1"},
			fakeNodeHandler: &FakeNodeHandler{
				CreateHook: func(fake *FakeNodeHandler, node *api.Node) bool {
					if fake.RequestCount == 0 || fake.RequestCount == 1 {
						return false
					}
					return true
				},
			},
			retryCount:           10,
			expectedRequestCount: 4,
			expectedCreateCount:  2,
			expectedFail:         false,
		},
		{
			// One node already exists
			machines: []string{"node0", "node1"},
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "node1",
						},
					},
				},
			},
			retryCount:           10,
			expectedRequestCount: 2,
			expectedCreateCount:  1,
			expectedFail:         false,
		},
		{
			// The first node always fails.
			machines: []string{"node0", "node1"},
			fakeNodeHandler: &FakeNodeHandler{
				CreateHook: func(fake *FakeNodeHandler, node *api.Node) bool {
					if node.Name == "node0" {
						return false
					}
					return true
				},
			},
			retryCount:           2,
			expectedRequestCount: 3, // 2 for node0, 1 for node1
			expectedCreateCount:  1,
			expectedFail:         true,
		},
	}

	for _, item := range table {
		nodes := api.NodeList{}
		for _, machine := range item.machines {
			nodes.Items = append(nodes.Items, *newNode(machine))
		}
		nodeController := NewNodeController(nil, "", item.machines, &api.NodeResources{}, item.fakeNodeHandler, nil)
		err := nodeController.RegisterNodes(&nodes, item.retryCount, time.Millisecond)
		if !item.expectedFail && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.expectedFail && err == nil {
			t.Errorf("unexpected non-error")
		}
		if item.fakeNodeHandler.RequestCount != item.expectedRequestCount {
			t.Errorf("expected %v calls, but got %v.", item.expectedRequestCount, item.fakeNodeHandler.RequestCount)
		}
		if len(item.fakeNodeHandler.CreatedNodes) != item.expectedCreateCount {
			t.Errorf("expected %v nodes, but got %v.", item.expectedCreateCount, item.fakeNodeHandler.CreatedNodes)
		}
	}
}

func TestCreateStaticNodes(t *testing.T) {
	table := []struct {
		machines      []string
		expectedNodes *api.NodeList
	}{
		{
			machines:      []string{},
			expectedNodes: &api.NodeList{},
		},
		{
			machines: []string{"node0"},
			expectedNodes: &api.NodeList{
				Items: []api.Node{
					{
						ObjectMeta: api.ObjectMeta{Name: "node0"},
						Spec:       api.NodeSpec{},
						Status:     api.NodeStatus{},
					},
				},
			},
		},
	}

	for _, item := range table {
		nodeController := NewNodeController(nil, "", item.machines, &api.NodeResources{}, nil, nil)
		nodes, err := nodeController.StaticNodes()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(item.expectedNodes, nodes) {
			t.Errorf("expected node list %+v, got %+v", item.expectedNodes, nodes)
		}
	}
}

func TestCreateCloudNodes(t *testing.T) {
	resourceList := api.ResourceList{
		api.ResourceCPU:    *resource.NewMilliQuantity(1000, resource.DecimalSI),
		api.ResourceMemory: *resource.NewQuantity(3000, resource.DecimalSI),
	}

	table := []struct {
		fakeCloud     *fake_cloud.FakeCloud
		machines      []string
		expectedNodes *api.NodeList
	}{
		{
			fakeCloud:     &fake_cloud.FakeCloud{},
			expectedNodes: &api.NodeList{},
		},
		{
			fakeCloud: &fake_cloud.FakeCloud{
				Machines:      []string{"node0"},
				NodeResources: &api.NodeResources{Capacity: resourceList},
			},
			expectedNodes: &api.NodeList{
				Items: []api.Node{
					{
						ObjectMeta: api.ObjectMeta{Name: "node0"},
						Spec:       api.NodeSpec{Capacity: resourceList},
					},
				},
			},
		},
	}

	for _, item := range table {
		nodeController := NewNodeController(item.fakeCloud, ".*", nil, &api.NodeResources{}, nil, nil)
		nodes, err := nodeController.CloudNodes()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(item.expectedNodes, nodes) {
			t.Errorf("expected node list %+v, got %+v", item.expectedNodes, nodes)
		}
	}
}

func TestSyncCloud(t *testing.T) {
	table := []struct {
		fakeNodeHandler      *FakeNodeHandler
		fakeCloud            *fake_cloud.FakeCloud
		matchRE              string
		expectedRequestCount int
		expectedCreated      []string
		expectedDeleted      []string
	}{
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{newNode("node0")},
			},
			fakeCloud: &fake_cloud.FakeCloud{
				Machines: []string{"node0", "node1"},
			},
			matchRE:              ".*",
			expectedRequestCount: 2, // List + Create
			expectedCreated:      []string{"node1"},
			expectedDeleted:      []string{},
		},
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{newNode("node0"), newNode("node1")},
			},
			fakeCloud: &fake_cloud.FakeCloud{
				Machines: []string{"node0"},
			},
			matchRE:              ".*",
			expectedRequestCount: 2, // List + Delete
			expectedCreated:      []string{},
			expectedDeleted:      []string{"node1"},
		},
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{newNode("node0")},
			},
			fakeCloud: &fake_cloud.FakeCloud{
				Machines: []string{"node0", "node1", "fake"},
			},
			matchRE:              "node[0-9]+",
			expectedRequestCount: 2, // List + Create
			expectedCreated:      []string{"node1"},
			expectedDeleted:      []string{},
		},
	}

	for _, item := range table {
		nodeController := NewNodeController(item.fakeCloud, item.matchRE, nil, &api.NodeResources{}, item.fakeNodeHandler, nil)
		if err := nodeController.SyncCloud(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.fakeNodeHandler.RequestCount != item.expectedRequestCount {
			t.Errorf("expected %v call, but got %v.", item.expectedRequestCount, item.fakeNodeHandler.RequestCount)
		}
		nodes := sortedNodeNames(item.fakeNodeHandler.CreatedNodes)
		if !reflect.DeepEqual(item.expectedCreated, nodes) {
			t.Errorf("expected node list %+v, got %+v", item.expectedCreated, nodes)
		}
		nodes = sortedNodeNames(item.fakeNodeHandler.DeletedNodes)
		if !reflect.DeepEqual(item.expectedDeleted, nodes) {
			t.Errorf("expected node list %+v, got %+v", item.expectedDeleted, nodes)
		}
	}
}

func TestHealthCheckNode(t *testing.T) {
	table := []struct {
		node               *api.Node
		fakeKubeletClient  *FakeKubeletClient
		expectedConditions []api.NodeCondition
	}{
		{
			node: newNode("node0"),
			fakeKubeletClient: &FakeKubeletClient{
				Status: probe.Success,
				Err:    nil,
			},
			expectedConditions: []api.NodeCondition{
				{
					Kind:   api.NodeReady,
					Status: api.ConditionFull,
					Reason: "Node health check succeeded: kubelet /healthz endpoint returns ok",
				},
			},
		},
		{
			node: newNode("node0"),
			fakeKubeletClient: &FakeKubeletClient{
				Status: probe.Failure,
				Err:    nil,
			},
			expectedConditions: []api.NodeCondition{
				{
					Kind:   api.NodeReady,
					Status: api.ConditionNone,
					Reason: "Node health check failed: kubelet /healthz endpoint returns not ok",
				},
			},
		},
		{
			node: newNode("node1"),
			fakeKubeletClient: &FakeKubeletClient{
				Status: probe.Failure,
				Err:    errors.New("Error"),
			},
			expectedConditions: []api.NodeCondition{
				{
					Kind:   api.NodeReady,
					Status: api.ConditionUnknown,
					Reason: "Node health check error: Error",
				},
			},
		},
	}

	for _, item := range table {
		nodeController := NewNodeController(nil, "", nil, nil, nil, item.fakeKubeletClient)
		conditions := nodeController.DoCheck(item.node)
		for i := range conditions {
			if conditions[i].LastTransitionTime.IsZero() {
				t.Errorf("unexpected zero timestamp")
			}
			conditions[i].LastTransitionTime = util.Time{}
		}
		if !reflect.DeepEqual(item.expectedConditions, conditions) {
			t.Errorf("expected conditions %+v, got %+v", item.expectedConditions, conditions)
		}
	}
}

func TestPopulateNodeIPs(t *testing.T) {
	table := []struct {
		nodes        *api.NodeList
		fakeCloud    *fake_cloud.FakeCloud
		expectedFail bool
		expectedIP   string
	}{
		{
			nodes:      &api.NodeList{Items: []api.Node{*newNode("node0"), *newNode("node1")}},
			fakeCloud:  &fake_cloud.FakeCloud{IP: net.ParseIP("1.2.3.4")},
			expectedIP: "1.2.3.4",
		},
		{
			nodes:      &api.NodeList{Items: []api.Node{*newNode("node0"), *newNode("node1")}},
			fakeCloud:  &fake_cloud.FakeCloud{Err: ErrQueryIPAddress},
			expectedIP: "",
		},
	}

	for _, item := range table {
		nodeController := NewNodeController(item.fakeCloud, ".*", nil, nil, nil, nil)
		result, err := nodeController.PopulateIPs(item.nodes)
		// In case of IP querying error, we should continue.
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		for _, node := range result.Items {
			if node.Status.HostIP != item.expectedIP {
				t.Errorf("expect HostIP %s, got %s", item.expectedIP, node.Status.HostIP)
			}
		}
	}
}

func TestNodeStatusTransitionTime(t *testing.T) {
	table := []struct {
		fakeNodeHandler      *FakeNodeHandler
		fakeKubeletClient    *FakeKubeletClient
		expectedNodes        []*api.Node
		expectedRequestCount int
	}{
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{Name: "node0"},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Kind:               api.NodeReady,
									Status:             api.ConditionFull,
									Reason:             "Node health check succeeded: kubelet /healthz endpoint returns ok",
									LastTransitionTime: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
			},
			fakeKubeletClient: &FakeKubeletClient{
				Status: probe.Success,
				Err:    nil,
			},
			expectedNodes:        []*api.Node{},
			expectedRequestCount: 1,
		},
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{Name: "node0"},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Kind:               api.NodeReady,
									Status:             api.ConditionFull,
									Reason:             "Node health check succeeded: kubelet /healthz endpoint returns ok",
									LastTransitionTime: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
			},
			fakeKubeletClient: &FakeKubeletClient{
				Status: probe.Failure,
				Err:    nil,
			},
			expectedNodes: []*api.Node{
				{
					ObjectMeta: api.ObjectMeta{Name: "node0"},
					Status: api.NodeStatus{
						Conditions: []api.NodeCondition{
							{
								Kind:               api.NodeReady,
								Status:             api.ConditionFull,
								Reason:             "Node health check failed: kubelet /healthz endpoint returns not ok",
								LastTransitionTime: util.Now(), // Placeholder expected transition time, due to inability to mock time.
							},
						},
					},
				},
			},
			expectedRequestCount: 2,
		},
	}

	for _, item := range table {
		nodeController := NewNodeController(nil, "", []string{"node0"}, nil, item.fakeNodeHandler, item.fakeKubeletClient)
		if err := nodeController.SyncNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.expectedRequestCount != item.fakeNodeHandler.RequestCount {
			t.Errorf("expected %v call, but got %v.", item.expectedRequestCount, item.fakeNodeHandler.RequestCount)
		}
		for i := range item.fakeNodeHandler.UpdatedNodes {
			conditions := item.fakeNodeHandler.UpdatedNodes[i].Status.Conditions
			for j := range conditions {
				if !conditions[j].LastTransitionTime.After(time.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC)) {
					t.Errorf("unexpected timestamp %v", conditions[j].LastTransitionTime)
				}
			}
		}
	}
}

func TestSyncNodeStatus(t *testing.T) {
	table := []struct {
		fakeNodeHandler      *FakeNodeHandler
		fakeKubeletClient    *FakeKubeletClient
		fakeCloud            *fake_cloud.FakeCloud
		expectedNodes        []*api.Node
		expectedRequestCount int
	}{
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{newNode("node0"), newNode("node1")},
			},
			fakeKubeletClient: &FakeKubeletClient{
				Status: probe.Success,
				Err:    nil,
			},
			fakeCloud: &fake_cloud.FakeCloud{
				IP: net.ParseIP("1.2.3.4"),
			},
			expectedNodes: []*api.Node{
				{
					ObjectMeta: api.ObjectMeta{Name: "node0"},
					Status: api.NodeStatus{
						Conditions: []api.NodeCondition{
							{
								Kind:   api.NodeReady,
								Status: api.ConditionFull,
								Reason: "Node health check succeeded: kubelet /healthz endpoint returns ok",
							},
						},
						HostIP: "1.2.3.4",
					},
				},
				{
					ObjectMeta: api.ObjectMeta{Name: "node1"},
					Status: api.NodeStatus{
						Conditions: []api.NodeCondition{
							{
								Kind:   api.NodeReady,
								Status: api.ConditionFull,
								Reason: "Node health check succeeded: kubelet /healthz endpoint returns ok",
							},
						},
						HostIP: "1.2.3.4",
					},
				},
			},
			expectedRequestCount: 3, // List + 2xUpdate
		},
	}

	for _, item := range table {
		nodeController := NewNodeController(item.fakeCloud, ".*", nil, nil, item.fakeNodeHandler, item.fakeKubeletClient)
		if err := nodeController.SyncNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.fakeNodeHandler.RequestCount != item.expectedRequestCount {
			t.Errorf("expected %v call, but got %v.", item.expectedRequestCount, item.fakeNodeHandler.RequestCount)
		}
		for i := range item.fakeNodeHandler.UpdatedNodes {
			conditions := item.fakeNodeHandler.UpdatedNodes[i].Status.Conditions
			for j := range conditions {
				if conditions[j].LastTransitionTime.IsZero() {
					t.Errorf("unexpected zero timestamp")
				}
				conditions[j].LastTransitionTime = util.Time{}
			}
		}
		if !reflect.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodes) {
			t.Errorf("expected nodes %+v, got %+v", item.expectedNodes[0], item.fakeNodeHandler.UpdatedNodes[0])
		}
		item.fakeNodeHandler.RequestCount = 0
		if err := nodeController.SyncNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.fakeNodeHandler.RequestCount != 1 {
			t.Errorf("expected one list for updating same status, but got %v.", item.fakeNodeHandler.RequestCount)
		}
	}
}

func newNode(name string) *api.Node {
	return &api.Node{ObjectMeta: api.ObjectMeta{Name: name}}
}

func sortedNodeNames(nodes []*api.Node) []string {
	nodeNames := []string{}
	for _, node := range nodes {
		nodeNames = append(nodeNames, node.Name)
	}
	sort.Strings(nodeNames)
	return nodeNames
}

func contains(node *api.Node, nodes []*api.Node) bool {
	for i := 0; i < len(nodes); i++ {
		if node.Name == nodes[i].Name {
			return true
		}
	}
	return false
}
