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

package nodecontroller

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

const (
	testNodeMonitorGracePeriod = 40 * time.Second
	testNodeStartupGracePeriod = 60 * time.Second
	testNodeMonitorPeriod      = 5 * time.Second
)

// FakeNodeHandler is a fake implementation of NodesInterface and NodeInterface. It
// allows test cases to have fine-grained control over mock behaviors. We also need
// PodsInterface and PodInterface to test list & delet pods, which is implemented in
// the embeded client.Fake field.
type FakeNodeHandler struct {
	*testclient.Fake

	// Input: Hooks determine if request is valid or not
	CreateHook func(*FakeNodeHandler, *api.Node) bool
	Existing   []*api.Node

	// Output
	CreatedNodes        []*api.Node
	DeletedNodes        []*api.Node
	UpdatedNodes        []*api.Node
	UpdatedNodeStatuses []*api.Node
	RequestCount        int

	// Synchronization
	createLock sync.Mutex
}

func (c *FakeNodeHandler) Nodes() client.NodeInterface {
	return c
}

func (m *FakeNodeHandler) Create(node *api.Node) (*api.Node, error) {
	m.createLock.Lock()
	defer func() {
		m.RequestCount++
		m.createLock.Unlock()
	}()
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

func (m *FakeNodeHandler) Get(name string) (*api.Node, error) {
	return nil, nil
}

func (m *FakeNodeHandler) List(label labels.Selector, field fields.Selector) (*api.NodeList, error) {
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

func (m *FakeNodeHandler) UpdateStatus(node *api.Node) (*api.Node, error) {
	nodeCopy := *node
	m.UpdatedNodeStatuses = append(m.UpdatedNodeStatuses, &nodeCopy)
	m.RequestCount++
	return node, nil
}

func (m *FakeNodeHandler) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return nil, nil
}

func TestMonitorNodeStatusEvictPods(t *testing.T) {
	fakeNow := util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute

	table := []struct {
		fakeNodeHandler   *FakeNodeHandler
		timeToPass        time.Duration
		newNodeStatus     api.NodeStatus
		expectedEvictPods bool
		description       string
	}{
		// Node created recently, with no status (happens only at cluster startup).
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: fakeNow,
						},
					},
				},
				Fake: testclient.NewSimpleFake(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			timeToPass:        0,
			newNodeStatus:     api.NodeStatus{},
			expectedEvictPods: false,
			description:       "Node created recently, with no status.",
		},
		// Node created long time ago, and kubelet posted NotReady for a short period of time.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionFalse,
									LastHeartbeatTime:  util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Fake: testclient.NewSimpleFake(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			timeToPass: evictionTimeout,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionFalse,
						// Node status has just been updated, and is NotReady for 10min.
						LastHeartbeatTime:  util.Date(2015, 1, 1, 12, 9, 0, 0, time.UTC),
						LastTransitionTime: util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			expectedEvictPods: false,
			description:       "Node created long time ago, and kubelet posted NotReady for a short period of time.",
		},
		// Node created long time ago, and kubelet posted NotReady for a long period of time.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionFalse,
									LastHeartbeatTime:  util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Fake: testclient.NewSimpleFake(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			timeToPass: time.Hour,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionFalse,
						// Node status has just been updated, and is NotReady for 1hr.
						LastHeartbeatTime:  util.Date(2015, 1, 1, 12, 59, 0, 0, time.UTC),
						LastTransitionTime: util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			expectedEvictPods: true,
			description:       "Node created long time ago, and kubelet posted NotReady for a long period of time.",
		},
		// Node created long time ago, node controller posted Unknown for a short period of time.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionUnknown,
									LastHeartbeatTime:  util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Fake: testclient.NewSimpleFake(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			timeToPass: evictionTimeout - testNodeMonitorGracePeriod,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionUnknown,
						// Node status was updated by nodecontroller 10min ago
						LastHeartbeatTime:  util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			expectedEvictPods: false,
			description:       "Node created long time ago, node controller posted Unknown for a short period of time.",
		},
		// Node created long time ago, node controller posted Unknown for a long period of time.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionUnknown,
									LastHeartbeatTime:  util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Fake: testclient.NewSimpleFake(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			timeToPass: 60 * time.Minute,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionUnknown,
						// Node status was updated by nodecontroller 1hr ago
						LastHeartbeatTime:  util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			expectedEvictPods: true,
			description:       "Node created long time ago, node controller posted Unknown for a long period of time.",
		},
	}

	for _, item := range table {
		nodeController := NewNodeController(nil, item.fakeNodeHandler, 10,
			evictionTimeout, util.NewFakeRateLimiter(), testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod, testNodeMonitorPeriod, nil, false)
		nodeController.now = func() util.Time { return fakeNow }
		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() util.Time { return util.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
		}
		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		podEvicted := false
		for _, action := range item.fakeNodeHandler.Actions {
			if action.Action == "delete-pod" {
				podEvicted = true
			}
		}
		if item.expectedEvictPods != podEvicted {
			t.Errorf("expected pod eviction: %+v, got %+v for %+v", item.expectedEvictPods,
				podEvicted, item.description)
		}
	}
}

func TestMonitorNodeStatusUpdateStatus(t *testing.T) {
	fakeNow := util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	table := []struct {
		fakeNodeHandler      *FakeNodeHandler
		timeToPass           time.Duration
		newNodeStatus        api.NodeStatus
		expectedEvictPods    bool
		expectedRequestCount int
		expectedNodes        []*api.Node
	}{
		// Node created long time ago, without status:
		// Expect Unknown status posted from node controller.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
					},
				},
				Fake: testclient.NewSimpleFake(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			expectedRequestCount: 2, // List+Update
			expectedNodes: []*api.Node{
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					},
					Status: api.NodeStatus{
						Conditions: []api.NodeCondition{
							{
								Type:               api.NodeReady,
								Status:             api.ConditionUnknown,
								Reason:             fmt.Sprintf("Kubelet never posted node status."),
								LastHeartbeatTime:  util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
								LastTransitionTime: fakeNow,
							},
						},
					},
				},
			},
		},
		// Node created recently, without status.
		// Expect no action from node controller (within startup grace period).
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: fakeNow,
						},
					},
				},
				Fake: testclient.NewSimpleFake(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			expectedRequestCount: 1, // List
			expectedNodes:        nil,
		},
		// Node created long time ago, with status updated by kubelet exceeds grace period.
		// Expect Unknown status posted from node controller.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:   api.NodeReady,
									Status: api.ConditionTrue,
									// Node status hasn't been updated for 1hr.
									LastHeartbeatTime:  util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
							Capacity: api.ResourceList{
								api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
								api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
							},
						},
						Spec: api.NodeSpec{
							ExternalID: "node0",
						},
					},
				},
				Fake: testclient.NewSimpleFake(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			expectedRequestCount: 3, // (List+)List+Update
			timeToPass:           time.Hour,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionTrue,
						// Node status hasn't been updated for 1hr.
						LastHeartbeatTime:  util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
				},
			},
			expectedNodes: []*api.Node{
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					},
					Status: api.NodeStatus{
						Conditions: []api.NodeCondition{
							{
								Type:               api.NodeReady,
								Status:             api.ConditionUnknown,
								Reason:             fmt.Sprintf("Kubelet stopped posting node status."),
								LastHeartbeatTime:  util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: util.Time{util.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC).Add(time.Hour)},
							},
						},
						Capacity: api.ResourceList{
							api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
							api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
						},
					},
					Spec: api.NodeSpec{
						ExternalID: "node0",
					},
				},
			},
		},
		// Node created long time ago, with status updated recently.
		// Expect no action from node controller (within monitor grace period).
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:   api.NodeReady,
									Status: api.ConditionTrue,
									// Node status has just been updated.
									LastHeartbeatTime:  fakeNow,
									LastTransitionTime: fakeNow,
								},
							},
							Capacity: api.ResourceList{
								api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
								api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
							},
						},
						Spec: api.NodeSpec{
							ExternalID: "node0",
						},
					},
				},
				Fake: testclient.NewSimpleFake(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			expectedRequestCount: 1, // List
			expectedNodes:        nil,
		},
	}

	for _, item := range table {
		nodeController := NewNodeController(nil, item.fakeNodeHandler, 10, 5*time.Minute, util.NewFakeRateLimiter(),
			testNodeMonitorGracePeriod, testNodeStartupGracePeriod, testNodeMonitorPeriod, nil, false)
		nodeController.now = func() util.Time { return fakeNow }
		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() util.Time { return util.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
			if err := nodeController.monitorNodeStatus(); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		}
		if item.expectedRequestCount != item.fakeNodeHandler.RequestCount {
			t.Errorf("expected %v call, but got %v.", item.expectedRequestCount, item.fakeNodeHandler.RequestCount)
		}
		if len(item.fakeNodeHandler.UpdatedNodes) > 0 && !api.Semantic.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodes) {
			t.Errorf("expected nodes %+v, got %+v", item.expectedNodes[0],
				item.fakeNodeHandler.UpdatedNodes[0])
		}
	}
}

func newNode(name string) *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{Name: name},
		Spec: api.NodeSpec{
			ExternalID: name,
		},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
				api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
			},
		},
	}
}

func newPod(name, host string) *api.Pod {
	return &api.Pod{ObjectMeta: api.ObjectMeta{Name: name}, Spec: api.PodSpec{Host: host}}
}

func sortedNodeNames(nodes []*api.Node) []string {
	nodeNames := []string{}
	for _, node := range nodes {
		nodeNames = append(nodeNames, node.Name)
	}
	sort.Strings(nodeNames)
	return nodeNames
}

func sortedNodeAddresses(nodes []*api.Node) []string {
	nodeAddresses := []string{}
	for _, node := range nodes {
		for _, addr := range node.Status.Addresses {
			nodeAddresses = append(nodeAddresses, addr.Address)
		}
	}
	sort.Strings(nodeAddresses)
	return nodeAddresses
}

func sortedNodeExternalIDs(nodes []*api.Node) []string {
	nodeExternalIDs := []string{}
	for _, node := range nodes {
		nodeExternalIDs = append(nodeExternalIDs, node.Spec.ExternalID)
	}
	sort.Strings(nodeExternalIDs)
	return nodeExternalIDs
}

func contains(node *api.Node, nodes []*api.Node) bool {
	for i := 0; i < len(nodes); i++ {
		if node.Name == nodes[i].Name {
			return true
		}
	}
	return false
}
