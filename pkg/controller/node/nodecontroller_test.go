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

package node

import (
	"errors"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	testNodeMonitorGracePeriod = 40 * time.Second
	testNodeStartupGracePeriod = 60 * time.Second
	testNodeMonitorPeriod      = 5 * time.Second
)

// FakeNodeHandler is a fake implementation of NodesInterface and NodeInterface. It
// allows test cases to have fine-grained control over mock behaviors. We also need
// PodsInterface and PodInterface to test list & delet pods, which is implemented in
// the embedded client.Fake field.
type FakeNodeHandler struct {
	*fake.Clientset

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
	createLock     sync.Mutex
	deleteWaitChan chan struct{}
}

type FakeLegacyHandler struct {
	unversionedcore.CoreInterface
	n *FakeNodeHandler
}

func (c *FakeNodeHandler) Core() unversionedcore.CoreInterface {
	return &FakeLegacyHandler{c.Clientset.Core(), c}
}

func (m *FakeLegacyHandler) Nodes() unversionedcore.NodeInterface {
	return m.n
}

func (m *FakeNodeHandler) Create(node *api.Node) (*api.Node, error) {
	m.createLock.Lock()
	defer func() {
		m.RequestCount++
		m.createLock.Unlock()
	}()
	for _, n := range m.Existing {
		if n.Name == node.Name {
			return nil, apierrors.NewAlreadyExists(api.Resource("nodes"), node.Name)
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

func (m *FakeNodeHandler) List(opts api.ListOptions) (*api.NodeList, error) {
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

func (m *FakeNodeHandler) Delete(id string, opt *api.DeleteOptions) error {
	defer func() {
		if m.deleteWaitChan != nil {
			m.deleteWaitChan <- struct{}{}
		}
	}()
	m.DeletedNodes = append(m.DeletedNodes, newNode(id))
	m.RequestCount++
	return nil
}

func (m *FakeNodeHandler) DeleteCollection(opt *api.DeleteOptions, listOpts api.ListOptions) error {
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

func (m *FakeNodeHandler) Watch(opts api.ListOptions) (watch.Interface, error) {
	return nil, nil
}

func TestMonitorNodeStatusEvictPods(t *testing.T) {
	fakeNow := unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute

	table := []struct {
		fakeNodeHandler   *FakeNodeHandler
		daemonSets        []extensions.DaemonSet
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
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			daemonSets:        nil,
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
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionFalse,
									LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			daemonSets: nil,
			timeToPass: evictionTimeout,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionFalse,
						// Node status has just been updated, and is NotReady for 10min.
						LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 9, 0, 0, time.UTC),
						LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			expectedEvictPods: false,
			description:       "Node created long time ago, and kubelet posted NotReady for a short period of time.",
		},
		// Pod is ds-managed, and kubelet posted NotReady for a long period of time.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionFalse,
									LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(
					&api.PodList{
						Items: []api.Pod{
							{
								ObjectMeta: api.ObjectMeta{
									Name:      "pod0",
									Namespace: "default",
									Labels:    map[string]string{"daemon": "yes"},
								},
								Spec: api.PodSpec{
									NodeName: "node0",
								},
							},
						},
					},
				),
			},
			daemonSets: []extensions.DaemonSet{
				{
					ObjectMeta: api.ObjectMeta{
						Name:      "ds0",
						Namespace: "default",
					},
					Spec: extensions.DaemonSetSpec{
						Selector: &unversioned.LabelSelector{
							MatchLabels: map[string]string{"daemon": "yes"},
						},
					},
				},
			},
			timeToPass: time.Hour,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionFalse,
						// Node status has just been updated, and is NotReady for 1hr.
						LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 59, 0, 0, time.UTC),
						LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			expectedEvictPods: false,
			description:       "Pod is ds-managed, and kubelet posted NotReady for a long period of time.",
		},
		// Node created long time ago, and kubelet posted NotReady for a long period of time.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionFalse,
									LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			daemonSets: nil,
			timeToPass: time.Hour,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionFalse,
						// Node status has just been updated, and is NotReady for 1hr.
						LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 59, 0, 0, time.UTC),
						LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
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
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionUnknown,
									LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			daemonSets: nil,
			timeToPass: evictionTimeout - testNodeMonitorGracePeriod,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionUnknown,
						// Node status was updated by nodecontroller 10min ago
						LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
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
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionUnknown,
									LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			daemonSets: nil,
			timeToPass: 60 * time.Minute,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionUnknown,
						// Node status was updated by nodecontroller 1hr ago
						LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			expectedEvictPods: true,
			description:       "Node created long time ago, node controller posted Unknown for a long period of time.",
		},
	}

	for _, item := range table {
		nodeController := NewNodeController(nil, item.fakeNodeHandler,
			evictionTimeout, flowcontrol.NewFakeAlwaysRateLimiter(), flowcontrol.NewFakeAlwaysRateLimiter(), testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod, testNodeMonitorPeriod, nil, false)
		nodeController.now = func() unversioned.Time { return fakeNow }
		for _, ds := range item.daemonSets {
			nodeController.daemonSetStore.Add(&ds)
		}
		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() unversioned.Time { return unversioned.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
		}
		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		nodeController.podEvictor.Try(func(value TimedValue) (bool, time.Duration) {
			remaining, _ := nodeController.deletePods(value.Value)
			if remaining {
				nodeController.terminationEvictor.Add(value.Value)
			}
			return true, 0
		})
		nodeController.podEvictor.Try(func(value TimedValue) (bool, time.Duration) {
			nodeController.terminatePods(value.Value, value.AddedAt)
			return true, 0
		})
		podEvicted := false
		for _, action := range item.fakeNodeHandler.Actions() {
			if action.GetVerb() == "delete" && action.GetResource().Resource == "pods" {
				podEvicted = true
			}
		}

		if item.expectedEvictPods != podEvicted {
			t.Errorf("expected pod eviction: %+v, got %+v for %+v", item.expectedEvictPods,
				podEvicted, item.description)
		}
	}
}

// TestCloudProviderNoRateLimit tests that monitorNodes() immediately deletes
// pods and the node when kubelet has not reported, and the cloudprovider says
// the node is gone.
func TestCloudProviderNoRateLimit(t *testing.T) {
	fnh := &FakeNodeHandler{
		Existing: []*api.Node{
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: api.NodeStatus{
					Conditions: []api.NodeCondition{
						{
							Type:               api.NodeReady,
							Status:             api.ConditionUnknown,
							LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
		},
		Clientset:      fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0"), *newPod("pod1", "node0")}}),
		deleteWaitChan: make(chan struct{}),
	}
	nodeController := NewNodeController(nil, fnh, 10*time.Minute,
		flowcontrol.NewFakeAlwaysRateLimiter(), flowcontrol.NewFakeAlwaysRateLimiter(),
		testNodeMonitorGracePeriod, testNodeStartupGracePeriod,
		testNodeMonitorPeriod, nil, false)
	nodeController.cloud = &fakecloud.FakeCloud{}
	nodeController.now = func() unversioned.Time { return unversioned.Date(2016, 1, 1, 12, 0, 0, 0, time.UTC) }
	nodeController.nodeExistsInCloudProvider = func(nodeName string) (bool, error) {
		return false, nil
	}
	// monitorNodeStatus should allow this node to be immediately deleted
	if err := nodeController.monitorNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	select {
	case <-fnh.deleteWaitChan:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("Timed out waiting %v for node to be deleted", wait.ForeverTestTimeout)
	}
	if len(fnh.DeletedNodes) != 1 || fnh.DeletedNodes[0].Name != "node0" {
		t.Errorf("Node was not deleted")
	}
	if nodeOnQueue := nodeController.podEvictor.Remove("node0"); nodeOnQueue {
		t.Errorf("Node was queued for eviction. Should have been immediately deleted.")
	}
}

func TestMonitorNodeStatusUpdateStatus(t *testing.T) {
	fakeNow := unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
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
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			expectedRequestCount: 2, // List+Update
			expectedNodes: []*api.Node{
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					},
					Status: api.NodeStatus{
						Conditions: []api.NodeCondition{
							{
								Type:               api.NodeReady,
								Status:             api.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
								LastTransitionTime: fakeNow,
							},
							{
								Type:               api.NodeOutOfDisk,
								Status:             api.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
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
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
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
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:   api.NodeReady,
									Status: api.ConditionTrue,
									// Node status hasn't been updated for 1hr.
									LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
								{
									Type:   api.NodeOutOfDisk,
									Status: api.ConditionFalse,
									// Node status hasn't been updated for 1hr.
									LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
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
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			expectedRequestCount: 3, // (List+)List+Update
			timeToPass:           time.Hour,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionTrue,
						// Node status hasn't been updated for 1hr.
						LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
					{
						Type:   api.NodeOutOfDisk,
						Status: api.ConditionFalse,
						// Node status hasn't been updated for 1hr.
						LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
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
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					},
					Status: api.NodeStatus{
						Conditions: []api.NodeCondition{
							{
								Type:               api.NodeReady,
								Status:             api.ConditionUnknown,
								Reason:             "NodeStatusUnknown",
								Message:            "Kubelet stopped posting node status.",
								LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: unversioned.Time{Time: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC).Add(time.Hour)},
							},
							{
								Type:               api.NodeOutOfDisk,
								Status:             api.ConditionUnknown,
								Reason:             "NodeStatusUnknown",
								Message:            "Kubelet stopped posting node status.",
								LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: unversioned.Time{Time: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC).Add(time.Hour)},
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
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
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
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			expectedRequestCount: 1, // List
			expectedNodes:        nil,
		},
	}

	for i, item := range table {
		nodeController := NewNodeController(nil, item.fakeNodeHandler, 5*time.Minute, flowcontrol.NewFakeAlwaysRateLimiter(),
			flowcontrol.NewFakeAlwaysRateLimiter(), testNodeMonitorGracePeriod, testNodeStartupGracePeriod, testNodeMonitorPeriod, nil, false)
		nodeController.now = func() unversioned.Time { return fakeNow }
		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() unversioned.Time { return unversioned.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
			if err := nodeController.monitorNodeStatus(); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		}
		if item.expectedRequestCount != item.fakeNodeHandler.RequestCount {
			t.Errorf("expected %v call, but got %v.", item.expectedRequestCount, item.fakeNodeHandler.RequestCount)
		}
		if len(item.fakeNodeHandler.UpdatedNodes) > 0 && !api.Semantic.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodes) {
			t.Errorf("Case[%d] unexpected nodes: %s", i, diff.ObjectDiff(item.expectedNodes[0], item.fakeNodeHandler.UpdatedNodes[0]))
		}
		if len(item.fakeNodeHandler.UpdatedNodeStatuses) > 0 && !api.Semantic.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodeStatuses) {
			t.Errorf("Case[%d] unexpected nodes: %s", i, diff.ObjectDiff(item.expectedNodes[0], item.fakeNodeHandler.UpdatedNodeStatuses[0]))
		}
	}
}

func TestMonitorNodeStatusMarkPodsNotReady(t *testing.T) {
	fakeNow := unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	table := []struct {
		fakeNodeHandler         *FakeNodeHandler
		timeToPass              time.Duration
		newNodeStatus           api.NodeStatus
		expectedPodStatusUpdate bool
	}{
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
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			expectedPodStatusUpdate: false,
		},
		// Node created long time ago, with status updated recently.
		// Expect no action from node controller (within monitor grace period).
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
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
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			expectedPodStatusUpdate: false,
		},
		// Node created long time ago, with status updated by kubelet exceeds grace period.
		// Expect pods status updated and Unknown node status posted from node controller
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:   api.NodeReady,
									Status: api.ConditionTrue,
									// Node status hasn't been updated for 1hr.
									LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
								{
									Type:   api.NodeOutOfDisk,
									Status: api.ConditionFalse,
									// Node status hasn't been updated for 1hr.
									LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
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
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			timeToPass: 1 * time.Minute,
			newNodeStatus: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:   api.NodeReady,
						Status: api.ConditionTrue,
						// Node status hasn't been updated for 1hr.
						LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
					{
						Type:   api.NodeOutOfDisk,
						Status: api.ConditionFalse,
						// Node status hasn't been updated for 1hr.
						LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
				},
			},
			expectedPodStatusUpdate: true,
		},
	}

	for i, item := range table {
		nodeController := NewNodeController(nil, item.fakeNodeHandler, 5*time.Minute, flowcontrol.NewFakeAlwaysRateLimiter(),
			flowcontrol.NewFakeAlwaysRateLimiter(), testNodeMonitorGracePeriod, testNodeStartupGracePeriod, testNodeMonitorPeriod, nil, false)
		nodeController.now = func() unversioned.Time { return fakeNow }
		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("Case[%d] unexpected error: %v", i, err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() unversioned.Time { return unversioned.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
			if err := nodeController.monitorNodeStatus(); err != nil {
				t.Errorf("Case[%d] unexpected error: %v", i, err)
			}
		}

		podStatusUpdated := false
		for _, action := range item.fakeNodeHandler.Actions() {
			if action.GetVerb() == "update" && action.GetResource().Resource == "pods" && action.GetSubresource() == "status" {
				podStatusUpdated = true
			}
		}
		if podStatusUpdated != item.expectedPodStatusUpdate {
			t.Errorf("Case[%d] expect pod status updated to be %v, but got %v", i, item.expectedPodStatusUpdate, podStatusUpdated)
		}
	}
}

func TestNodeDeletion(t *testing.T) {
	fakeNow := unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	fakeNodeHandler := &FakeNodeHandler{
		Existing: []*api.Node{
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
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
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "node1",
					CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
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
		Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0"), *newPod("pod1", "node1")}}),
	}

	nodeController := NewNodeController(nil, fakeNodeHandler, 5*time.Minute, flowcontrol.NewFakeAlwaysRateLimiter(), flowcontrol.NewFakeAlwaysRateLimiter(),
		testNodeMonitorGracePeriod, testNodeStartupGracePeriod, testNodeMonitorPeriod, nil, false)
	nodeController.now = func() unversioned.Time { return fakeNow }
	if err := nodeController.monitorNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fakeNodeHandler.Delete("node1", nil)
	if err := nodeController.monitorNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.podEvictor.Try(func(value TimedValue) (bool, time.Duration) {
		nodeController.deletePods(value.Value)
		return true, 0
	})
	podEvicted := false
	for _, action := range fakeNodeHandler.Actions() {
		if action.GetVerb() == "delete" && action.GetResource().Resource == "pods" {
			podEvicted = true
		}
	}
	if !podEvicted {
		t.Error("expected pods to be evicted from the deleted node")
	}
}

func TestCheckPod(t *testing.T) {

	tcs := []struct {
		pod   api.Pod
		prune bool
	}{

		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{DeletionTimestamp: nil},
				Spec:       api.PodSpec{NodeName: "new"},
			},
			prune: false,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{DeletionTimestamp: nil},
				Spec:       api.PodSpec{NodeName: "old"},
			},
			prune: false,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{DeletionTimestamp: nil},
				Spec:       api.PodSpec{NodeName: ""},
			},
			prune: false,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{DeletionTimestamp: nil},
				Spec:       api.PodSpec{NodeName: "nonexistant"},
			},
			prune: false,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{DeletionTimestamp: &unversioned.Time{}},
				Spec:       api.PodSpec{NodeName: "new"},
			},
			prune: false,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{DeletionTimestamp: &unversioned.Time{}},
				Spec:       api.PodSpec{NodeName: "old"},
			},
			prune: true,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{DeletionTimestamp: &unversioned.Time{}},
				Spec:       api.PodSpec{NodeName: "older"},
			},
			prune: true,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{DeletionTimestamp: &unversioned.Time{}},
				Spec:       api.PodSpec{NodeName: "oldest"},
			},
			prune: true,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{DeletionTimestamp: &unversioned.Time{}},
				Spec:       api.PodSpec{NodeName: ""},
			},
			prune: true,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{DeletionTimestamp: &unversioned.Time{}},
				Spec:       api.PodSpec{NodeName: "nonexistant"},
			},
			prune: true,
		},
	}

	nc := NewNodeController(nil, nil, 0, nil, nil, 0, 0, 0, nil, false)
	nc.nodeStore.Store = cache.NewStore(cache.MetaNamespaceKeyFunc)
	nc.nodeStore.Store.Add(&api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: "new",
		},
		Status: api.NodeStatus{
			NodeInfo: api.NodeSystemInfo{
				KubeletVersion: "v1.1.0",
			},
		},
	})
	nc.nodeStore.Store.Add(&api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: "old",
		},
		Status: api.NodeStatus{
			NodeInfo: api.NodeSystemInfo{
				KubeletVersion: "v1.0.0",
			},
		},
	})
	nc.nodeStore.Store.Add(&api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: "older",
		},
		Status: api.NodeStatus{
			NodeInfo: api.NodeSystemInfo{
				KubeletVersion: "v0.21.4",
			},
		},
	})
	nc.nodeStore.Store.Add(&api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: "oldest",
		},
		Status: api.NodeStatus{
			NodeInfo: api.NodeSystemInfo{
				KubeletVersion: "v0.19.3",
			},
		},
	})

	for i, tc := range tcs {
		var deleteCalls int
		nc.forcefullyDeletePod = func(_ *api.Pod) error {
			deleteCalls++
			return nil
		}

		nc.maybeDeleteTerminatingPod(&tc.pod)

		if tc.prune && deleteCalls != 1 {
			t.Errorf("[%v] expected number of delete calls to be 1 but got %v", i, deleteCalls)
		}
		if !tc.prune && deleteCalls != 0 {
			t.Errorf("[%v] expected number of delete calls to be 0 but got %v", i, deleteCalls)
		}
	}
}

func TestCleanupOrphanedPods(t *testing.T) {
	newPod := func(name, node string) api.Pod {
		return api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: name,
			},
			Spec: api.PodSpec{
				NodeName: node,
			},
		}
	}
	pods := []api.Pod{
		newPod("a", "foo"),
		newPod("b", "bar"),
		newPod("c", "gone"),
	}
	nc := NewNodeController(nil, nil, 0, nil, nil, 0, 0, 0, nil, false)

	nc.nodeStore.Store.Add(newNode("foo"))
	nc.nodeStore.Store.Add(newNode("bar"))
	for _, pod := range pods {
		p := pod
		nc.podStore.Indexer.Add(&p)
	}

	var deleteCalls int
	var deletedPodName string
	nc.forcefullyDeletePod = func(p *api.Pod) error {
		deleteCalls++
		deletedPodName = p.ObjectMeta.Name
		return nil
	}
	nc.cleanupOrphanedPods()

	if deleteCalls != 1 {
		t.Fatalf("expected one delete, got: %v", deleteCalls)
	}
	if deletedPodName != "c" {
		t.Fatalf("expected deleted pod name to be 'c', but got: %q", deletedPodName)
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
	return &api.Pod{ObjectMeta: api.ObjectMeta{Name: name}, Spec: api.PodSpec{NodeName: host},
		Status: api.PodStatus{Conditions: []api.PodCondition{{Type: api.PodReady, Status: api.ConditionTrue}}}}
}

func contains(node *api.Node, nodes []*api.Node) bool {
	for i := 0; i < len(nodes); i++ {
		if node.Name == nodes[i].Name {
			return true
		}
	}
	return false
}
