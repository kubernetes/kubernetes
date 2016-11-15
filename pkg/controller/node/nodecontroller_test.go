/*
Copyright 2014 The Kubernetes Authors.

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
	"net"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	testcore "k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/cloudprovider"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	testNodeMonitorGracePeriod = 40 * time.Second
	testNodeStartupGracePeriod = 60 * time.Second
	testNodeMonitorPeriod      = 5 * time.Second
	testRateLimiterQPS         = float32(10000)
	testLargeClusterThreshold  = 20
	testUnhealtyThreshold      = float32(0.55)
)

func NewNodeControllerFromClient(
	cloud cloudprovider.Interface,
	kubeClient clientset.Interface,
	podEvictionTimeout time.Duration,
	evictionLimiterQPS float32,
	secondaryEvictionLimiterQPS float32,
	largeClusterThreshold int32,
	unhealthyZoneThreshold float32,
	nodeMonitorGracePeriod time.Duration,
	nodeStartupGracePeriod time.Duration,
	nodeMonitorPeriod time.Duration,
	clusterCIDR *net.IPNet,
	serviceCIDR *net.IPNet,
	nodeCIDRMaskSize int,
	allocateNodeCIDRs bool) (*NodeController, error) {

	factory := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())

	nc, err := NewNodeController(factory.Pods(), factory.Nodes(), factory.DaemonSets(), cloud, kubeClient, podEvictionTimeout, evictionLimiterQPS, secondaryEvictionLimiterQPS,
		largeClusterThreshold, unhealthyZoneThreshold, nodeMonitorGracePeriod, nodeStartupGracePeriod, nodeMonitorPeriod, clusterCIDR,
		serviceCIDR, nodeCIDRMaskSize, allocateNodeCIDRs)
	if err != nil {
		return nil, err
	}

	return nc, nil
}

func TestMonitorNodeStatusEvictPods(t *testing.T) {
	fakeNow := unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute

	// Because of the logic that prevents NC from evicting anything when all Nodes are NotReady
	// we need second healthy node in tests. Because of how the tests are written we need to update
	// the status of this Node.
	healthyNodeNewStatus := api.NodeStatus{
		Conditions: []api.NodeCondition{
			{
				Type:   api.NodeReady,
				Status: api.ConditionTrue,
				// Node status has just been updated, and is NotReady for 10min.
				LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 9, 0, 0, time.UTC),
				LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}

	table := []struct {
		fakeNodeHandler     *FakeNodeHandler
		daemonSets          []extensions.DaemonSet
		timeToPass          time.Duration
		newNodeStatus       api.NodeStatus
		secondNodeNewStatus api.NodeStatus
		expectedEvictPods   bool
		description         string
	}{
		// Node created recently, with no status (happens only at cluster startup).
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: fakeNow,
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
						},
					},
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionTrue,
									LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
			daemonSets:          nil,
			timeToPass:          0,
			newNodeStatus:       api.NodeStatus{},
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   false,
			description:         "Node created recently, with no status.",
		},
		// Node created long time ago, and kubelet posted NotReady for a short period of time.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
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
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionTrue,
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
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   false,
			description:         "Node created long time ago, and kubelet posted NotReady for a short period of time.",
		},
		// Pod is ds-managed, and kubelet posted NotReady for a long period of time.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
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
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionTrue,
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
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   false,
			description:         "Pod is ds-managed, and kubelet posted NotReady for a long period of time.",
		},
		// Node created long time ago, and kubelet posted NotReady for a long period of time.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
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
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionTrue,
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
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   true,
			description:         "Node created long time ago, and kubelet posted NotReady for a long period of time.",
		},
		// Node created long time ago, node controller posted Unknown for a short period of time.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
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
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionTrue,
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
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   false,
			description:         "Node created long time ago, node controller posted Unknown for a short period of time.",
		},
		// Node created long time ago, node controller posted Unknown for a long period of time.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
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
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionTrue,
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
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   true,
			description:         "Node created long time ago, node controller posted Unknown for a long period of time.",
		},
	}

	for _, item := range table {
		nodeController, _ := NewNodeControllerFromClient(nil, item.fakeNodeHandler,
			evictionTimeout, testRateLimiterQPS, testRateLimiterQPS, testLargeClusterThreshold, testUnhealtyThreshold, testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod, testNodeMonitorPeriod, nil, nil, 0, false)
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
			item.fakeNodeHandler.Existing[1].Status = item.secondNodeNewStatus
		}
		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		zones := getZones(item.fakeNodeHandler)
		for _, zone := range zones {
			nodeController.zonePodEvictor[zone].Try(func(value TimedValue) (bool, time.Duration) {
				nodeUid, _ := value.UID.(string)
				deletePods(item.fakeNodeHandler, nodeController.recorder, value.Value, nodeUid, nodeController.daemonSetStore)
				return true, 0
			})
		}

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

func TestPodStatusChange(t *testing.T) {
	fakeNow := unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute

	// Because of the logic that prevents NC from evicting anything when all Nodes are NotReady
	// we need second healthy node in tests. Because of how the tests are written we need to update
	// the status of this Node.
	healthyNodeNewStatus := api.NodeStatus{
		Conditions: []api.NodeCondition{
			{
				Type:   api.NodeReady,
				Status: api.ConditionTrue,
				// Node status has just been updated, and is NotReady for 10min.
				LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 9, 0, 0, time.UTC),
				LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}

	// Node created long time ago, node controller posted Unknown for a long period of time.
	table := []struct {
		fakeNodeHandler     *FakeNodeHandler
		daemonSets          []extensions.DaemonSet
		timeToPass          time.Duration
		newNodeStatus       api.NodeStatus
		secondNodeNewStatus api.NodeStatus
		expectedPodUpdate   bool
		expectedReason      string
		description         string
	}{
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
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
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								unversioned.LabelZoneRegion:        "region1",
								unversioned.LabelZoneFailureDomain: "zone1",
							},
						},
						Status: api.NodeStatus{
							Conditions: []api.NodeCondition{
								{
									Type:               api.NodeReady,
									Status:             api.ConditionTrue,
									LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
			},
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
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedPodUpdate:   true,
			expectedReason:      node.NodeUnreachablePodReason,
			description: "Node created long time ago, node controller posted Unknown for a " +
				"long period of time, the pod status must include reason for termination.",
		},
	}

	for _, item := range table {
		nodeController, _ := NewNodeControllerFromClient(nil, item.fakeNodeHandler,
			evictionTimeout, testRateLimiterQPS, testRateLimiterQPS, testLargeClusterThreshold, testUnhealtyThreshold, testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod, testNodeMonitorPeriod, nil, nil, 0, false)
		nodeController.now = func() unversioned.Time { return fakeNow }
		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() unversioned.Time { return unversioned.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
			item.fakeNodeHandler.Existing[1].Status = item.secondNodeNewStatus
		}
		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		zones := getZones(item.fakeNodeHandler)
		for _, zone := range zones {
			nodeController.zonePodEvictor[zone].Try(func(value TimedValue) (bool, time.Duration) {
				nodeUid, _ := value.UID.(string)
				deletePods(item.fakeNodeHandler, nodeController.recorder, value.Value, nodeUid, nodeController.daemonSetStore)
				return true, 0
			})
		}

		podReasonUpdate := false
		for _, action := range item.fakeNodeHandler.Actions() {
			if action.GetVerb() == "update" && action.GetResource().Resource == "pods" {
				updateReason := action.(testcore.UpdateActionImpl).GetObject().(*api.Pod).Status.Reason
				podReasonUpdate = true
				if updateReason != item.expectedReason {
					t.Errorf("expected pod status reason: %+v, got %+v for %+v", item.expectedReason, updateReason, item.description)
				}
			}
		}

		if podReasonUpdate != item.expectedPodUpdate {
			t.Errorf("expected pod update: %+v, got %+v for %+v", podReasonUpdate, item.expectedPodUpdate, item.description)
		}
	}

}

func TestMonitorNodeStatusEvictPodsWithDisruption(t *testing.T) {
	fakeNow := unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute
	timeToPass := 60 * time.Minute

	// Because of the logic that prevents NC from evicting anything when all Nodes are NotReady
	// we need second healthy node in tests. Because of how the tests are written we need to update
	// the status of this Node.
	healthyNodeNewStatus := api.NodeStatus{
		Conditions: []api.NodeCondition{
			{
				Type:               api.NodeReady,
				Status:             api.ConditionTrue,
				LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 13, 0, 0, 0, time.UTC),
				LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}
	unhealthyNodeNewStatus := api.NodeStatus{
		Conditions: []api.NodeCondition{
			{
				Type:   api.NodeReady,
				Status: api.ConditionUnknown,
				// Node status was updated by nodecontroller 1hr ago
				LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
				LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}

	table := []struct {
		nodeList                []*api.Node
		podList                 []api.Pod
		updatedNodeStatuses     []api.NodeStatus
		expectedInitialStates   map[string]zoneState
		expectedFollowingStates map[string]zoneState
		expectedEvictPods       bool
		description             string
	}{
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period of time on both Nodes.
		// Only zone is down - eviction shouldn't take place
		{
			nodeList: []*api.Node{
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
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
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node1",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
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
			podList: []api.Pod{*newPod("pod0", "node0")},
			updatedNodeStatuses: []api.NodeStatus{
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
			},
			expectedInitialStates:   map[string]zoneState{createZoneID("region1", "zone1"): stateFullDisruption},
			expectedFollowingStates: map[string]zoneState{createZoneID("region1", "zone1"): stateFullDisruption},
			expectedEvictPods:       false,
			description:             "Network Disruption: Only zone is down - eviction shouldn't take place.",
		},
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period of time on both Nodes.
		// Both zones down - eviction shouldn't take place
		{
			nodeList: []*api.Node{
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
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
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node1",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region2",
							unversioned.LabelZoneFailureDomain: "zone2",
						},
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

			podList: []api.Pod{*newPod("pod0", "node0")},
			updatedNodeStatuses: []api.NodeStatus{
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
			},
			expectedInitialStates: map[string]zoneState{
				createZoneID("region1", "zone1"): stateFullDisruption,
				createZoneID("region2", "zone2"): stateFullDisruption,
			},
			expectedFollowingStates: map[string]zoneState{
				createZoneID("region1", "zone1"): stateFullDisruption,
				createZoneID("region2", "zone2"): stateFullDisruption,
			},
			expectedEvictPods: false,
			description:       "Network Disruption: Both zones down - eviction shouldn't take place.",
		},
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period of time on both Nodes.
		// One zone is down - eviction should take place
		{
			nodeList: []*api.Node{
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
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
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node1",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone2",
						},
					},
					Status: api.NodeStatus{
						Conditions: []api.NodeCondition{
							{
								Type:               api.NodeReady,
								Status:             api.ConditionTrue,
								LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
			},
			podList: []api.Pod{*newPod("pod0", "node0")},
			updatedNodeStatuses: []api.NodeStatus{
				unhealthyNodeNewStatus,
				healthyNodeNewStatus,
			},
			expectedInitialStates: map[string]zoneState{
				createZoneID("region1", "zone1"): stateFullDisruption,
				createZoneID("region1", "zone2"): stateNormal,
			},
			expectedFollowingStates: map[string]zoneState{
				createZoneID("region1", "zone1"): stateFullDisruption,
				createZoneID("region1", "zone2"): stateNormal,
			},
			expectedEvictPods: true,
			description:       "Network Disruption: One zone is down - eviction should take place.",
		},
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period
		// of on first Node, eviction should stop even though -master Node is healthy.
		{
			nodeList: []*api.Node{
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
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
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node-master",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
					},
					Status: api.NodeStatus{
						Conditions: []api.NodeCondition{
							{
								Type:               api.NodeReady,
								Status:             api.ConditionTrue,
								LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
			},
			podList: []api.Pod{*newPod("pod0", "node0")},
			updatedNodeStatuses: []api.NodeStatus{
				unhealthyNodeNewStatus,
				healthyNodeNewStatus,
			},
			expectedInitialStates: map[string]zoneState{
				createZoneID("region1", "zone1"): stateFullDisruption,
			},
			expectedFollowingStates: map[string]zoneState{
				createZoneID("region1", "zone1"): stateFullDisruption,
			},
			expectedEvictPods: false,
			description:       "NetworkDisruption: eviction should stop, only -master Node is healthy",
		},
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period of time on both Nodes.
		// Initially both zones down, one comes back - eviction should take place
		{
			nodeList: []*api.Node{
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
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
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node1",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone2",
						},
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

			podList: []api.Pod{*newPod("pod0", "node0")},
			updatedNodeStatuses: []api.NodeStatus{
				unhealthyNodeNewStatus,
				healthyNodeNewStatus,
			},
			expectedInitialStates: map[string]zoneState{
				createZoneID("region1", "zone1"): stateFullDisruption,
				createZoneID("region1", "zone2"): stateFullDisruption,
			},
			expectedFollowingStates: map[string]zoneState{
				createZoneID("region1", "zone1"): stateFullDisruption,
				createZoneID("region1", "zone2"): stateNormal,
			},
			expectedEvictPods: true,
			description:       "Initially both zones down, one comes back - eviction should take place",
		},
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period of time on both Nodes.
		// Zone is partially disrupted - eviction should take place
		{
			nodeList: []*api.Node{
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
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
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node1",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
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
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node2",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
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
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node3",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
					},
					Status: api.NodeStatus{
						Conditions: []api.NodeCondition{
							{
								Type:               api.NodeReady,
								Status:             api.ConditionTrue,
								LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
				{
					ObjectMeta: api.ObjectMeta{
						Name:              "node4",
						CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							unversioned.LabelZoneRegion:        "region1",
							unversioned.LabelZoneFailureDomain: "zone1",
						},
					},
					Status: api.NodeStatus{
						Conditions: []api.NodeCondition{
							{
								Type:               api.NodeReady,
								Status:             api.ConditionTrue,
								LastHeartbeatTime:  unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: unversioned.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
			},

			podList: []api.Pod{*newPod("pod0", "node0")},
			updatedNodeStatuses: []api.NodeStatus{
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
				healthyNodeNewStatus,
				healthyNodeNewStatus,
			},
			expectedInitialStates: map[string]zoneState{
				createZoneID("region1", "zone1"): statePartialDisruption,
			},
			expectedFollowingStates: map[string]zoneState{
				createZoneID("region1", "zone1"): statePartialDisruption,
			},
			expectedEvictPods: true,
			description:       "Zone is partially disrupted - eviction should take place.",
		},
	}

	for _, item := range table {
		fakeNodeHandler := &FakeNodeHandler{
			Existing:  item.nodeList,
			Clientset: fake.NewSimpleClientset(&api.PodList{Items: item.podList}),
		}
		nodeController, _ := NewNodeControllerFromClient(nil, fakeNodeHandler,
			evictionTimeout, testRateLimiterQPS, testRateLimiterQPS, testLargeClusterThreshold, testUnhealtyThreshold, testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod, testNodeMonitorPeriod, nil, nil, 0, false)
		nodeController.now = func() unversioned.Time { return fakeNow }
		nodeController.enterPartialDisruptionFunc = func(nodeNum int) float32 {
			return testRateLimiterQPS
		}
		nodeController.enterFullDisruptionFunc = func(nodeNum int) float32 {
			return testRateLimiterQPS
		}
		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("%v: unexpected error: %v", item.description, err)
		}

		for zone, state := range item.expectedInitialStates {
			if state != nodeController.zoneStates[zone] {
				t.Errorf("%v: Unexpected zone state: %v: %v instead %v", item.description, zone, nodeController.zoneStates[zone], state)
			}
		}

		nodeController.now = func() unversioned.Time { return unversioned.Time{Time: fakeNow.Add(timeToPass)} }
		for i := range item.updatedNodeStatuses {
			fakeNodeHandler.Existing[i].Status = item.updatedNodeStatuses[i]
		}

		if err := nodeController.monitorNodeStatus(); err != nil {
			t.Errorf("%v: unexpected error: %v", item.description, err)
		}
		// Give some time for rate-limiter to reload
		time.Sleep(50 * time.Millisecond)

		for zone, state := range item.expectedFollowingStates {
			if state != nodeController.zoneStates[zone] {
				t.Errorf("%v: Unexpected zone state: %v: %v instead %v", item.description, zone, nodeController.zoneStates[zone], state)
			}
		}
		zones := getZones(fakeNodeHandler)
		for _, zone := range zones {
			nodeController.zonePodEvictor[zone].Try(func(value TimedValue) (bool, time.Duration) {
				uid, _ := value.UID.(string)
				deletePods(fakeNodeHandler, nodeController.recorder, value.Value, uid, nodeController.daemonSetStore)
				return true, 0
			})
		}

		podEvicted := false
		for _, action := range fakeNodeHandler.Actions() {
			if action.GetVerb() == "delete" && action.GetResource().Resource == "pods" {
				podEvicted = true
				break
			}
		}

		if item.expectedEvictPods != podEvicted {
			t.Errorf("%v: expected pod eviction: %+v, got %+v", item.description, item.expectedEvictPods, podEvicted)
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
	nodeController, _ := NewNodeControllerFromClient(nil, fnh, 10*time.Minute,
		testRateLimiterQPS, testRateLimiterQPS, testLargeClusterThreshold, testUnhealtyThreshold,
		testNodeMonitorGracePeriod, testNodeStartupGracePeriod,
		testNodeMonitorPeriod, nil, nil, 0, false)
	nodeController.cloud = &fakecloud.FakeCloud{}
	nodeController.now = func() unversioned.Time { return unversioned.Date(2016, 1, 1, 12, 0, 0, 0, time.UTC) }
	nodeController.nodeExistsInCloudProvider = func(nodeName types.NodeName) (bool, error) {
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
	if nodeOnQueue := nodeController.zonePodEvictor[""].Remove("node0"); nodeOnQueue {
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
		nodeController, _ := NewNodeControllerFromClient(nil, item.fakeNodeHandler, 5*time.Minute,
			testRateLimiterQPS, testRateLimiterQPS, testLargeClusterThreshold, testUnhealtyThreshold,
			testNodeMonitorGracePeriod, testNodeStartupGracePeriod, testNodeMonitorPeriod, nil, nil, 0, false)
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
							NodeInfo: api.NodeSystemInfo{
								KubeletVersion: "v1.2.0",
							},
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
				NodeInfo: api.NodeSystemInfo{
					KubeletVersion: "v1.2.0",
				},
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
		// Node created long time ago, with outdated kubelet version 1.1.0 and status
		// updated by kubelet exceeds grace period. Expect no action from node controller.
		{
			fakeNodeHandler: &FakeNodeHandler{
				Existing: []*api.Node{
					{
						ObjectMeta: api.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: api.NodeStatus{
							NodeInfo: api.NodeSystemInfo{
								KubeletVersion: "v1.1.0",
							},
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
				NodeInfo: api.NodeSystemInfo{
					KubeletVersion: "v1.1.0",
				},
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
			expectedPodStatusUpdate: false,
		},
	}

	for i, item := range table {
		nodeController, _ := NewNodeControllerFromClient(nil, item.fakeNodeHandler, 5*time.Minute,
			testRateLimiterQPS, testRateLimiterQPS, testLargeClusterThreshold, testUnhealtyThreshold,
			testNodeMonitorGracePeriod, testNodeStartupGracePeriod, testNodeMonitorPeriod, nil, nil, 0, false)
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

func TestNodeEventGeneration(t *testing.T) {
	fakeNow := unversioned.Date(2016, 9, 10, 12, 0, 0, 0, time.UTC)
	fakeNodeHandler := &FakeNodeHandler{
		Existing: []*api.Node{
			{
				ObjectMeta: api.ObjectMeta{
					Name:              "node0",
					UID:               "1234567890",
					CreationTimestamp: unversioned.Date(2015, 8, 10, 0, 0, 0, 0, time.UTC),
				},
				Spec: api.NodeSpec{
					ExternalID: "node0",
				},
				Status: api.NodeStatus{
					Conditions: []api.NodeCondition{
						{
							Type:               api.NodeReady,
							Status:             api.ConditionUnknown,
							LastHeartbeatTime:  unversioned.Date(2015, 8, 10, 0, 0, 0, 0, time.UTC),
							LastTransitionTime: unversioned.Date(2015, 8, 10, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
		},
		Clientset: fake.NewSimpleClientset(&api.PodList{Items: []api.Pod{*newPod("pod0", "node0")}}),
	}

	nodeController, _ := NewNodeControllerFromClient(nil, fakeNodeHandler, 5*time.Minute,
		testRateLimiterQPS, testRateLimiterQPS, testLargeClusterThreshold, testUnhealtyThreshold,
		testNodeMonitorGracePeriod, testNodeStartupGracePeriod,
		testNodeMonitorPeriod, nil, nil, 0, false)
	nodeController.cloud = &fakecloud.FakeCloud{}
	nodeController.nodeExistsInCloudProvider = func(nodeName types.NodeName) (bool, error) {
		return false, nil
	}
	nodeController.now = func() unversioned.Time { return fakeNow }
	fakeRecorder := NewFakeRecorder()
	nodeController.recorder = fakeRecorder
	if err := nodeController.monitorNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(fakeRecorder.events) != 2 {
		t.Fatalf("unexpected events, got %v, expected %v: %+v", len(fakeRecorder.events), 2, fakeRecorder.events)
	}
	if fakeRecorder.events[0].Reason != "RegisteredNode" || fakeRecorder.events[1].Reason != "DeletingNode" {
		var reasons []string
		for _, event := range fakeRecorder.events {
			reasons = append(reasons, event.Reason)
		}
		t.Fatalf("unexpected events generation: %v", strings.Join(reasons, ","))
	}
	for _, event := range fakeRecorder.events {
		involvedObject := event.InvolvedObject
		actualUID := string(involvedObject.UID)
		if actualUID != "1234567890" {
			t.Fatalf("unexpected event uid: %v", actualUID)
		}
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
			prune: false,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{DeletionTimestamp: &unversioned.Time{}},
				Spec:       api.PodSpec{NodeName: "nonexistant"},
			},
			prune: false,
		},
	}

	nc, _ := NewNodeControllerFromClient(nil, fake.NewSimpleClientset(), 0, 0, 0, 0, 0, 0, 0, 0, nil, nil, 0, false)
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

func TestCheckNodeKubeletVersionParsing(t *testing.T) {
	tests := []struct {
		version  string
		outdated bool
	}{
		{
			version:  "",
			outdated: true,
		},
		{
			version:  "v0.21.4",
			outdated: true,
		},
		{
			version:  "v1.0.0",
			outdated: true,
		},
		{
			version:  "v1.1.0",
			outdated: true,
		},
		{
			version:  "v1.1.0-alpha.2.961+9d4c6846fc03b9-dirty",
			outdated: true,
		},
		{
			version:  "v1.2.0",
			outdated: false,
		},
		{
			version:  "v1.3.3",
			outdated: false,
		},
		{
			version:  "v1.4.0-alpha.2.961+9d4c6846fc03b9-dirty",
			outdated: false,
		},
		{
			version:  "v2.0.0",
			outdated: false,
		},
	}

	for _, ov := range tests {
		n := &api.Node{
			Status: api.NodeStatus{
				NodeInfo: api.NodeSystemInfo{
					KubeletVersion: ov.version,
				},
			},
		}
		isOutdated := nodeRunningOutdatedKubelet(n)
		if ov.outdated != isOutdated {
			t.Errorf("Version %v doesn't match test expectation. Expected outdated %v got %v", n.Status.NodeInfo.KubeletVersion, ov.outdated, isOutdated)
		} else {
			t.Logf("Version %v outdated %v", ov.version, isOutdated)
		}
	}
}
