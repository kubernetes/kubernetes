/*
Copyright 2017 The Kubernetes Authors.

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

package nodelifecycle

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	coordv1 "k8s.io/api/coordination/v1"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	appsinformers "k8s.io/client-go/informers/apps/v1"
	coordinformers "k8s.io/client-go/informers/coordination/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	testcore "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/nodelifecycle/scheduler"
	"k8s.io/kubernetes/pkg/controller/testutil"
	nodeutil "k8s.io/kubernetes/pkg/controller/util/node"
	"k8s.io/kubernetes/pkg/features"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/util/node"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
	"k8s.io/utils/pointer"
)

const (
	testNodeMonitorGracePeriod = 40 * time.Second
	testNodeStartupGracePeriod = 60 * time.Second
	testNodeMonitorPeriod      = 5 * time.Second
	testRateLimiterQPS         = float32(10000)
	testLargeClusterThreshold  = 20
	testUnhealthyThreshold     = float32(0.55)
)

func alwaysReady() bool { return true }

func fakeGetPodsAssignedToNode(c *fake.Clientset) func(string) ([]*v1.Pod, error) {
	return func(nodeName string) ([]*v1.Pod, error) {
		selector := fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName})
		pods, err := c.CoreV1().Pods(v1.NamespaceAll).List(context.TODO(), metav1.ListOptions{
			FieldSelector: selector.String(),
			LabelSelector: labels.Everything().String(),
		})
		if err != nil {
			return nil, fmt.Errorf("failed to get Pods assigned to node %v", nodeName)
		}
		rPods := make([]*v1.Pod, len(pods.Items))
		for i := range pods.Items {
			rPods[i] = &pods.Items[i]
		}
		return rPods, nil
	}
}

type nodeLifecycleController struct {
	*Controller
	leaseInformer     coordinformers.LeaseInformer
	nodeInformer      coreinformers.NodeInformer
	daemonSetInformer appsinformers.DaemonSetInformer
}

// doEviction does the fake eviction and returns the status of eviction operation.
func (nc *nodeLifecycleController) doEviction(fakeNodeHandler *testutil.FakeNodeHandler) bool {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	zones := testutil.GetZones(fakeNodeHandler)
	for _, zone := range zones {
		nc.zonePodEvictor[zone].Try(func(value scheduler.TimedValue) (bool, time.Duration) {
			uid, _ := value.UID.(string)
			pods, _ := nc.getPodsAssignedToNode(value.Value)
			nodeutil.DeletePods(fakeNodeHandler, pods, nc.recorder, value.Value, uid, nc.daemonSetStore)
			_ = nc.nodeEvictionMap.setStatus(value.Value, evicted)
			return true, 0
		})
	}

	for _, action := range fakeNodeHandler.Actions() {
		if action.GetVerb() == "delete" && action.GetResource().Resource == "pods" {
			return true
		}
	}
	return false
}

func createNodeLease(nodeName string, renewTime metav1.MicroTime) *coordv1.Lease {
	return &coordv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      nodeName,
			Namespace: v1.NamespaceNodeLease,
		},
		Spec: coordv1.LeaseSpec{
			HolderIdentity: pointer.StringPtr(nodeName),
			RenewTime:      &renewTime,
		},
	}
}

func (nc *nodeLifecycleController) syncLeaseStore(lease *coordv1.Lease) error {
	if lease == nil {
		return nil
	}
	newElems := make([]interface{}, 0, 1)
	newElems = append(newElems, lease)
	return nc.leaseInformer.Informer().GetStore().Replace(newElems, "newRV")
}

func (nc *nodeLifecycleController) syncNodeStore(fakeNodeHandler *testutil.FakeNodeHandler) error {
	nodes, err := fakeNodeHandler.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return err
	}
	newElems := make([]interface{}, 0, len(nodes.Items))
	for i := range nodes.Items {
		newElems = append(newElems, &nodes.Items[i])
	}
	return nc.nodeInformer.Informer().GetStore().Replace(newElems, "newRV")
}

func newNodeLifecycleControllerFromClient(
	kubeClient clientset.Interface,
	podEvictionTimeout time.Duration,
	evictionLimiterQPS float32,
	secondaryEvictionLimiterQPS float32,
	largeClusterThreshold int32,
	unhealthyZoneThreshold float32,
	nodeMonitorGracePeriod time.Duration,
	nodeStartupGracePeriod time.Duration,
	nodeMonitorPeriod time.Duration,
	useTaints bool,
) (*nodeLifecycleController, error) {

	factory := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())

	leaseInformer := factory.Coordination().V1().Leases()
	nodeInformer := factory.Core().V1().Nodes()
	daemonSetInformer := factory.Apps().V1().DaemonSets()

	nc, err := NewNodeLifecycleController(
		leaseInformer,
		factory.Core().V1().Pods(),
		nodeInformer,
		daemonSetInformer,
		kubeClient,
		nodeMonitorPeriod,
		nodeStartupGracePeriod,
		nodeMonitorGracePeriod,
		podEvictionTimeout,
		evictionLimiterQPS,
		secondaryEvictionLimiterQPS,
		largeClusterThreshold,
		unhealthyZoneThreshold,
		useTaints,
	)
	if err != nil {
		return nil, err
	}

	nc.leaseInformerSynced = alwaysReady
	nc.podInformerSynced = alwaysReady
	nc.nodeInformerSynced = alwaysReady
	nc.daemonSetInformerSynced = alwaysReady

	return &nodeLifecycleController{nc, leaseInformer, nodeInformer, daemonSetInformer}, nil
}

func TestMonitorNodeHealthEvictPods(t *testing.T) {
	fakeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute
	labels := map[string]string{
		v1.LabelZoneRegionStable:        "region1",
		v1.LabelZoneFailureDomainStable: "zone1",
		v1.LabelZoneRegion:              "region1",
		v1.LabelZoneFailureDomain:       "zone1",
	}

	// Because of the logic that prevents NC from evicting anything when all Nodes are NotReady
	// we need second healthy node in tests. Because of how the tests are written we need to update
	// the status of this Node.
	healthyNodeNewStatus := v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:   v1.NodeReady,
				Status: v1.ConditionTrue,
				// Node status has just been updated, and is NotReady for 10min.
				LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 9, 0, 0, time.UTC),
				LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}

	table := []struct {
		fakeNodeHandler     *testutil.FakeNodeHandler
		daemonSets          []apps.DaemonSet
		timeToPass          time.Duration
		newNodeStatus       v1.NodeStatus
		secondNodeNewStatus v1.NodeStatus
		expectedEvictPods   bool
		description         string
	}{
		// Node created recently, with no status (happens only at cluster startup).
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: fakeNow,
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionTrue,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			daemonSets:          nil,
			timeToPass:          0,
			newNodeStatus:       v1.NodeStatus{},
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   false,
			description:         "Node created recently, with no status.",
		},
		// Node created recently without FailureDomain labels which is added back later, with no status (happens only at cluster startup).
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: fakeNow,
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionTrue,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			daemonSets:          nil,
			timeToPass:          0,
			newNodeStatus:       v1.NodeStatus{},
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   false,
			description:         "Node created recently without FailureDomain labels which is added back later, with no status (happens only at cluster startup).",
		},
		// Node created long time ago, and kubelet posted NotReady for a short period of time.
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionFalse,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionTrue,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			daemonSets: nil,
			timeToPass: evictionTimeout,
			newNodeStatus: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{
						Type:   v1.NodeReady,
						Status: v1.ConditionFalse,
						// Node status has just been updated, and is NotReady for 10min.
						LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 9, 0, 0, time.UTC),
						LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   false,
			description:         "Node created long time ago, and kubelet posted NotReady for a short period of time.",
		},
		// Pod is ds-managed, and kubelet posted NotReady for a long period of time.
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionFalse,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionTrue,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(
					&v1.PodList{
						Items: []v1.Pod{
							{
								ObjectMeta: metav1.ObjectMeta{
									Name:      "pod0",
									Namespace: "default",
									Labels:    map[string]string{"daemon": "yes"},
								},
								Spec: v1.PodSpec{
									NodeName: "node0",
								},
							},
						},
					},
				),
			},
			daemonSets: []apps.DaemonSet{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "ds0",
						Namespace: "default",
					},
					Spec: apps.DaemonSetSpec{
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"daemon": "yes"},
						},
					},
				},
			},
			timeToPass: time.Hour,
			newNodeStatus: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{
						Type:   v1.NodeReady,
						Status: v1.ConditionFalse,
						// Node status has just been updated, and is NotReady for 1hr.
						LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 59, 0, 0, time.UTC),
						LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   false,
			description:         "Pod is ds-managed, and kubelet posted NotReady for a long period of time.",
		},
		// Node created long time ago, and kubelet posted NotReady for a long period of time.
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionFalse,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionTrue,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			daemonSets: nil,
			timeToPass: time.Hour,
			newNodeStatus: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{
						Type:   v1.NodeReady,
						Status: v1.ConditionFalse,
						// Node status has just been updated, and is NotReady for 1hr.
						LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 59, 0, 0, time.UTC),
						LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   true,
			description:         "Node created long time ago, and kubelet posted NotReady for a long period of time.",
		},
		// Node created long time ago, node controller posted Unknown for a short period of time.
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionUnknown,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionTrue,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			daemonSets: nil,
			timeToPass: evictionTimeout - testNodeMonitorGracePeriod,
			newNodeStatus: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{
						Type:   v1.NodeReady,
						Status: v1.ConditionUnknown,
						// Node status was updated by nodecontroller 10min ago
						LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   false,
			description:         "Node created long time ago, node controller posted Unknown for a short period of time.",
		},
		// Node created long time ago, node controller posted Unknown for a long period of time.
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionUnknown,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionTrue,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			daemonSets: nil,
			timeToPass: 60 * time.Minute,
			newNodeStatus: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{
						Type:   v1.NodeReady,
						Status: v1.ConditionUnknown,
						// Node status was updated by nodecontroller 1hr ago
						LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
			},
			secondNodeNewStatus: healthyNodeNewStatus,
			expectedEvictPods:   true,
			description:         "Node created long time ago, node controller posted Unknown for a long period of time.",
		},
	}

	for _, item := range table {
		nodeController, _ := newNodeLifecycleControllerFromClient(
			item.fakeNodeHandler,
			evictionTimeout,
			testRateLimiterQPS,
			testRateLimiterQPS,
			testLargeClusterThreshold,
			testUnhealthyThreshold,
			testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod,
			testNodeMonitorPeriod,
			false)
		nodeController.now = func() metav1.Time { return fakeNow }
		nodeController.recorder = testutil.NewFakeRecorder()
		nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(item.fakeNodeHandler.Clientset)
		for _, ds := range item.daemonSets {
			nodeController.daemonSetInformer.Informer().GetStore().Add(&ds)
		}
		if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
			item.fakeNodeHandler.Existing[1].Status = item.secondNodeNewStatus
		}
		if len(item.fakeNodeHandler.Existing[0].Labels) == 0 && len(item.fakeNodeHandler.Existing[1].Labels) == 0 {
			item.fakeNodeHandler.Existing[0].Labels = labels
			item.fakeNodeHandler.Existing[1].Labels = labels
		}
		if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		zones := testutil.GetZones(item.fakeNodeHandler)
		for _, zone := range zones {
			if _, ok := nodeController.zonePodEvictor[zone]; ok {
				nodeController.zonePodEvictor[zone].Try(func(value scheduler.TimedValue) (bool, time.Duration) {
					nodeUID, _ := value.UID.(string)
					pods, err := nodeController.getPodsAssignedToNode(value.Value)
					if err != nil {
						t.Errorf("unexpected error: %v", err)
					}
					t.Logf("listed pods %d for node %v", len(pods), value.Value)
					nodeutil.DeletePods(item.fakeNodeHandler, pods, nodeController.recorder, value.Value, nodeUID, nodeController.daemonSetInformer.Lister())
					return true, 0
				})
			} else {
				t.Fatalf("Zone %v was uninitialized!", zone)
			}
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
	fakeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute

	// Because of the logic that prevents NC from evicting anything when all Nodes are NotReady
	// we need second healthy node in tests. Because of how the tests are written we need to update
	// the status of this Node.
	healthyNodeNewStatus := v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:   v1.NodeReady,
				Status: v1.ConditionTrue,
				// Node status has just been updated, and is NotReady for 10min.
				LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 9, 0, 0, time.UTC),
				LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}

	// Node created long time ago, node controller posted Unknown for a long period of time.
	table := []struct {
		fakeNodeHandler     *testutil.FakeNodeHandler
		timeToPass          time.Duration
		newNodeStatus       v1.NodeStatus
		secondNodeNewStatus v1.NodeStatus
		expectedPodUpdate   bool
		expectedReason      string
		description         string
	}{
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegionStable:        "region1",
								v1.LabelZoneFailureDomainStable: "zone1",
								v1.LabelZoneRegion:              "region1",
								v1.LabelZoneFailureDomain:       "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionUnknown,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node1",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
							Labels: map[string]string{
								v1.LabelZoneRegion:        "region1",
								v1.LabelZoneFailureDomain: "zone1",
							},
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionTrue,
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			timeToPass: 60 * time.Minute,
			newNodeStatus: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{
						Type:   v1.NodeReady,
						Status: v1.ConditionUnknown,
						// Node status was updated by nodecontroller 1hr ago
						LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
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
		nodeController, _ := newNodeLifecycleControllerFromClient(
			item.fakeNodeHandler,
			evictionTimeout,
			testRateLimiterQPS,
			testRateLimiterQPS,
			testLargeClusterThreshold,
			testUnhealthyThreshold,
			testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod,
			testNodeMonitorPeriod,
			false)
		nodeController.now = func() metav1.Time { return fakeNow }
		nodeController.recorder = testutil.NewFakeRecorder()
		nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(item.fakeNodeHandler.Clientset)
		if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
			item.fakeNodeHandler.Existing[1].Status = item.secondNodeNewStatus
		}
		if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		zones := testutil.GetZones(item.fakeNodeHandler)
		for _, zone := range zones {
			nodeController.zonePodEvictor[zone].Try(func(value scheduler.TimedValue) (bool, time.Duration) {
				nodeUID, _ := value.UID.(string)
				pods, err := nodeController.getPodsAssignedToNode(value.Value)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				nodeutil.DeletePods(item.fakeNodeHandler, pods, nodeController.recorder, value.Value, nodeUID, nodeController.daemonSetStore)
				return true, 0
			})
		}

		podReasonUpdate := false
		for _, action := range item.fakeNodeHandler.Actions() {
			if action.GetVerb() == "update" && action.GetResource().Resource == "pods" {
				updateReason := action.(testcore.UpdateActionImpl).GetObject().(*v1.Pod).Status.Reason
				podReasonUpdate = true
				if updateReason != item.expectedReason {
					t.Errorf("expected pod status reason: %+v, got %+v for %+v", item.expectedReason, updateReason, item.description)
				}
			}
		}

		if podReasonUpdate != item.expectedPodUpdate {
			t.Errorf("expected pod update: %+v, got %+v for %+v", item.expectedPodUpdate, podReasonUpdate, item.description)
		}
	}
}

func TestMonitorNodeHealthEvictPodsWithDisruption(t *testing.T) {
	fakeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute
	timeToPass := 60 * time.Minute

	// Because of the logic that prevents NC from evicting anything when all Nodes are NotReady
	// we need second healthy node in tests. Because of how the tests are written we need to update
	// the status of this Node.
	healthyNodeNewStatus := v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:               v1.NodeReady,
				Status:             v1.ConditionTrue,
				LastHeartbeatTime:  metav1.Date(2015, 1, 1, 13, 0, 0, 0, time.UTC),
				LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}
	unhealthyNodeNewStatus := v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:   v1.NodeReady,
				Status: v1.ConditionUnknown,
				// Node status was updated by nodecontroller 1hr ago
				LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
				LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}

	table := []struct {
		nodeList                []*v1.Node
		podList                 []v1.Pod
		updatedNodeStatuses     []v1.NodeStatus
		expectedInitialStates   map[string]ZoneState
		expectedFollowingStates map[string]ZoneState
		expectedEvictPods       bool
		description             string
	}{
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period of time on both Nodes.
		// Only zone is down - eviction shouldn't take place
		{
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node1",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
			},
			podList: []v1.Pod{*testutil.NewPod("pod0", "node0")},
			updatedNodeStatuses: []v1.NodeStatus{
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
			},
			expectedInitialStates:   map[string]ZoneState{testutil.CreateZoneID("region1", "zone1"): stateFullDisruption},
			expectedFollowingStates: map[string]ZoneState{testutil.CreateZoneID("region1", "zone1"): stateFullDisruption},
			expectedEvictPods:       false,
			description:             "Network Disruption: Only zone is down - eviction shouldn't take place.",
		},
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period of time on both Nodes.
		// Both zones down - eviction shouldn't take place
		{
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node1",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region2",
							v1.LabelZoneFailureDomainStable: "zone2",
							v1.LabelZoneRegion:              "region2",
							v1.LabelZoneFailureDomain:       "zone2",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
			},

			podList: []v1.Pod{*testutil.NewPod("pod0", "node0")},
			updatedNodeStatuses: []v1.NodeStatus{
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
			},
			expectedInitialStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateFullDisruption,
				testutil.CreateZoneID("region2", "zone2"): stateFullDisruption,
			},
			expectedFollowingStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateFullDisruption,
				testutil.CreateZoneID("region2", "zone2"): stateFullDisruption,
			},
			expectedEvictPods: false,
			description:       "Network Disruption: Both zones down - eviction shouldn't take place.",
		},
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period of time on both Nodes.
		// One zone is down - eviction should take place
		{
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node1",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone2",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone2",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionTrue,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
			},
			podList: []v1.Pod{*testutil.NewPod("pod0", "node0")},
			updatedNodeStatuses: []v1.NodeStatus{
				unhealthyNodeNewStatus,
				healthyNodeNewStatus,
			},
			expectedInitialStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateFullDisruption,
				testutil.CreateZoneID("region1", "zone2"): stateNormal,
			},
			expectedFollowingStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateFullDisruption,
				testutil.CreateZoneID("region1", "zone2"): stateNormal,
			},
			expectedEvictPods: true,
			description:       "Network Disruption: One zone is down - eviction should take place.",
		},
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period
		// of on first Node, eviction should stop even though -master Node is healthy.
		{
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node-master",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionTrue,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
			},
			podList: []v1.Pod{*testutil.NewPod("pod0", "node0")},
			updatedNodeStatuses: []v1.NodeStatus{
				unhealthyNodeNewStatus,
				healthyNodeNewStatus,
			},
			expectedInitialStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateFullDisruption,
			},
			expectedFollowingStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateFullDisruption,
			},
			expectedEvictPods: false,
			description:       "NetworkDisruption: eviction should stop, only -master Node is healthy",
		},
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period of time on both Nodes.
		// Initially both zones down, one comes back - eviction should take place
		{
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node1",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone2",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone2",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
			},

			podList: []v1.Pod{*testutil.NewPod("pod0", "node0")},
			updatedNodeStatuses: []v1.NodeStatus{
				unhealthyNodeNewStatus,
				healthyNodeNewStatus,
			},
			expectedInitialStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateFullDisruption,
				testutil.CreateZoneID("region1", "zone2"): stateFullDisruption,
			},
			expectedFollowingStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateFullDisruption,
				testutil.CreateZoneID("region1", "zone2"): stateNormal,
			},
			expectedEvictPods: true,
			description:       "Initially both zones down, one comes back - eviction should take place",
		},
		// NetworkDisruption: Node created long time ago, node controller posted Unknown for a long period of time on both Nodes.
		// Zone is partially disrupted - eviction should take place
		{
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node1",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node2",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node3",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionTrue,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node4",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						Labels: map[string]string{
							v1.LabelZoneRegionStable:        "region1",
							v1.LabelZoneFailureDomainStable: "zone1",
							v1.LabelZoneRegion:              "region1",
							v1.LabelZoneFailureDomain:       "zone1",
						},
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionTrue,
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							},
						},
					},
				},
			},

			podList: []v1.Pod{*testutil.NewPod("pod0", "node0")},
			updatedNodeStatuses: []v1.NodeStatus{
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
				healthyNodeNewStatus,
				healthyNodeNewStatus,
			},
			expectedInitialStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): statePartialDisruption,
			},
			expectedFollowingStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): statePartialDisruption,
			},
			expectedEvictPods: true,
			description:       "Zone is partially disrupted - eviction should take place.",
		},
	}

	for _, item := range table {
		fakeNodeHandler := &testutil.FakeNodeHandler{
			Existing:  item.nodeList,
			Clientset: fake.NewSimpleClientset(&v1.PodList{Items: item.podList}),
		}
		nodeController, _ := newNodeLifecycleControllerFromClient(
			fakeNodeHandler,
			evictionTimeout,
			testRateLimiterQPS,
			testRateLimiterQPS,
			testLargeClusterThreshold,
			testUnhealthyThreshold,
			testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod,
			testNodeMonitorPeriod,
			false)
		nodeController.now = func() metav1.Time { return fakeNow }
		nodeController.recorder = testutil.NewFakeRecorder()
		nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)
		nodeController.enterPartialDisruptionFunc = func(nodeNum int) float32 {
			return testRateLimiterQPS
		}
		nodeController.enterFullDisruptionFunc = func(nodeNum int) float32 {
			return testRateLimiterQPS
		}
		if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(); err != nil {
			t.Errorf("%v: unexpected error: %v", item.description, err)
		}

		for zone, state := range item.expectedInitialStates {
			if state != nodeController.zoneStates[zone] {
				t.Errorf("%v: Unexpected zone state: %v: %v instead %v", item.description, zone, nodeController.zoneStates[zone], state)
			}
		}

		nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Add(timeToPass)} }
		for i := range item.updatedNodeStatuses {
			fakeNodeHandler.Existing[i].Status = item.updatedNodeStatuses[i]
		}

		if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(); err != nil {
			t.Errorf("%v: unexpected error: %v", item.description, err)
		}
		for zone, state := range item.expectedFollowingStates {
			if state != nodeController.zoneStates[zone] {
				t.Errorf("%v: Unexpected zone state: %v: %v instead %v", item.description, zone, nodeController.zoneStates[zone], state)
			}
		}
		var podEvicted bool
		start := time.Now()
		// Infinite loop, used for retrying in case ratelimiter fails to reload for Try function.
		// this breaks when we have the status that we need for test case or when we don't see the
		// intended result after 1 minute.
		for {
			podEvicted = nodeController.doEviction(fakeNodeHandler)
			if podEvicted == item.expectedEvictPods || time.Since(start) > 1*time.Minute {
				break
			}
		}
		if item.expectedEvictPods != podEvicted {
			t.Errorf("%v: expected pod eviction: %+v, got %+v", item.description, item.expectedEvictPods, podEvicted)
		}
	}
}

func TestMonitorNodeHealthUpdateStatus(t *testing.T) {
	fakeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	table := []struct {
		fakeNodeHandler         *testutil.FakeNodeHandler
		timeToPass              time.Duration
		newNodeStatus           v1.NodeStatus
		expectedRequestCount    int
		expectedNodes           []*v1.Node
		expectedPodStatusUpdate bool
	}{
		// Node created long time ago, without status:
		// Expect Unknown status posted from node controller.
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			expectedRequestCount: 2, // List+Update
			expectedNodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
								LastTransitionTime: fakeNow,
							},
							{
								Type:               v1.NodeMemoryPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
								LastTransitionTime: fakeNow,
							},
							{
								Type:               v1.NodeDiskPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
								LastTransitionTime: fakeNow,
							},
							{
								Type:               v1.NodePIDPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
								LastTransitionTime: fakeNow,
							},
						},
					},
				},
			},
			expectedPodStatusUpdate: false, // Pod was never scheduled
		},
		// Node created recently, without status.
		// Expect no action from node controller (within startup grace period).
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: fakeNow,
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			expectedRequestCount:    1, // List
			expectedNodes:           nil,
			expectedPodStatusUpdate: false,
		},
		// Node created long time ago, with status updated by kubelet exceeds grace period.
		// Expect Unknown status posted from node controller.
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:   v1.NodeReady,
									Status: v1.ConditionTrue,
									// Node status hasn't been updated for 1hr.
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
							Capacity: v1.ResourceList{
								v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
								v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			expectedRequestCount: 3, // (List+)List+Update
			timeToPass:           time.Hour,
			newNodeStatus: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{
						Type:   v1.NodeReady,
						Status: v1.ConditionTrue,
						// Node status hasn't been updated for 1hr.
						LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
				},
			},
			expectedNodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusUnknown",
								Message:            "Kubelet stopped posting node status.",
								LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								LastTransitionTime: metav1.Time{Time: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC).Add(time.Hour)},
							},
							{
								Type:               v1.NodeMemoryPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC), // should default to node creation time if condition was never updated
								LastTransitionTime: metav1.Time{Time: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC).Add(time.Hour)},
							},
							{
								Type:               v1.NodeDiskPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC), // should default to node creation time if condition was never updated
								LastTransitionTime: metav1.Time{Time: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC).Add(time.Hour)},
							},
							{
								Type:               v1.NodePIDPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC), // should default to node creation time if condition was never updated
								LastTransitionTime: metav1.Time{Time: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC).Add(time.Hour)},
							},
						},
						Capacity: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
						},
					},
				},
			},
			expectedPodStatusUpdate: true,
		},
		// Node created long time ago, with status updated recently.
		// Expect no action from node controller (within monitor grace period).
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:   v1.NodeReady,
									Status: v1.ConditionTrue,
									// Node status has just been updated.
									LastHeartbeatTime:  fakeNow,
									LastTransitionTime: fakeNow,
								},
							},
							Capacity: v1.ResourceList{
								v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
								v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			expectedRequestCount:    1, // List
			expectedNodes:           nil,
			expectedPodStatusUpdate: false,
		},
	}
	for i, item := range table {
		nodeController, _ := newNodeLifecycleControllerFromClient(
			item.fakeNodeHandler,
			5*time.Minute,
			testRateLimiterQPS,
			testRateLimiterQPS,
			testLargeClusterThreshold,
			testUnhealthyThreshold,
			testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod,
			testNodeMonitorPeriod,
			false)
		nodeController.now = func() metav1.Time { return fakeNow }
		nodeController.recorder = testutil.NewFakeRecorder()
		nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(item.fakeNodeHandler.Clientset)
		if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
			if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if err := nodeController.monitorNodeHealth(); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		}
		if item.expectedRequestCount != item.fakeNodeHandler.RequestCount {
			t.Errorf("expected %v call, but got %v.", item.expectedRequestCount, item.fakeNodeHandler.RequestCount)
		}
		if len(item.fakeNodeHandler.UpdatedNodes) > 0 && !apiequality.Semantic.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodes) {
			t.Errorf("Case[%d] unexpected nodes: %s", i, diff.ObjectDiff(item.expectedNodes[0], item.fakeNodeHandler.UpdatedNodes[0]))
		}
		if len(item.fakeNodeHandler.UpdatedNodeStatuses) > 0 && !apiequality.Semantic.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodeStatuses) {
			t.Errorf("Case[%d] unexpected nodes: %s", i, diff.ObjectDiff(item.expectedNodes[0], item.fakeNodeHandler.UpdatedNodeStatuses[0]))
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

func TestMonitorNodeHealthUpdateNodeAndPodStatusWithLease(t *testing.T) {
	nodeCreationTime := metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC)
	fakeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	testcases := []struct {
		description             string
		fakeNodeHandler         *testutil.FakeNodeHandler
		lease                   *coordv1.Lease
		timeToPass              time.Duration
		newNodeStatus           v1.NodeStatus
		newLease                *coordv1.Lease
		expectedRequestCount    int
		expectedNodes           []*v1.Node
		expectedPodStatusUpdate bool
	}{
		// Node created recently, without status. Node lease is missing.
		// Expect no action from node controller (within startup grace period).
		{
			description: "Node created recently, without status. Node lease is missing.",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: fakeNow,
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			expectedRequestCount:    1, // List
			expectedNodes:           nil,
			expectedPodStatusUpdate: false,
		},
		// Node created recently, without status. Node lease is renewed recently.
		// Expect no action from node controller (within startup grace period).
		{
			description: "Node created recently, without status. Node lease is renewed recently.",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: fakeNow,
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			lease:                   createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time)),
			expectedRequestCount:    1, // List
			expectedNodes:           nil,
			expectedPodStatusUpdate: false,
		},
		// Node created long time ago, without status. Node lease is missing.
		// Expect Unknown status posted from node controller.
		{
			description: "Node created long time ago, without status. Node lease is missing.",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: nodeCreationTime,
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			expectedRequestCount: 2, // List+Update
			expectedNodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: nodeCreationTime,
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  nodeCreationTime,
								LastTransitionTime: fakeNow,
							},
							{
								Type:               v1.NodeMemoryPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  nodeCreationTime,
								LastTransitionTime: fakeNow,
							},
							{
								Type:               v1.NodeDiskPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  nodeCreationTime,
								LastTransitionTime: fakeNow,
							},
							{
								Type:               v1.NodePIDPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  nodeCreationTime,
								LastTransitionTime: fakeNow,
							},
						},
					},
				},
			},
			expectedPodStatusUpdate: false, // Pod was never scheduled because the node was never ready.
		},
		// Node created long time ago, without status. Node lease is renewed recently.
		// Expect no action from node controller (within monitor grace period).
		{
			description: "Node created long time ago, without status. Node lease is renewed recently.",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: nodeCreationTime,
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			lease:                createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time)),
			timeToPass:           time.Hour,
			newLease:             createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time.Add(time.Hour))), // Lease is renewed after 1 hour.
			expectedRequestCount: 2,                                                                          // List+List
			expectedNodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: nodeCreationTime,
					},
				},
			},
			expectedPodStatusUpdate: false,
		},
		// Node created long time ago, without status. Node lease is expired.
		// Expect Unknown status posted from node controller.
		{
			description: "Node created long time ago, without status. Node lease is expired.",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: nodeCreationTime,
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			lease:                createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time)),
			timeToPass:           time.Hour,
			newLease:             createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time)), // Lease is not renewed after 1 hour.
			expectedRequestCount: 3,                                                           // List+List+Update
			expectedNodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: nodeCreationTime,
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  nodeCreationTime,
								LastTransitionTime: metav1.Time{Time: fakeNow.Add(time.Hour)},
							},
							{
								Type:               v1.NodeMemoryPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  nodeCreationTime,
								LastTransitionTime: metav1.Time{Time: fakeNow.Add(time.Hour)},
							},
							{
								Type:               v1.NodeDiskPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  nodeCreationTime,
								LastTransitionTime: metav1.Time{Time: fakeNow.Add(time.Hour)},
							},
							{
								Type:               v1.NodePIDPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  nodeCreationTime,
								LastTransitionTime: metav1.Time{Time: fakeNow.Add(time.Hour)},
							},
						},
					},
				},
			},
			expectedPodStatusUpdate: false,
		},
		// Node created long time ago, with status updated by kubelet exceeds grace period. Node lease is renewed.
		// Expect no action from node controller (within monitor grace period).
		{
			description: "Node created long time ago, with status updated by kubelet exceeds grace period. Node lease is renewed.",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: nodeCreationTime,
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionTrue,
									LastHeartbeatTime:  fakeNow,
									LastTransitionTime: fakeNow,
								},
								{
									Type:               v1.NodeDiskPressure,
									Status:             v1.ConditionFalse,
									LastHeartbeatTime:  fakeNow,
									LastTransitionTime: fakeNow,
								},
							},
							Capacity: v1.ResourceList{
								v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
								v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			lease:                createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time)),
			expectedRequestCount: 2, // List+List
			timeToPass:           time.Hour,
			newNodeStatus: v1.NodeStatus{
				// Node status hasn't been updated for 1 hour.
				Conditions: []v1.NodeCondition{
					{
						Type:               v1.NodeReady,
						Status:             v1.ConditionTrue,
						LastHeartbeatTime:  fakeNow,
						LastTransitionTime: fakeNow,
					},
					{
						Type:               v1.NodeDiskPressure,
						Status:             v1.ConditionFalse,
						LastHeartbeatTime:  fakeNow,
						LastTransitionTime: fakeNow,
					},
				},
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
				},
			},
			newLease: createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time.Add(time.Hour))), // Lease is renewed after 1 hour.
			expectedNodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: nodeCreationTime,
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionTrue,
								LastHeartbeatTime:  fakeNow,
								LastTransitionTime: fakeNow,
							},
							{
								Type:               v1.NodeDiskPressure,
								Status:             v1.ConditionFalse,
								LastHeartbeatTime:  fakeNow,
								LastTransitionTime: fakeNow,
							},
						},
						Capacity: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
						},
					},
				},
			},
			expectedPodStatusUpdate: false,
		},
		// Node created long time ago, with status updated by kubelet recently. Node lease is expired.
		// Expect no action from node controller (within monitor grace period).
		{
			description: "Node created long time ago, with status updated by kubelet recently. Node lease is expired.",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: nodeCreationTime,
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionTrue,
									LastHeartbeatTime:  fakeNow,
									LastTransitionTime: fakeNow,
								},
								{
									Type:               v1.NodeDiskPressure,
									Status:             v1.ConditionFalse,
									LastHeartbeatTime:  fakeNow,
									LastTransitionTime: fakeNow,
								},
							},
							Capacity: v1.ResourceList{
								v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
								v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			lease:                createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time)),
			expectedRequestCount: 2, // List+List
			timeToPass:           time.Hour,
			newNodeStatus: v1.NodeStatus{
				// Node status is updated after 1 hour.
				Conditions: []v1.NodeCondition{
					{
						Type:               v1.NodeReady,
						Status:             v1.ConditionTrue,
						LastHeartbeatTime:  metav1.Time{Time: fakeNow.Add(time.Hour)},
						LastTransitionTime: fakeNow,
					},
					{
						Type:               v1.NodeDiskPressure,
						Status:             v1.ConditionFalse,
						LastHeartbeatTime:  metav1.Time{Time: fakeNow.Add(time.Hour)},
						LastTransitionTime: fakeNow,
					},
				},
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
				},
			},
			newLease: createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time)), // Lease is not renewed after 1 hour.
			expectedNodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: nodeCreationTime,
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionTrue,
								LastHeartbeatTime:  metav1.Time{Time: fakeNow.Add(time.Hour)},
								LastTransitionTime: fakeNow,
							},
							{
								Type:               v1.NodeDiskPressure,
								Status:             v1.ConditionFalse,
								LastHeartbeatTime:  metav1.Time{Time: fakeNow.Add(time.Hour)},
								LastTransitionTime: fakeNow,
							},
						},
						Capacity: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
						},
					},
				},
			},
			expectedPodStatusUpdate: false,
		},
		// Node created long time ago, with status updated by kubelet exceeds grace period. Node lease is also expired.
		// Expect Unknown status posted from node controller.
		{
			description: "Node created long time ago, with status updated by kubelet exceeds grace period. Node lease is also expired.",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: nodeCreationTime,
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionTrue,
									LastHeartbeatTime:  fakeNow,
									LastTransitionTime: fakeNow,
								},
							},
							Capacity: v1.ResourceList{
								v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
								v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			lease:                createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time)),
			expectedRequestCount: 3, // List+List+Update
			timeToPass:           time.Hour,
			newNodeStatus: v1.NodeStatus{
				// Node status hasn't been updated for 1 hour.
				Conditions: []v1.NodeCondition{
					{
						Type:               v1.NodeReady,
						Status:             v1.ConditionTrue,
						LastHeartbeatTime:  fakeNow,
						LastTransitionTime: fakeNow,
					},
				},
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
				},
			},
			newLease: createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time)), // Lease is not renewed after 1 hour.
			expectedNodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: nodeCreationTime,
					},
					Status: v1.NodeStatus{
						Conditions: []v1.NodeCondition{
							{
								Type:               v1.NodeReady,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusUnknown",
								Message:            "Kubelet stopped posting node status.",
								LastHeartbeatTime:  fakeNow,
								LastTransitionTime: metav1.Time{Time: fakeNow.Add(time.Hour)},
							},
							{
								Type:               v1.NodeMemoryPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  nodeCreationTime, // should default to node creation time if condition was never updated
								LastTransitionTime: metav1.Time{Time: fakeNow.Add(time.Hour)},
							},
							{
								Type:               v1.NodeDiskPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  nodeCreationTime, // should default to node creation time if condition was never updated
								LastTransitionTime: metav1.Time{Time: fakeNow.Add(time.Hour)},
							},
							{
								Type:               v1.NodePIDPressure,
								Status:             v1.ConditionUnknown,
								Reason:             "NodeStatusNeverUpdated",
								Message:            "Kubelet never posted node status.",
								LastHeartbeatTime:  nodeCreationTime, // should default to node creation time if condition was never updated
								LastTransitionTime: metav1.Time{Time: fakeNow.Add(time.Hour)},
							},
						},
						Capacity: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
						},
					},
				},
			},
			expectedPodStatusUpdate: true,
		},
	}

	for _, item := range testcases {
		t.Run(item.description, func(t *testing.T) {
			nodeController, _ := newNodeLifecycleControllerFromClient(
				item.fakeNodeHandler,
				5*time.Minute,
				testRateLimiterQPS,
				testRateLimiterQPS,
				testLargeClusterThreshold,
				testUnhealthyThreshold,
				testNodeMonitorGracePeriod,
				testNodeStartupGracePeriod,
				testNodeMonitorPeriod,
				false)
			nodeController.now = func() metav1.Time { return fakeNow }
			nodeController.recorder = testutil.NewFakeRecorder()
			nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(item.fakeNodeHandler.Clientset)
			if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if err := nodeController.syncLeaseStore(item.lease); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if err := nodeController.monitorNodeHealth(); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if item.timeToPass > 0 {
				nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Add(item.timeToPass)} }
				item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
				if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if err := nodeController.syncLeaseStore(item.newLease); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if err := nodeController.monitorNodeHealth(); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}
			if item.expectedRequestCount != item.fakeNodeHandler.RequestCount {
				t.Errorf("expected %v call, but got %v.", item.expectedRequestCount, item.fakeNodeHandler.RequestCount)
			}
			if len(item.fakeNodeHandler.UpdatedNodes) > 0 && !apiequality.Semantic.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodes) {
				t.Errorf("unexpected nodes: %s", diff.ObjectDiff(item.expectedNodes[0], item.fakeNodeHandler.UpdatedNodes[0]))
			}
			if len(item.fakeNodeHandler.UpdatedNodeStatuses) > 0 && !apiequality.Semantic.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodeStatuses) {
				t.Errorf("unexpected nodes: %s", diff.ObjectDiff(item.expectedNodes[0], item.fakeNodeHandler.UpdatedNodeStatuses[0]))
			}

			podStatusUpdated := false
			for _, action := range item.fakeNodeHandler.Actions() {
				if action.GetVerb() == "update" && action.GetResource().Resource == "pods" && action.GetSubresource() == "status" {
					podStatusUpdated = true
				}
			}
			if podStatusUpdated != item.expectedPodStatusUpdate {
				t.Errorf("expect pod status updated to be %v, but got %v", item.expectedPodStatusUpdate, podStatusUpdated)
			}
		})
	}
}

func TestMonitorNodeHealthMarkPodsNotReady(t *testing.T) {
	fakeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	table := []struct {
		fakeNodeHandler         *testutil.FakeNodeHandler
		timeToPass              time.Duration
		newNodeStatus           v1.NodeStatus
		expectedPodStatusUpdate bool
	}{
		// Node created recently, without status.
		// Expect no action from node controller (within startup grace period).
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: fakeNow,
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			expectedPodStatusUpdate: false,
		},
		// Node created long time ago, with status updated recently.
		// Expect no action from node controller (within monitor grace period).
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:   v1.NodeReady,
									Status: v1.ConditionTrue,
									// Node status has just been updated.
									LastHeartbeatTime:  fakeNow,
									LastTransitionTime: fakeNow,
								},
							},
							Capacity: v1.ResourceList{
								v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
								v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			expectedPodStatusUpdate: false,
		},
		// Node created long time ago, with status updated by kubelet exceeds grace period.
		// Expect pods status updated and Unknown node status posted from node controller
		{
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:   v1.NodeReady,
									Status: v1.ConditionTrue,
									// Node status hasn't been updated for 1hr.
									LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
									LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
								},
							},
							Capacity: v1.ResourceList{
								v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
								v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			timeToPass: 1 * time.Minute,
			newNodeStatus: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{
						Type:   v1.NodeReady,
						Status: v1.ConditionTrue,
						// Node status hasn't been updated for 1hr.
						LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					},
				},
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
				},
			},
			expectedPodStatusUpdate: true,
		},
	}

	for i, item := range table {
		nodeController, _ := newNodeLifecycleControllerFromClient(
			item.fakeNodeHandler,
			5*time.Minute,
			testRateLimiterQPS,
			testRateLimiterQPS,
			testLargeClusterThreshold,
			testUnhealthyThreshold,
			testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod,
			testNodeMonitorPeriod,
			false)
		nodeController.now = func() metav1.Time { return fakeNow }
		nodeController.recorder = testutil.NewFakeRecorder()
		nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(item.fakeNodeHandler.Clientset)
		if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(); err != nil {
			t.Errorf("Case[%d] unexpected error: %v", i, err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
			if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if err := nodeController.monitorNodeHealth(); err != nil {
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

func TestMonitorNodeHealthMarkPodsNotReadyRetry(t *testing.T) {
	type nodeIteration struct {
		timeToPass time.Duration
		newNodes   []*v1.Node
	}
	timeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	timePlusTwoMinutes := metav1.Date(2015, 1, 1, 12, 0, 2, 0, time.UTC)
	makeNodes := func(status v1.ConditionStatus, lastHeartbeatTime, lastTransitionTime metav1.Time) []*v1.Node {
		return []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: timeNow,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             status,
							LastHeartbeatTime:  lastHeartbeatTime,
							LastTransitionTime: lastTransitionTime,
						},
					},
				},
			},
		}
	}
	table := []struct {
		desc                      string
		fakeNodeHandler           *testutil.FakeNodeHandler
		updateReactor             func(action testcore.Action) (bool, runtime.Object, error)
		fakeGetPodsAssignedToNode func(c *fake.Clientset) func(string) ([]*v1.Pod, error)
		nodeIterations            []nodeIteration
		expectedPodStatusUpdates  int
	}{
		// Node created long time ago, with status updated by kubelet exceeds grace period.
		// First monitorNodeHealth check will update pod status to NotReady.
		// Second monitorNodeHealth check will do no updates (no retry).
		{
			desc: "successful pod status update, no retry required",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			fakeGetPodsAssignedToNode: fakeGetPodsAssignedToNode,
			nodeIterations: []nodeIteration{
				{
					timeToPass: 0,
					newNodes:   makeNodes(v1.ConditionTrue, timeNow, timeNow),
				},
				{
					timeToPass: 1 * time.Minute,
					newNodes:   makeNodes(v1.ConditionTrue, timeNow, timeNow),
				},
				{
					timeToPass: 1 * time.Minute,
					newNodes:   makeNodes(v1.ConditionFalse, timePlusTwoMinutes, timePlusTwoMinutes),
				},
			},
			expectedPodStatusUpdates: 1,
		},
		// Node created long time ago, with status updated by kubelet exceeds grace period.
		// First monitorNodeHealth check will fail to update pod status to NotReady.
		// Second monitorNodeHealth check will update pod status to NotReady (retry).
		{
			desc: "unsuccessful pod status update, retry required",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			updateReactor: func() func(action testcore.Action) (bool, runtime.Object, error) {
				i := 0
				return func(action testcore.Action) (bool, runtime.Object, error) {
					if action.GetVerb() == "update" && action.GetResource().Resource == "pods" && action.GetSubresource() == "status" {
						i++
						switch i {
						case 1:
							return true, nil, fmt.Errorf("fake error")
						default:
							return true, testutil.NewPod("pod0", "node0"), nil
						}
					}

					return true, nil, fmt.Errorf("unsupported action")
				}
			}(),
			fakeGetPodsAssignedToNode: fakeGetPodsAssignedToNode,
			nodeIterations: []nodeIteration{
				{
					timeToPass: 0,
					newNodes:   makeNodes(v1.ConditionTrue, timeNow, timeNow),
				},
				{
					timeToPass: 1 * time.Minute,
					newNodes:   makeNodes(v1.ConditionTrue, timeNow, timeNow),
				},
				{
					timeToPass: 1 * time.Minute,
					newNodes:   makeNodes(v1.ConditionFalse, timePlusTwoMinutes, timePlusTwoMinutes),
				},
			},
			expectedPodStatusUpdates: 2, // One failed and one retry.
		},
		// Node created long time ago, with status updated by kubelet exceeds grace period.
		// First monitorNodeHealth check will fail to list pods.
		// Second monitorNodeHealth check will update pod status to NotReady (retry).
		{
			desc: "unsuccessful pod list, retry required",
			fakeNodeHandler: &testutil.FakeNodeHandler{
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			fakeGetPodsAssignedToNode: func(c *fake.Clientset) func(string) ([]*v1.Pod, error) {
				i := 0
				f := fakeGetPodsAssignedToNode(c)
				return func(nodeName string) ([]*v1.Pod, error) {
					i++
					if i == 1 {
						return nil, fmt.Errorf("fake error")
					}
					return f(nodeName)
				}
			},
			nodeIterations: []nodeIteration{
				{
					timeToPass: 0,
					newNodes:   makeNodes(v1.ConditionTrue, timeNow, timeNow),
				},
				{
					timeToPass: 1 * time.Minute,
					newNodes:   makeNodes(v1.ConditionTrue, timeNow, timeNow),
				},
				{
					timeToPass: 1 * time.Minute,
					newNodes:   makeNodes(v1.ConditionFalse, timePlusTwoMinutes, timePlusTwoMinutes),
				},
			},
			expectedPodStatusUpdates: 1,
		},
	}

	for _, item := range table {
		t.Run(item.desc, func(t *testing.T) {
			nodeController, _ := newNodeLifecycleControllerFromClient(
				item.fakeNodeHandler,
				5*time.Minute,
				testRateLimiterQPS,
				testRateLimiterQPS,
				testLargeClusterThreshold,
				testUnhealthyThreshold,
				testNodeMonitorGracePeriod,
				testNodeStartupGracePeriod,
				testNodeMonitorPeriod,
				false)
			if item.updateReactor != nil {
				item.fakeNodeHandler.Clientset.PrependReactor("update", "pods", item.updateReactor)
			}
			nodeController.now = func() metav1.Time { return timeNow }
			nodeController.recorder = testutil.NewFakeRecorder()
			nodeController.getPodsAssignedToNode = item.fakeGetPodsAssignedToNode(item.fakeNodeHandler.Clientset)
			for _, itertion := range item.nodeIterations {
				nodeController.now = func() metav1.Time { return metav1.Time{Time: timeNow.Add(itertion.timeToPass)} }
				item.fakeNodeHandler.Existing = itertion.newNodes
				if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if err := nodeController.monitorNodeHealth(); err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}

			podStatusUpdates := 0
			for _, action := range item.fakeNodeHandler.Actions() {
				if action.GetVerb() == "update" && action.GetResource().Resource == "pods" && action.GetSubresource() == "status" {
					podStatusUpdates++
				}
			}
			if podStatusUpdates != item.expectedPodStatusUpdates {
				t.Errorf("expect pod status updated to happen %d times, but got %d", item.expectedPodStatusUpdates, podStatusUpdates)
			}
		})
	}
}

// TestApplyNoExecuteTaints, ensures we just have a NoExecute taint applied to node.
// NodeController is just responsible for enqueuing the node to tainting queue from which taint manager picks up
// and evicts the pods on the node.
func TestApplyNoExecuteTaints(t *testing.T) {
	fakeNow := metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute

	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			// Unreachable Taint with effect 'NoExecute' should be applied to this node.
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegion:              "region1",
						v1.LabelZoneFailureDomain:       "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			// Because of the logic that prevents NC from evicting anything when all Nodes are NotReady
			// we need second healthy node in tests.
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node1",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegion:              "region1",
						v1.LabelZoneFailureDomain:       "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			// NotReady Taint with NoExecute effect should be applied to this node.
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node2",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegion:              "region1",
						v1.LabelZoneFailureDomain:       "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
		},
		Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
	}
	healthyNodeNewStatus := v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:               v1.NodeReady,
				Status:             v1.ConditionTrue,
				LastHeartbeatTime:  metav1.Date(2017, 1, 1, 12, 10, 0, 0, time.UTC),
				LastTransitionTime: metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}
	originalTaint := UnreachableTaintTemplate
	nodeController, _ := newNodeLifecycleControllerFromClient(
		fakeNodeHandler,
		evictionTimeout,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
		true)
	nodeController.now = func() metav1.Time { return fakeNow }
	nodeController.recorder = testutil.NewFakeRecorder()
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass()
	node0, err := fakeNodeHandler.Get(context.TODO(), "node0", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node0...")
		return
	}
	if !taintutils.TaintExists(node0.Spec.Taints, UnreachableTaintTemplate) {
		t.Errorf("Can't find taint %v in %v", originalTaint, node0.Spec.Taints)
	}
	node2, err := fakeNodeHandler.Get(context.TODO(), "node2", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node2...")
		return
	}
	if !taintutils.TaintExists(node2.Spec.Taints, NotReadyTaintTemplate) {
		t.Errorf("Can't find taint %v in %v", NotReadyTaintTemplate, node2.Spec.Taints)
	}

	// Make node3 healthy again.
	node2.Status = healthyNodeNewStatus
	_, err = fakeNodeHandler.UpdateStatus(context.TODO(), node2, metav1.UpdateOptions{})
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass()

	node2, err = fakeNodeHandler.Get(context.TODO(), "node2", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node2...")
		return
	}
	// We should not see any taint on the node(especially the Not-Ready taint with NoExecute effect).
	if taintutils.TaintExists(node2.Spec.Taints, NotReadyTaintTemplate) || len(node2.Spec.Taints) > 0 {
		t.Errorf("Found taint %v in %v, which should not be present", NotReadyTaintTemplate, node2.Spec.Taints)
	}
}

func TestSwapUnreachableNotReadyTaints(t *testing.T) {
	fakeNow := metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute

	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegion:              "region1",
						v1.LabelZoneFailureDomain:       "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			// Because of the logic that prevents NC from evicting anything when all Nodes are NotReady
			// we need second healthy node in tests. Because of how the tests are written we need to update
			// the status of this Node.
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node1",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegion:              "region1",
						v1.LabelZoneFailureDomain:       "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
		},
		Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
	}
	timeToPass := evictionTimeout
	newNodeStatus := v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:   v1.NodeReady,
				Status: v1.ConditionFalse,
				// Node status has just been updated, and is NotReady for 10min.
				LastHeartbeatTime:  metav1.Date(2017, 1, 1, 12, 9, 0, 0, time.UTC),
				LastTransitionTime: metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}
	healthyNodeNewStatus := v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:               v1.NodeReady,
				Status:             v1.ConditionTrue,
				LastHeartbeatTime:  metav1.Date(2017, 1, 1, 12, 10, 0, 0, time.UTC),
				LastTransitionTime: metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}
	originalTaint := UnreachableTaintTemplate
	updatedTaint := NotReadyTaintTemplate

	nodeController, _ := newNodeLifecycleControllerFromClient(
		fakeNodeHandler,
		evictionTimeout,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
		true)
	nodeController.now = func() metav1.Time { return fakeNow }
	nodeController.recorder = testutil.NewFakeRecorder()
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass()

	node0, err := fakeNodeHandler.Get(context.TODO(), "node0", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node0...")
		return
	}
	node1, err := fakeNodeHandler.Get(context.TODO(), "node1", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node1...")
		return
	}

	if originalTaint != nil && !taintutils.TaintExists(node0.Spec.Taints, originalTaint) {
		t.Errorf("Can't find taint %v in %v", originalTaint, node0.Spec.Taints)
	}

	nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Add(timeToPass)} }

	node0.Status = newNodeStatus
	node1.Status = healthyNodeNewStatus
	_, err = fakeNodeHandler.UpdateStatus(context.TODO(), node0, metav1.UpdateOptions{})
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	_, err = fakeNodeHandler.UpdateStatus(context.TODO(), node1, metav1.UpdateOptions{})
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass()

	node0, err = fakeNodeHandler.Get(context.TODO(), "node0", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node0...")
		return
	}
	if updatedTaint != nil {
		if !taintutils.TaintExists(node0.Spec.Taints, updatedTaint) {
			t.Errorf("Can't find taint %v in %v", updatedTaint, node0.Spec.Taints)
		}
	}
}

func TestTaintsNodeByCondition(t *testing.T) {
	fakeNow := metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute

	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegion:              "region1",
						v1.LabelZoneFailureDomain:       "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
		},
		Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
	}

	nodeController, _ := newNodeLifecycleControllerFromClient(
		fakeNodeHandler,
		evictionTimeout,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
		true)
	nodeController.now = func() metav1.Time { return fakeNow }
	nodeController.recorder = testutil.NewFakeRecorder()
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)

	networkUnavailableTaint := &v1.Taint{
		Key:    v1.TaintNodeNetworkUnavailable,
		Effect: v1.TaintEffectNoSchedule,
	}
	notReadyTaint := &v1.Taint{
		Key:    v1.TaintNodeNotReady,
		Effect: v1.TaintEffectNoSchedule,
	}
	unreachableTaint := &v1.Taint{
		Key:    v1.TaintNodeUnreachable,
		Effect: v1.TaintEffectNoSchedule,
	}

	tests := []struct {
		Name           string
		Node           *v1.Node
		ExpectedTaints []*v1.Taint
	}{
		{
			Name: "NetworkUnavailable is true",
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegion:              "region1",
						v1.LabelZoneFailureDomain:       "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
						{
							Type:               v1.NodeNetworkUnavailable,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			ExpectedTaints: []*v1.Taint{networkUnavailableTaint},
		},
		{
			Name: "NetworkUnavailable is true",
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegion:              "region1",
						v1.LabelZoneFailureDomain:       "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
						{
							Type:               v1.NodeNetworkUnavailable,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			ExpectedTaints: []*v1.Taint{networkUnavailableTaint},
		},
		{
			Name: "Ready is false",
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegion:              "region1",
						v1.LabelZoneFailureDomain:       "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			ExpectedTaints: []*v1.Taint{notReadyTaint},
		},
		{
			Name: "Ready is unknown",
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegion:              "region1",
						v1.LabelZoneFailureDomain:       "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			ExpectedTaints: []*v1.Taint{unreachableTaint},
		},
	}

	for _, test := range tests {
		fakeNodeHandler.Update(context.TODO(), test.Node, metav1.UpdateOptions{})
		if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		nodeController.doNoScheduleTaintingPass(test.Node.Name)
		if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		node0, err := nodeController.nodeLister.Get("node0")
		if err != nil {
			t.Errorf("Can't get current node0...")
			return
		}
		if len(node0.Spec.Taints) != len(test.ExpectedTaints) {
			t.Errorf("%s: Unexpected number of taints: expected %d, got %d",
				test.Name, len(test.ExpectedTaints), len(node0.Spec.Taints))
		}
		for _, taint := range test.ExpectedTaints {
			if !taintutils.TaintExists(node0.Spec.Taints, taint) {
				t.Errorf("%s: Can't find taint %v in %v", test.Name, taint, node0.Spec.Taints)
			}
		}
	}
}

func TestNodeEventGeneration(t *testing.T) {
	fakeNow := metav1.Date(2016, 9, 10, 12, 0, 0, 0, time.UTC)
	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					UID:               "1234567890",
					CreationTimestamp: metav1.Date(2015, 8, 10, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 8, 10, 0, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 8, 10, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
		},
		Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
	}

	nodeController, _ := newNodeLifecycleControllerFromClient(
		fakeNodeHandler,
		5*time.Minute,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
		false)
	nodeController.now = func() metav1.Time { return fakeNow }
	fakeRecorder := testutil.NewFakeRecorder()
	nodeController.recorder = fakeRecorder
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)

	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(fakeRecorder.Events) != 1 {
		t.Fatalf("unexpected events, got %v, expected %v: %+v", len(fakeRecorder.Events), 1, fakeRecorder.Events)
	}
	if fakeRecorder.Events[0].Reason != "RegisteredNode" {
		var reasons []string
		for _, event := range fakeRecorder.Events {
			reasons = append(reasons, event.Reason)
		}
		t.Fatalf("unexpected events generation: %v", strings.Join(reasons, ","))
	}
	for _, event := range fakeRecorder.Events {
		involvedObject := event.InvolvedObject
		actualUID := string(involvedObject.UID)
		if actualUID != "1234567890" {
			t.Fatalf("unexpected event uid: %v", actualUID)
		}
	}
}

func TestReconcileNodeLabels(t *testing.T) {
	fakeNow := metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute

	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable:        "region1",
						v1.LabelZoneFailureDomainStable: "zone1",
						v1.LabelZoneRegion:              "region1",
						v1.LabelZoneFailureDomain:       "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
		},
		Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
	}

	nodeController, _ := newNodeLifecycleControllerFromClient(
		fakeNodeHandler,
		evictionTimeout,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
		true)
	nodeController.now = func() metav1.Time { return fakeNow }
	nodeController.recorder = testutil.NewFakeRecorder()
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)

	tests := []struct {
		Name           string
		Node           *v1.Node
		ExpectedLabels map[string]string
	}{
		{
			Name: "No-op if node has no labels",
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			ExpectedLabels: nil,
		},
		{
			Name: "No-op if no target labels present",
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelZoneRegionStable: "region1",
					},
				},
			},
			ExpectedLabels: map[string]string{
				v1.LabelZoneRegionStable: "region1",
			},
		},
		{
			Name: "Create OS/arch beta labels when they don't exist",
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelOSStable:   "linux",
						v1.LabelArchStable: "amd64",
					},
				},
			},
			ExpectedLabels: map[string]string{
				kubeletapis.LabelOS:   "linux",
				kubeletapis.LabelArch: "amd64",
				v1.LabelOSStable:      "linux",
				v1.LabelArchStable:    "amd64",
			},
		},
		{
			Name: "Reconcile OS/arch beta labels to match stable labels",
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						kubeletapis.LabelOS:   "windows",
						kubeletapis.LabelArch: "arm",
						v1.LabelOSStable:      "linux",
						v1.LabelArchStable:    "amd64",
					},
				},
			},
			ExpectedLabels: map[string]string{
				kubeletapis.LabelOS:   "linux",
				kubeletapis.LabelArch: "amd64",
				v1.LabelOSStable:      "linux",
				v1.LabelArchStable:    "amd64",
			},
		},
	}

	for _, test := range tests {
		fakeNodeHandler.Update(context.TODO(), test.Node, metav1.UpdateOptions{})
		if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		nodeController.reconcileNodeLabels(test.Node.Name)
		if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		node0, err := nodeController.nodeLister.Get("node0")
		if err != nil {
			t.Fatalf("Can't get current node0...")
		}
		if len(node0.Labels) != len(test.ExpectedLabels) {
			t.Errorf("%s: Unexpected number of taints: expected %d, got %d",
				test.Name, len(test.ExpectedLabels), len(node0.Labels))
		}
		for key, expectedValue := range test.ExpectedLabels {
			actualValue, ok := node0.Labels[key]
			if !ok {
				t.Errorf("%s: Can't find label %v in %v", test.Name, key, node0.Labels)
			}
			if actualValue != expectedValue {
				t.Errorf("%s: label %q: expected value %q, got value %q", test.Name, key, expectedValue, actualValue)
			}
		}
	}
}

func TestTryUpdateNodeHealth(t *testing.T) {
	fakeNow := metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC)
	fakeOld := metav1.Date(2016, 1, 1, 12, 0, 0, 0, time.UTC)
	evictionTimeout := 10 * time.Minute

	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: fakeNow,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  fakeNow,
							LastTransitionTime: fakeNow,
						},
					},
				},
			},
		},
		Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
	}

	nodeController, _ := newNodeLifecycleControllerFromClient(
		fakeNodeHandler,
		evictionTimeout,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
		true)
	nodeController.now = func() metav1.Time { return fakeNow }
	nodeController.recorder = testutil.NewFakeRecorder()
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)

	getStatus := func(cond *v1.NodeCondition) *v1.ConditionStatus {
		if cond == nil {
			return nil
		}
		return &cond.Status
	}

	tests := []struct {
		name string
		node *v1.Node
	}{
		{
			name: "Status true",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: fakeNow,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  fakeNow,
							LastTransitionTime: fakeNow,
						},
					},
				},
			},
		},
		{
			name: "Status false",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: fakeNow,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  fakeNow,
							LastTransitionTime: fakeNow,
						},
					},
				},
			},
		},
		{
			name: "Status unknown",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: fakeNow,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  fakeNow,
							LastTransitionTime: fakeNow,
						},
					},
				},
			},
		},
		{
			name: "Status nil",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: fakeNow,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{},
				},
			},
		},
		{
			name: "Status true - after grace period",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: fakeOld,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  fakeOld,
							LastTransitionTime: fakeOld,
						},
					},
				},
			},
		},
		{
			name: "Status false - after grace period",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: fakeOld,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  fakeOld,
							LastTransitionTime: fakeOld,
						},
					},
				},
			},
		},
		{
			name: "Status unknown - after grace period",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: fakeOld,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  fakeOld,
							LastTransitionTime: fakeOld,
						},
					},
				},
			},
		},
		{
			name: "Status nil - after grace period",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: fakeOld,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeController.nodeHealthMap.set(test.node.Name, &nodeHealthData{
				status:                   &test.node.Status,
				probeTimestamp:           test.node.CreationTimestamp,
				readyTransitionTimestamp: test.node.CreationTimestamp,
			})
			_, _, currentReadyCondition, err := nodeController.tryUpdateNodeHealth(test.node)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			_, savedReadyCondition := nodeutil.GetNodeCondition(nodeController.nodeHealthMap.getDeepCopy(test.node.Name).status, v1.NodeReady)
			savedStatus := getStatus(savedReadyCondition)
			currentStatus := getStatus(currentReadyCondition)
			if !apiequality.Semantic.DeepEqual(currentStatus, savedStatus) {
				t.Errorf("expected %v, got %v", savedStatus, currentStatus)
			}
		})
	}
}

func Test_isNodeExcludedFromDisruptionChecks(t *testing.T) {
	validNodeStatus := v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: "Test"}}}
	tests := []struct {
		name string

		enableExclusion bool
		enableLegacy    bool
		input           *v1.Node
		want            bool
	}{
		{want: false, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{}}}},
		{want: false, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Name: "master-abc"}}},
		{want: false, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelNodeDisruptionExclusion: ""}}}},

		{want: false, enableExclusion: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Name: "master-abc"}}},
		{want: false, enableLegacy: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelNodeDisruptionExclusion: ""}}}},

		{want: true, enableLegacy: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Name: "master-abc"}}},
		{want: true, enableExclusion: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelNodeDisruptionExclusion: ""}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDisruptionExclusion, tt.enableExclusion)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LegacyNodeRoleBehavior, tt.enableLegacy)()

			if result := isNodeExcludedFromDisruptionChecks(tt.input); result != tt.want {
				t.Errorf("isNodeExcludedFromDisruptionChecks() = %v, want %v", result, tt.want)
			}
		})
	}
}
