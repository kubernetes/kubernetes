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
	goruntime "runtime"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	coordv1 "k8s.io/api/coordination/v1"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	appsinformers "k8s.io/client-go/informers/apps/v1"
	coordinformers "k8s.io/client-go/informers/coordination/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	testcore "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	metricstestutil "k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	kubeletapis "k8s.io/kubelet/pkg/apis"
	"k8s.io/kubernetes/pkg/controller"
	nodelifecycleconfig "k8s.io/kubernetes/pkg/controller/nodelifecycle/config"
	"k8s.io/kubernetes/pkg/controller/nodelifecycle/scheduler"
	"k8s.io/kubernetes/pkg/controller/testutil"
	controllerutil "k8s.io/kubernetes/pkg/controller/util/node"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/node"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
	"k8s.io/utils/ptr"
)

const (
	testNodeMonitorGracePeriod = 50 * time.Second
	testNodeStartupGracePeriod = 60 * time.Second
	testNodeMonitorPeriod      = 5 * time.Second
	testRateLimiterQPS         = float32(100000)
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
	podInformer       coreinformers.PodInformer
}

func createNodeLease(nodeName string, renewTime metav1.MicroTime) *coordv1.Lease {
	return &coordv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      nodeName,
			Namespace: v1.NamespaceNodeLease,
		},
		Spec: coordv1.LeaseSpec{
			HolderIdentity: ptr.To(nodeName),
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

func (nc *nodeLifecycleController) syncPodStore(pod *v1.Pod) error {
	if pod == nil {
		return nil
	}
	newElems := make([]interface{}, 0, 1)
	newElems = append(newElems, pod)
	return nc.podInformer.Informer().GetStore().Replace(newElems, "newRV")
}

func newNodeLifecycleControllerFromClient(
	ctx context.Context,
	kubeClient clientset.Interface,
	evictionLimiterQPS float32,
	secondaryEvictionLimiterQPS float32,
	largeClusterThreshold int32,
	unhealthyZoneThreshold float32,
	nodeMonitorGracePeriod time.Duration,
	nodeStartupGracePeriod time.Duration,
	nodeMonitorPeriod time.Duration,
) (*nodeLifecycleController, error) {

	factory := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())

	leaseInformer := factory.Coordination().V1().Leases()
	nodeInformer := factory.Core().V1().Nodes()
	daemonSetInformer := factory.Apps().V1().DaemonSets()
	podInformer := factory.Core().V1().Pods()

	nc, err := NewNodeLifecycleController(ctx, ControllerOptions{
		LeaseInformer:               leaseInformer,
		PodInformer:                 podInformer,
		NodeInformer:                nodeInformer,
		DaemonSetInformer:           daemonSetInformer,
		KubeClient:                  kubeClient,
		NodeMonitorPeriod:           nodeMonitorPeriod,
		NodeStartupGracePeriod:      nodeStartupGracePeriod,
		NodeMonitorGracePeriod:      nodeMonitorGracePeriod,
		EvictionLimiterQPS:          evictionLimiterQPS,
		SecondaryEvictionLimiterQPS: secondaryEvictionLimiterQPS,
		LargeClusterThreshold:       largeClusterThreshold,
		UnhealthyZoneThreshold:      unhealthyZoneThreshold,
	})
	if err != nil {
		return nil, err
	}

	nc.leaseInformerSynced = alwaysReady
	nc.podInformerSynced = alwaysReady
	nc.nodeInformerSynced = alwaysReady
	nc.daemonSetInformerSynced = alwaysReady

	return &nodeLifecycleController{nc, leaseInformer, nodeInformer, daemonSetInformer, podInformer}, nil
}

func TestMonitorNodeHealth(t *testing.T) {
	fakeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	timeToPass := 60 * time.Minute
	healthyNodeNewStatus := v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:               v1.NodeReady,
				Status:             v1.ConditionTrue,
				LastHeartbeatTime:  metav1.NewTime(fakeNow.Add(timeToPass)),
				LastTransitionTime: fakeNow,
			},
		},
	}
	unhealthyNodeNewStatus := v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:   v1.NodeReady,
				Status: v1.ConditionUnknown,
				// Node status was updated by nodecontroller timeToPass ago
				LastHeartbeatTime:  fakeNow,
				LastTransitionTime: fakeNow,
			},
		},
	}

	tests := map[string]struct {
		nodeList                []*v1.Node
		updatedNodeStatuses     []v1.NodeStatus
		expectedInitialStates   map[string]ZoneState
		expectedFollowingStates map[string]ZoneState
	}{
		"No Disruption: Node created recently without failure domain labels (happens only at cluster startup)": {
			nodeList: []*v1.Node{
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
			updatedNodeStatuses: []v1.NodeStatus{
				healthyNodeNewStatus,
			},
			expectedInitialStates: map[string]ZoneState{
				"": stateNormal,
			},
			expectedFollowingStates: map[string]ZoneState{
				"": stateNormal,
			},
		},
		"No Disruption: Initially both zones down, one comes back": {
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone2",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone2",
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
		},
		"Partial Disruption: Nodes created recently without status conditions (happens only at cluster startup)": {
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node1",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node2",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node3",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
						},
					},
				},
			},
			updatedNodeStatuses: []v1.NodeStatus{
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
				healthyNodeNewStatus,
			},
			expectedInitialStates: map[string]ZoneState{
				// we've not received any status for the nodes yet
				// so the controller assumes the zones is fully disrupted
				testutil.CreateZoneID("region1", "zone1"): stateFullDisruption,
			},
			expectedFollowingStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): statePartialDisruption,
			},
		},
		"Partial Disruption: one Node failed leading to the number of healthy Nodes to exceed the configured threshold": {
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
						Name:              "node3",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
			updatedNodeStatuses: []v1.NodeStatus{
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
				healthyNodeNewStatus,
				healthyNodeNewStatus,
			},
			expectedInitialStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateNormal,
			},
			expectedFollowingStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): statePartialDisruption,
			},
		},
		"Full Disruption: the zone has less than 2 Nodes down, the last healthy Node has failed": {
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
			updatedNodeStatuses: []v1.NodeStatus{
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
				unhealthyNodeNewStatus,
			},
			expectedInitialStates: map[string]ZoneState{
				// if a zone has a number of unhealthy nodes less or equal to 2
				// the controller will consider it normal regardless on
				// the ration of healthy vs unhealthy nodes
				testutil.CreateZoneID("region1", "zone1"): stateNormal,
			},
			expectedFollowingStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateFullDisruption,
			},
		},
		"Full Disruption: all the Nodes in one zone are down": {
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone2",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone2",
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
		},
		"Full Disruption: all the Nodes in both the zones are down": {
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region2",
							v1.LabelTopologyZone:            "zone2",
							v1.LabelFailureDomainBetaRegion: "region2",
							v1.LabelFailureDomainBetaZone:   "zone2",
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
		},
		"Full Disruption: Ready condition removed from the Node": {
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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

			updatedNodeStatuses: []v1.NodeStatus{
				{
					Conditions: []v1.NodeCondition{},
				},
			},
			expectedInitialStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateNormal,
			},
			expectedFollowingStates: map[string]ZoneState{
				testutil.CreateZoneID("region1", "zone1"): stateFullDisruption,
			},
		},
		"Full Disruption: the only available Node has the node.kubernetes.io/exclude-disruption label": {
			nodeList: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "node0",
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
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
						CreationTimestamp: fakeNow,
						Labels: map[string]string{
							v1.LabelTopologyRegion:          "region1",
							v1.LabelTopologyZone:            "zone1",
							v1.LabelFailureDomainBetaRegion: "region1",
							v1.LabelFailureDomainBetaZone:   "zone1",
							labelNodeDisruptionExclusion:    "",
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
		},
	}

	for testName, tt := range tests {
		t.Run(testName, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			fakeNodeHandler := &testutil.FakeNodeHandler{
				Existing:  tt.nodeList,
				Clientset: fake.NewSimpleClientset(),
			}
			nodeController, _ := newNodeLifecycleControllerFromClient(
				tCtx,
				fakeNodeHandler,
				testRateLimiterQPS,
				testRateLimiterQPS,
				testLargeClusterThreshold,
				testUnhealthyThreshold,
				testNodeMonitorGracePeriod,
				testNodeStartupGracePeriod,
				testNodeMonitorPeriod)
			nodeController.recorder = testutil.NewFakeRecorder()
			nodeController.enterPartialDisruptionFunc = func(nodeNum int) float32 {
				return testRateLimiterQPS
			}
			nodeController.enterFullDisruptionFunc = func(nodeNum int) float32 {
				return testRateLimiterQPS
			}

			syncAndDiffZoneState := func(wanted map[string]ZoneState) {
				if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if err := nodeController.monitorNodeHealth(tCtx); err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if diff := cmp.Diff(wanted, nodeController.zoneStates); diff != "" {
					t.Errorf("unexpected zone state (-want +got):\n%s", diff)
				}
			}

			// initial zone state
			nodeController.now = func() metav1.Time { return fakeNow }
			syncAndDiffZoneState(tt.expectedInitialStates)

			// following zone state
			nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Add(timeToPass)} }
			for i := range tt.updatedNodeStatuses {
				fakeNodeHandler.Existing[i].Status = tt.updatedNodeStatuses[i]
			}
			syncAndDiffZoneState(tt.expectedFollowingStates)
		})
	}
}

func TestPodStatusChange(t *testing.T) {
	fakeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)

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
								v1.LabelTopologyRegion:          "region1",
								v1.LabelTopologyZone:            "zone1",
								v1.LabelFailureDomainBetaRegion: "region1",
								v1.LabelFailureDomainBetaZone:   "zone1",
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
								v1.LabelFailureDomainBetaRegion: "region1",
								v1.LabelFailureDomainBetaZone:   "zone1",
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

	tCtx := ktesting.Init(t)
	for _, item := range table {
		nodeController, _ := newNodeLifecycleControllerFromClient(
			tCtx,
			item.fakeNodeHandler,
			testRateLimiterQPS,
			testRateLimiterQPS,
			testLargeClusterThreshold,
			testUnhealthyThreshold,
			testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod,
			testNodeMonitorPeriod,
		)
		nodeController.now = func() metav1.Time { return fakeNow }
		nodeController.recorder = testutil.NewFakeRecorder()
		nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(item.fakeNodeHandler.Clientset)
		if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(tCtx); err != nil {
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
		if err := nodeController.monitorNodeHealth(tCtx); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		zones := testutil.GetZones(item.fakeNodeHandler)
		for _, zone := range zones {
			nodeController.zoneNoExecuteTainter[zone].Try(tCtx.Logger(), func(value scheduler.TimedValue) (bool, time.Duration) {
				nodeUID, _ := value.UID.(string)
				pods, err := nodeController.getPodsAssignedToNode(value.Value)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				_, err = controllerutil.DeletePods(tCtx, item.fakeNodeHandler, pods, nodeController.recorder, value.Value, nodeUID, nodeController.daemonSetStore)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
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
	tCtx := ktesting.Init(t)
	for i, item := range table {
		nodeController, _ := newNodeLifecycleControllerFromClient(
			tCtx,
			item.fakeNodeHandler,
			testRateLimiterQPS,
			testRateLimiterQPS,
			testLargeClusterThreshold,
			testUnhealthyThreshold,
			testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod,
			testNodeMonitorPeriod,
		)
		nodeController.now = func() metav1.Time { return fakeNow }
		nodeController.recorder = testutil.NewFakeRecorder()
		nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(item.fakeNodeHandler.Clientset)
		if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(tCtx); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
			if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if err := nodeController.monitorNodeHealth(tCtx); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		}
		if item.expectedRequestCount != item.fakeNodeHandler.RequestCount {
			t.Errorf("expected %v call, but got %v.", item.expectedRequestCount, item.fakeNodeHandler.RequestCount)
		}
		if len(item.fakeNodeHandler.UpdatedNodes) > 0 && !apiequality.Semantic.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodes) {
			t.Errorf("Case[%d] unexpected nodes: %s", i, cmp.Diff(item.expectedNodes[0], item.fakeNodeHandler.UpdatedNodes[0]))
		}
		if len(item.fakeNodeHandler.UpdatedNodeStatuses) > 0 && !apiequality.Semantic.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodeStatuses) {
			t.Errorf("Case[%d] unexpected nodes: %s", i, cmp.Diff(item.expectedNodes[0], item.fakeNodeHandler.UpdatedNodeStatuses[0]))
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

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeControllerLeaseCircuitBreaker, true)

	testcases := []struct {
		description             string
		fakeNodeHandler         *testutil.FakeNodeHandler
		lease                   *coordv1.Lease
		timeToPass              time.Duration
		newNodeStatus           v1.NodeStatus
		newLease                *coordv1.Lease
		apiServerLease          *coordv1.Lease
		newAPIServerLease       *coordv1.Lease
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
		// Node created long time ago, with status updated by kubelet exceeds grace period.
		// Node lease is expired in cache, but fresh in API server.
		// Expect no action from node controller (within monitor grace period) because of double check.
		{
			description: "Node created long time ago, with status updated by kubelet exceeds grace period. Node lease is expired in cache, but fresh in API server.",
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
			lease:      createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time)),
			timeToPass: time.Hour,
			newNodeStatus: v1.NodeStatus{
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
			newLease: func() *coordv1.Lease {
				l := createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time))
				l.ResourceVersion = "50"
				return l
			}(),
			apiServerLease: func() *coordv1.Lease {
				l := createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time))
				l.ResourceVersion = "50"
				return l
			}(),
			newAPIServerLease: func() *coordv1.Lease {
				l := createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time.Add(time.Hour)))
				l.ResourceVersion = "100"
				return l
			}(),
			expectedRequestCount: 2, // List nodes + List nodes + GET lease (GET not counted)
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
		// Node lease expired in cache but fresh in API server. EnsureReady should skip on second run.
		{
			description: "Node lease expired in cache but fresh in API server. EnsureReady should skip on second run.",
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
			lease: func() *coordv1.Lease {
				l := createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time.Add(-60*time.Second)))
				l.ResourceVersion = "50"
				return l
			}(),
			timeToPass: 1 * time.Second,
			newNodeStatus: v1.NodeStatus{
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
			newLease: func() *coordv1.Lease {
				l := createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time.Add(-60*time.Second)))
				l.ResourceVersion = "50"
				return l
			}(),
			apiServerLease: func() *coordv1.Lease {
				l := createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time))
				l.ResourceVersion = "100"
				return l
			}(),
			newAPIServerLease: func() *coordv1.Lease {
				l := createNodeLease("node0", metav1.NewMicroTime(fakeNow.Time))
				l.ResourceVersion = "100"
				return l
			}(),
			expectedRequestCount: 2, // List nodes + List nodes (GET lease not counted)
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
	}

	for _, item := range testcases {
		t.Run(item.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			nodeController, _ := newNodeLifecycleControllerFromClient(
				tCtx,
				item.fakeNodeHandler,
				testRateLimiterQPS,
				testRateLimiterQPS,
				testLargeClusterThreshold,
				testUnhealthyThreshold,
				testNodeMonitorGracePeriod,
				testNodeStartupGracePeriod,
				testNodeMonitorPeriod,
			)
			nodeController.now = func() metav1.Time { return fakeNow }
			nodeController.recorder = testutil.NewFakeRecorder()
			nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(item.fakeNodeHandler.Clientset)
			if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if err := nodeController.syncLeaseStore(item.lease); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if item.apiServerLease != nil {
				_, err := item.fakeNodeHandler.CoordinationV1().Leases(v1.NamespaceNodeLease).Create(tCtx, item.apiServerLease, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unexpected error creating lease in fake client: %v", err)
				}
			}
			if err := nodeController.monitorNodeHealth(tCtx); err != nil {
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
				if item.newAPIServerLease != nil {
					_, err := item.fakeNodeHandler.CoordinationV1().Leases(v1.NamespaceNodeLease).Update(tCtx, item.newAPIServerLease, metav1.UpdateOptions{})
					if err != nil {
						t.Fatalf("unexpected error updating lease in fake client: %v", err)
					}
				}
				if err := nodeController.monitorNodeHealth(tCtx); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}
			if item.expectedRequestCount != item.fakeNodeHandler.RequestCount {
				t.Errorf("expected %v call, but got %v.", item.expectedRequestCount, item.fakeNodeHandler.RequestCount)
			}
			if len(item.fakeNodeHandler.UpdatedNodes) > 0 && !apiequality.Semantic.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodes) {
				t.Errorf("unexpected nodes: %s", cmp.Diff(item.expectedNodes[0], item.fakeNodeHandler.UpdatedNodes[0]))
			}
			if len(item.fakeNodeHandler.UpdatedNodeStatuses) > 0 && !apiequality.Semantic.DeepEqual(item.expectedNodes, item.fakeNodeHandler.UpdatedNodeStatuses) {
				t.Errorf("unexpected nodes: %s", cmp.Diff(item.expectedNodes[0], item.fakeNodeHandler.UpdatedNodeStatuses[0]))
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

	tCtx := ktesting.Init(t)
	for i, item := range table {
		nodeController, _ := newNodeLifecycleControllerFromClient(
			tCtx,
			item.fakeNodeHandler,
			testRateLimiterQPS,
			testRateLimiterQPS,
			testLargeClusterThreshold,
			testUnhealthyThreshold,
			testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod,
			testNodeMonitorPeriod,
		)
		nodeController.now = func() metav1.Time { return fakeNow }
		nodeController.recorder = testutil.NewFakeRecorder()
		nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(item.fakeNodeHandler.Clientset)
		if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(tCtx); err != nil {
			t.Errorf("Case[%d] unexpected error: %v", i, err)
		}
		if item.timeToPass > 0 {
			nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Add(item.timeToPass)} }
			item.fakeNodeHandler.Existing[0].Status = item.newNodeStatus
			if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if err := nodeController.monitorNodeHealth(tCtx); err != nil {
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

// TestMonitorNodeHealthMarkPodsNotReadyWithWorkerSize tests the happy path of
// TestMonitorNodeHealthMarkPodsNotReady with a large number of nodes/pods and
// varying numbers of workers.
func TestMonitorNodeHealthMarkPodsNotReadyWithWorkerSize(t *testing.T) {
	const numNodes = 50
	const podsPerNode = 100
	makeNodes := func() []*v1.Node {
		nodes := make([]*v1.Node, numNodes)
		// Node created long time ago, with status updated by kubelet exceeds grace period.
		// Expect pods status updated and Unknown node status posted from node controller
		for i := 0; i < numNodes; i++ {
			nodes[i] = &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              fmt.Sprintf("node%d", i),
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
			}
		}
		return nodes
	}
	makePods := func() []v1.Pod {
		pods := make([]v1.Pod, numNodes*podsPerNode)
		for i := 0; i < numNodes*podsPerNode; i++ {
			pods[i] = *testutil.NewPod(fmt.Sprintf("pod%d", i), fmt.Sprintf("node%d", i%numNodes))
		}
		return pods
	}

	table := []struct {
		workers int
	}{
		{workers: 0}, // will default to scheduler.UpdateWorkerSize
		{workers: 1},
	}

	tCtx := ktesting.Init(t)
	for i, item := range table {
		fakeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)

		fakeNodeHandler := &testutil.FakeNodeHandler{
			Existing:  makeNodes(),
			Clientset: fake.NewSimpleClientset(&v1.PodList{Items: makePods()}),
		}

		nodeController, _ := newNodeLifecycleControllerFromClient(
			tCtx,
			fakeNodeHandler,
			testRateLimiterQPS,
			testRateLimiterQPS,
			testLargeClusterThreshold,
			testUnhealthyThreshold,
			testNodeMonitorGracePeriod,
			testNodeStartupGracePeriod,
			testNodeMonitorPeriod)
		nodeController.now = func() metav1.Time { return fakeNow }
		nodeController.recorder = testutil.NewFakeRecorder()
		nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)
		if item.workers != 0 {
			nodeController.nodeUpdateWorkerSize = item.workers
		}
		if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(tCtx); err != nil {
			t.Errorf("Case[%d] unexpected error: %v", i, err)
		}

		nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Add(1 * time.Minute)} }
		for i := range fakeNodeHandler.Existing {
			fakeNodeHandler.Existing[i].Status = v1.NodeStatus{
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
			}
		}

		if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.monitorNodeHealth(tCtx); err != nil {
			t.Errorf("Case[%d] unexpected error: %v", i, err)
		}

		podStatusUpdates := 0
		for _, action := range fakeNodeHandler.Actions() {
			if action.GetVerb() == "update" && action.GetResource().Resource == "pods" && action.GetSubresource() == "status" {
				podStatusUpdates++
			}
		}
		const expectedPodStatusUpdates = numNodes * podsPerNode
		if podStatusUpdates != expectedPodStatusUpdates {
			t.Errorf("Case[%d] expect pod status updated to be %v, but got %v", i, expectedPodStatusUpdates, podStatusUpdates)
		}
	}
}

func TestMonitorNodeHealthRemovesDeletedNodeHealth(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	fakeNodeHandler := &testutil.FakeNodeHandler{
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
					CreationTimestamp: fakeNow,
				},
			},
		},
		Clientset: fake.NewSimpleClientset(&v1.PodList{}),
	}
	nodeController, _ := newNodeLifecycleControllerFromClient(
		tCtx,
		fakeNodeHandler,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod)
	nodeController.now = func() metav1.Time { return fakeNow }
	nodeController.recorder = testutil.NewFakeRecorder()
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)

	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(tCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, name := range []string{"node0", "node1"} {
		if nodeController.nodeHealthMap.getDeepCopy(name) == nil {
			t.Fatalf("expected node health of %s to be recorded", name)
		}
	}

	fakeNodeHandler.Existing = fakeNodeHandler.Existing[:1]
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(tCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if nodeController.nodeHealthMap.getDeepCopy("node0") == nil {
		t.Errorf("expected node health of node0 to be kept")
	}
	if nodeController.nodeHealthMap.getDeepCopy("node1") != nil {
		t.Errorf("expected node health of deleted node1 to be removed")
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
			tCtx := ktesting.Init(t)
			nodeController, _ := newNodeLifecycleControllerFromClient(
				tCtx,
				item.fakeNodeHandler,
				testRateLimiterQPS,
				testRateLimiterQPS,
				testLargeClusterThreshold,
				testUnhealthyThreshold,
				testNodeMonitorGracePeriod,
				testNodeStartupGracePeriod,
				testNodeMonitorPeriod,
			)
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
				if err := nodeController.monitorNodeHealth(tCtx); err != nil {
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
	// TODO: Remove skip once https://github.com/kubernetes/kubernetes/pull/114607 merges.
	if goruntime.GOOS == "windows" {
		t.Skip("Skipping test on Windows.")
	}
	fakeNow := metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC)

	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			// Unreachable Taint with effect 'NoExecute' should be applied to this node.
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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
			// NotReady Taint with NoExecute effect should not be applied to a node if the NodeCondition Type NodeReady has been set to nil in the interval between the NodeController enqueuing the node and the taint manager picking up.
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node3",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  metav1.Date(2016, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2016, 1, 1, 12, 0, 0, 0, time.UTC),
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
	unhealthyNodeNewStatus := v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:               v1.NodeReady,
				Status:             v1.ConditionFalse,
				LastHeartbeatTime:  metav1.Date(2017, 1, 1, 12, 10, 0, 0, time.UTC),
				LastTransitionTime: metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	}
	overrideNodeNewStatusConditions := []v1.NodeCondition{
		{
			Type:               "MemoryPressure",
			Status:             v1.ConditionUnknown,
			LastHeartbeatTime:  metav1.Date(2017, 1, 1, 12, 10, 0, 0, time.UTC),
			LastTransitionTime: metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC),
		},
	}
	originalTaint := UnreachableTaintTemplate
	tCtx := ktesting.Init(t)
	nodeController, _ := newNodeLifecycleControllerFromClient(
		tCtx,
		fakeNodeHandler,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
	)
	nodeController.now = func() metav1.Time { return fakeNow }
	nodeController.recorder = testutil.NewFakeRecorder()
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(tCtx); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass(tCtx)
	node0, err := fakeNodeHandler.Get(tCtx, "node0", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node0...")
		return
	}
	if !taintutils.TaintExists(node0.Spec.Taints, UnreachableTaintTemplate) {
		t.Errorf("Can't find taint %v in %v", originalTaint, node0.Spec.Taints)
	}
	node2, err := fakeNodeHandler.Get(tCtx, "node2", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node2...")
		return
	}
	if !taintutils.TaintExists(node2.Spec.Taints, NotReadyTaintTemplate) {
		t.Errorf("Can't find taint %v in %v", NotReadyTaintTemplate, node2.Spec.Taints)
	}

	// Make node3 healthy again.
	node2.Status = healthyNodeNewStatus
	_, err = fakeNodeHandler.UpdateStatus(tCtx, node2, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err.Error())
		return
	}
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(tCtx); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass(tCtx)

	node2, err = fakeNodeHandler.Get(tCtx, "node2", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node2...")
		return
	}
	// We should not see any taint on the node(especially the Not-Ready taint with NoExecute effect).
	if taintutils.TaintExists(node2.Spec.Taints, NotReadyTaintTemplate) || len(node2.Spec.Taints) > 0 {
		t.Errorf("Found taint %v in %v, which should not be present", NotReadyTaintTemplate, node2.Spec.Taints)
	}

	node3, err := fakeNodeHandler.Get(tCtx, "node3", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node3...")
		return
	}
	node3.Status = unhealthyNodeNewStatus
	_, err = fakeNodeHandler.UpdateStatus(tCtx, node3, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err.Error())
		return
	}
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(tCtx); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// Before taint manager work, the status has been replaced(maybe merge-patch replace).
	node3.Status.Conditions = overrideNodeNewStatusConditions
	_, err = fakeNodeHandler.UpdateStatus(tCtx, node3, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err.Error())
		return
	}
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass(tCtx)
	node3, err = fakeNodeHandler.Get(tCtx, "node3", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node3...")
		return
	}
	// We should not see any taint on the node(especially the Not-Ready taint with NoExecute effect).
	if taintutils.TaintExists(node3.Spec.Taints, NotReadyTaintTemplate) || len(node3.Spec.Taints) > 0 {
		t.Errorf("Found taint %v in %v, which should not be present", NotReadyTaintTemplate, node3.Spec.Taints)
	}
}

// TestApplyNoExecuteTaintsToUnreachableNode ensures a NoExecute taint is applied to node that hasn't posted the
// node status for a period greater than nodeMonitorGracePeriod.
func TestApplyNoExecuteTaintsToUnreachableNode(t *testing.T) {
	// TODO: Remove skip once https://github.com/kubernetes/kubernetes/pull/114607 merges.
	if goruntime.GOOS == "windows" {
		t.Skip("Skipping test on Windows.")
	}
	time1 := metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC)
	// time2 is set to NodeMonitorGracePeriod plus 1 second after time1.
	time2 := metav1.Time{Time: time1.Add(testNodeMonitorGracePeriod + time.Second)}

	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			// Unreachable Taint with effect 'NoExecute' should be applied to this node after 2015-01-01 12:00:40.
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  time1,
							LastTransitionTime: time1,
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
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  time1,
							LastTransitionTime: time1,
						},
					},
				},
			},
		},
		Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
	}
	_, ctx := ktesting.NewTestContext(t)
	nodeController, _ := newNodeLifecycleControllerFromClient(
		ctx,
		fakeNodeHandler,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
	)
	// No taint should be added when the NodeReady conditions were not set longer ago than NodeMonitorGracePeriod.
	nodeController.now = func() metav1.Time { return time1 }
	nodeController.recorder = testutil.NewFakeRecorder()
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(ctx); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass(ctx)
	node0, err := fakeNodeHandler.Get(ctx, "node0", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node0...")
		return
	}
	if len(node0.Spec.Taints) > 0 {
		t.Errorf("Got unexpected taints %v", node0.Spec.Taints)
	}
	node1, err := fakeNodeHandler.Get(ctx, "node1", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node1...")
		return
	}
	if len(node1.Spec.Taints) > 0 {
		t.Errorf("Got unexpected taints %v", node1.Spec.Taints)
	}

	// Advance the clock to time2 to expire the previous NodeReady condition.
	nodeController.now = func() metav1.Time { return time2 }
	// Only node1 sends a new heartbeat at time2.
	node1.Status = v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:               v1.NodeReady,
				Status:             v1.ConditionTrue,
				LastHeartbeatTime:  time2,
				LastTransitionTime: time1,
			},
		},
	}
	_, err = fakeNodeHandler.UpdateStatus(ctx, node1, metav1.UpdateOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	// Unreachable taint should be added to node0 and not node1 because the former's NodeReadyCondition was set longer ago than NodeMonitorGracePeriod.
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(ctx); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass(ctx)

	node0, err = fakeNodeHandler.Get(ctx, "node0", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node0...")
		return
	}
	if !taintutils.TaintExists(node0.Spec.Taints, UnreachableTaintTemplate) {
		t.Errorf("Can't find taint %v in %v", UnreachableTaintTemplate, node0.Spec.Taints)
	}
	node1, err = fakeNodeHandler.Get(ctx, "node1", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node1...")
		return
	}
	if len(node1.Spec.Taints) > 0 {
		t.Errorf("Got unexpected taints %v", node1.Spec.Taints)
	}

	// Unreachable taint should be deleted from node0 after it sends a new heartbeat.
	node0.Status = v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:               v1.NodeReady,
				Status:             v1.ConditionTrue,
				LastHeartbeatTime:  time2,
				LastTransitionTime: time2,
			},
		},
	}
	_, err = fakeNodeHandler.UpdateStatus(ctx, node0, metav1.UpdateOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		return
	}
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(ctx); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass(ctx)

	node0, err = fakeNodeHandler.Get(ctx, "node0", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node0...")
		return
	}
	// We should not see any taint on the node.
	if len(node0.Spec.Taints) > 0 {
		t.Errorf("Got unexpected taints %v", node0.Spec.Taints)
	}
}

// TestApplyNoExecuteTaintsToNodesEnqueueTwice ensures we taint every node with NoExecute even if enqueued twice
func TestApplyNoExecuteTaintsToNodesEnqueueTwice(t *testing.T) {
	// TODO: Remove skip once https://github.com/kubernetes/kubernetes/pull/114607 merges.
	if goruntime.GOOS == "windows" {
		t.Skip("Skipping test on Windows.")
	}
	fakeNow := metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC)

	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			// Unreachable Taint with effect 'NoExecute' should be applied to this node.
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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
	tCtx := ktesting.Init(t)
	nodeController, _ := newNodeLifecycleControllerFromClient(
		tCtx,
		fakeNodeHandler,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
	)
	nodeController.now = func() metav1.Time { return fakeNow }
	nodeController.recorder = testutil.NewFakeRecorder()
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// 1. monitor node health twice, add untainted node once
	if err := nodeController.monitorNodeHealth(tCtx); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(tCtx); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// 2. mark node0 healthy
	node0, err := fakeNodeHandler.Get(tCtx, "node0", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node0...")
		return
	}
	node0.Status = healthyNodeNewStatus
	_, err = fakeNodeHandler.UpdateStatus(tCtx, node0, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err.Error())
		return
	}

	// add other notReady nodes
	fakeNodeHandler.Existing = append(fakeNodeHandler.Existing, []*v1.Node{
		// Unreachable Taint with effect 'NoExecute' should be applied to this node.
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:              "node3",
				CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				Labels: map[string]string{
					v1.LabelTopologyRegion:          "region1",
					v1.LabelTopologyZone:            "zone1",
					v1.LabelFailureDomainBetaRegion: "region1",
					v1.LabelFailureDomainBetaZone:   "zone1",
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
				Name:              "node4",
				CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				Labels: map[string]string{
					v1.LabelTopologyRegion:          "region1",
					v1.LabelTopologyZone:            "zone1",
					v1.LabelFailureDomainBetaRegion: "region1",
					v1.LabelFailureDomainBetaZone:   "zone1",
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
				Name:              "node5",
				CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				Labels: map[string]string{
					v1.LabelTopologyRegion:          "region1",
					v1.LabelTopologyZone:            "zone1",
					v1.LabelFailureDomainBetaRegion: "region1",
					v1.LabelFailureDomainBetaZone:   "zone1",
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
	}...)
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// 3. start monitor node health again, add untainted node twice, construct UniqueQueue with duplicated node cache
	if err := nodeController.monitorNodeHealth(tCtx); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// 4. do NoExecute taint pass
	// when processing with node0, condition.Status is NodeReady, and return true with default case
	// then remove the set value and queue value both, the taint job never stuck
	nodeController.doNoExecuteTaintingPass(tCtx)

	// 5. get node3 and node5, see if it has ready got NoExecute taint
	node3, err := fakeNodeHandler.Get(tCtx, "node3", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node3...")
		return
	}
	if !taintutils.TaintExists(node3.Spec.Taints, UnreachableTaintTemplate) || len(node3.Spec.Taints) == 0 {
		t.Errorf("Not found taint %v in %v, which should be present in %s", UnreachableTaintTemplate, node3.Spec.Taints, node3.Name)
	}
	node5, err := fakeNodeHandler.Get(tCtx, "node5", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node5...")
		return
	}
	if !taintutils.TaintExists(node5.Spec.Taints, NotReadyTaintTemplate) || len(node5.Spec.Taints) == 0 {
		t.Errorf("Not found taint %v in %v, which should be present in %s", NotReadyTaintTemplate, node5.Spec.Taints, node5.Name)
	}
}

func TestSwapUnreachableNotReadyTaints(t *testing.T) {
	fakeNow := metav1.Date(2017, 1, 1, 12, 0, 0, 0, time.UTC)

	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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

	tCtx := ktesting.Init(t)
	nodeController, _ := newNodeLifecycleControllerFromClient(
		tCtx,
		fakeNodeHandler,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
	)
	nodeController.now = func() metav1.Time { return fakeNow }
	nodeController.recorder = testutil.NewFakeRecorder()
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)
	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(tCtx); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass(tCtx)

	node0, err := fakeNodeHandler.Get(tCtx, "node0", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node0...")
		return
	}
	node1, err := fakeNodeHandler.Get(tCtx, "node1", metav1.GetOptions{})
	if err != nil {
		t.Errorf("Can't get current node1...")
		return
	}

	if originalTaint != nil && !taintutils.TaintExists(node0.Spec.Taints, originalTaint) {
		t.Errorf("Can't find taint %v in %v", originalTaint, node0.Spec.Taints)
	}

	nodeController.now = func() metav1.Time { return metav1.Time{Time: fakeNow.Time} }

	node0.Status = newNodeStatus
	node1.Status = healthyNodeNewStatus
	_, err = fakeNodeHandler.UpdateStatus(tCtx, node0, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err.Error())
		return
	}
	_, err = fakeNodeHandler.UpdateStatus(tCtx, node1, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err.Error())
		return
	}

	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(tCtx); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	nodeController.doNoExecuteTaintingPass(tCtx)

	node0, err = fakeNodeHandler.Get(tCtx, "node0", metav1.GetOptions{})
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

	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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

	tCtx := ktesting.Init(t)
	nodeController, _ := newNodeLifecycleControllerFromClient(
		tCtx,
		fakeNodeHandler,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
	)
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
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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
		if _, err := fakeNodeHandler.Update(tCtx, test.Node, metav1.UpdateOptions{}); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if err := nodeController.doNoScheduleTaintingPass(tCtx, test.Node.Name); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
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

	tCtx := ktesting.Init(t)
	nodeController, _ := newNodeLifecycleControllerFromClient(
		tCtx,
		fakeNodeHandler,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
	)
	nodeController.now = func() metav1.Time { return fakeNow }
	fakeRecorder := testutil.NewFakeRecorder()
	nodeController.recorder = fakeRecorder
	nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)

	if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := nodeController.monitorNodeHealth(tCtx); err != nil {
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

	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						v1.LabelTopologyRegion:          "region1",
						v1.LabelTopologyZone:            "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
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

	tCtx := ktesting.Init(t)
	nodeController, _ := newNodeLifecycleControllerFromClient(
		tCtx,
		fakeNodeHandler,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
	)
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
						v1.LabelTopologyRegion: "region1",
					},
				},
			},
			ExpectedLabels: map[string]string{
				v1.LabelTopologyRegion: "region1",
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
		if _, err := fakeNodeHandler.Update(tCtx, test.Node, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if err := nodeController.syncNodeStore(fakeNodeHandler); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if err := nodeController.reconcileNodeLabels(tCtx, test.Node.Name); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
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

	tCtx := ktesting.Init(t)
	nodeController, _ := newNodeLifecycleControllerFromClient(
		tCtx,
		fakeNodeHandler,
		testRateLimiterQPS,
		testRateLimiterQPS,
		testLargeClusterThreshold,
		testUnhealthyThreshold,
		testNodeMonitorGracePeriod,
		testNodeStartupGracePeriod,
		testNodeMonitorPeriod,
	)
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
			_, _, currentReadyCondition, err := nodeController.tryUpdateNodeHealth(tCtx, test.node)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			_, savedReadyCondition := controllerutil.GetNodeCondition(nodeController.nodeHealthMap.getDeepCopy(test.node.Name).status, v1.NodeReady)
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

		input *v1.Node
		want  bool
	}{
		{want: false, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{}}}},
		{want: false, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Name: "master-abc"}}},
		{want: true, input: &v1.Node{Status: validNodeStatus, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{labelNodeDisruptionExclusion: ""}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if result := isNodeExcludedFromDisruptionChecks(tt.input); result != tt.want {
				t.Errorf("isNodeExcludedFromDisruptionChecks() = %v, want %v", result, tt.want)
			}
		})
	}
}

func TestProcessPodMarkPodNotReady(t *testing.T) {
	fakeNow := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)

	table := []struct {
		desc                    string
		fakeNodeHandler         *testutil.FakeNodeHandler
		pod                     *v1.Pod
		expectedPodStatusUpdate bool
		monitorNodeHealth       bool
	}{
		{
			desc: "Do not mark pod as NotReady when the scheduled node's healthy is not gathered yet",
			fakeNodeHandler: &testutil.FakeNodeHandler{
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
									Status:             v1.ConditionFalse,
									LastHeartbeatTime:  fakeNow,
									LastTransitionTime: fakeNow,
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			pod:                     testutil.NewPod("pod0", "node0"),
			monitorNodeHealth:       false,
			expectedPodStatusUpdate: false,
		},
		{
			desc: "Do not mark pod as NotReady when the scheduled node is ready",
			fakeNodeHandler: &testutil.FakeNodeHandler{
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
			},
			pod:                     testutil.NewPod("pod0", "node0"),
			monitorNodeHealth:       true,
			expectedPodStatusUpdate: false,
		},
		{
			desc: "Pod marked as NotReady when the scheduled node is not ready",
			fakeNodeHandler: &testutil.FakeNodeHandler{
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
									Status:             v1.ConditionFalse,
									LastHeartbeatTime:  fakeNow,
									LastTransitionTime: fakeNow,
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			pod:                     testutil.NewPod("pod0", "node0"),
			monitorNodeHealth:       true,
			expectedPodStatusUpdate: true,
		},
	}

	for _, item := range table {
		t.Run(item.desc, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			nodeController, _ := newNodeLifecycleControllerFromClient(
				tCtx,
				item.fakeNodeHandler,
				testRateLimiterQPS,
				testRateLimiterQPS,
				testLargeClusterThreshold,
				testUnhealthyThreshold,
				testNodeMonitorGracePeriod,
				testNodeStartupGracePeriod,
				testNodeMonitorPeriod,
			)
			nodeController.now = func() metav1.Time { return fakeNow }
			nodeController.recorder = testutil.NewFakeRecorder()
			nodeController.getPodsAssignedToNode = fakeGetPodsAssignedToNode(item.fakeNodeHandler.Clientset)
			if err := nodeController.syncNodeStore(item.fakeNodeHandler); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if item.monitorNodeHealth {
				if err := nodeController.monitorNodeHealth(tCtx); err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}

			if err := nodeController.syncPodStore(item.pod); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			nodeController.podUpdated(nil, item.pod)
			nodeController.processPod(tCtx, podUpdateItem{name: item.pod.Name, namespace: item.pod.Namespace})

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

func newNodeLifecycleControllerWithLivenessSource(
	ctx context.Context,
	kubeClient clientset.Interface,
	nodeLivenessSource string,
) (*nodeLifecycleController, error) {
	factory := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())

	leaseInformer := factory.Coordination().V1().Leases()
	nodeInformer := factory.Core().V1().Nodes()
	daemonSetInformer := factory.Apps().V1().DaemonSets()
	podInformer := factory.Core().V1().Pods()

	nc, err := NewNodeLifecycleController(ctx, ControllerOptions{
		LeaseInformer:               leaseInformer,
		PodInformer:                 podInformer,
		NodeInformer:                nodeInformer,
		DaemonSetInformer:           daemonSetInformer,
		KubeClient:                  kubeClient,
		NodeMonitorPeriod:           testNodeMonitorPeriod,
		NodeStartupGracePeriod:      testNodeStartupGracePeriod,
		NodeMonitorGracePeriod:      testNodeMonitorGracePeriod,
		EvictionLimiterQPS:          testRateLimiterQPS,
		SecondaryEvictionLimiterQPS: testRateLimiterQPS,
		LargeClusterThreshold:       testLargeClusterThreshold,
		UnhealthyZoneThreshold:      testUnhealthyThreshold,
		NodeLivenessSource:          nodeLivenessSource,
	})
	if err != nil {
		return nil, err
	}

	nc.leaseInformerSynced = alwaysReady
	nc.podInformerSynced = alwaysReady
	nc.nodeInformerSynced = alwaysReady
	nc.daemonSetInformerSynced = alwaysReady

	return &nodeLifecycleController{nc, leaseInformer, nodeInformer, daemonSetInformer, podInformer}, nil
}

func countNodeScopedEvents(events []*v1.Event, reason string, t *testing.T) int {
	t.Helper()
	count := 0
	for _, ev := range events {
		if ev.InvolvedObject.Kind == "Node" && ev.Reason == reason {
			if ev.Count != 1 {
				t.Errorf("expected Node event %q to have Count=1, got %d", reason, ev.Count)
			}
			count++
		}
	}
	return count
}

func TestNewNodeLifecycleControllerLivenessSource(t *testing.T) {
	tests := []struct {
		name        string
		gateEnabled bool
		source      string
		wantErr     string
		wantMetric  string
	}{
		{
			name:        "empty defaults to kubelet when gate disabled",
			gateEnabled: false,
			source:      "",
			wantMetric: `
# HELP node_collector_liveness_source [ALPHA] Gauge set to 1 for the configured node liveness source.
# TYPE node_collector_liveness_source gauge
node_collector_liveness_source{source="kubelet"} 1
`,
		},
		{
			name:        "kubelet accepted when gate disabled",
			gateEnabled: false,
			source:      nodelifecycleconfig.NodeLivenessSourceKubelet,
			wantMetric: `
# HELP node_collector_liveness_source [ALPHA] Gauge set to 1 for the configured node liveness source.
# TYPE node_collector_liveness_source gauge
node_collector_liveness_source{source="kubelet"} 1
`,
		},
		{
			name:        "external rejected when gate disabled",
			gateEnabled: false,
			source:      nodelifecycleconfig.NodeLivenessSourceExternal,
			wantErr:     string(features.ExternalNodeLivenessDetection),
		},
		{
			name:        "external accepted when gate enabled and sets only external label",
			gateEnabled: true,
			source:      nodelifecycleconfig.NodeLivenessSourceExternal,
			wantMetric: `
# HELP node_collector_liveness_source [ALPHA] Gauge set to 1 for the configured node liveness source.
# TYPE node_collector_liveness_source gauge
node_collector_liveness_source{source="external"} 1
`,
		},
		{
			name:        "invalid value rejected when gate enabled",
			gateEnabled: true,
			source:      "unknown-source",
			wantErr:     "invalid node liveness source",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExternalNodeLivenessDetection, tc.gateEnabled)
			tCtx := ktesting.Init(t)
			_, err := newNodeLifecycleControllerWithLivenessSource(tCtx, fake.NewSimpleClientset(), tc.source)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got %v", tc.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if err := metricstestutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.wantMetric), "node_collector_liveness_source"); err != nil {
				t.Fatalf("unexpected metric output: %v", err)
			}
		})
	}
}

func TestNodeHealthDataDeepCopyAndDisruptionPreservesLastReady(t *testing.T) {
	now := metav1.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	orig := &nodeHealthData{
		probeTimestamp:           now,
		readyTransitionTimestamp: now,
		status:                   &v1.NodeStatus{},
		lastReady: &v1.NodeCondition{
			Type:               v1.NodeReady,
			Status:             v1.ConditionUnknown,
			Reason:             "ExternalDetector",
			LastTransitionTime: now,
		},
	}

	copied := orig.deepCopy()
	if !apiequality.Semantic.DeepEqual(orig.lastReady, copied.lastReady) {
		t.Fatalf("deepCopy did not copy lastReady: got %+v, want %+v", copied.lastReady, orig.lastReady)
	}
	copied.lastReady.Status = v1.ConditionTrue
	if orig.lastReady.Status != v1.ConditionUnknown {
		t.Fatalf("mutating deepCopy mutated original lastReady: got %v, want %v", orig.lastReady.Status, v1.ConditionUnknown)
	}

	withoutLastReady := (&nodeHealthData{probeTimestamp: now}).deepCopy()
	if withoutLastReady.lastReady != nil {
		t.Fatalf("expected nil lastReady to remain nil on deepCopy, got %+v", withoutLastReady.lastReady)
	}

	// Verify handleDisruption preserves lastReady when exiting full disruption.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExternalNodeLivenessDetection, true)
	tCtx := ktesting.Init(t)
	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node0", CreationTimestamp: now},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{Type: v1.NodeReady, Status: v1.ConditionTrue, LastHeartbeatTime: now, LastTransitionTime: now},
					},
				},
			},
		},
		Clientset: fake.NewSimpleClientset(),
	}
	nc, err := newNodeLifecycleControllerWithLivenessSource(tCtx, fakeNodeHandler, nodelifecycleconfig.NodeLivenessSourceExternal)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	later := metav1.NewTime(now.Add(10 * time.Second))
	nc.now = func() metav1.Time { return later }
	nc.addPodEvictorForNewZone(klog.FromContext(tCtx), fakeNodeHandler.Existing[0])
	nc.zoneStates[""] = stateFullDisruption
	expectedLastReady := &v1.NodeCondition{
		Type:               v1.NodeReady,
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  now,
		LastTransitionTime: now,
	}
	nc.nodeHealthMap.set("node0", &nodeHealthData{
		probeTimestamp:           now,
		readyTransitionTimestamp: now,
		status:                   &fakeNodeHandler.Existing[0].Status,
		lastReady:                expectedLastReady.DeepCopy(),
	})

	nc.handleDisruption(tCtx, map[string][]*v1.NodeCondition{"": {expectedLastReady}}, fakeNodeHandler.Existing)

	after := nc.nodeHealthMap.getDeepCopy("node0")
	if !apiequality.Semantic.DeepEqual(after.lastReady, expectedLastReady) {
		t.Fatalf("handleDisruption lost lastReady: got %+v, want %+v", after.lastReady, expectedLastReady)
	}
	if !after.probeTimestamp.Equal(&later) {
		t.Fatalf("expected probeTimestamp updated to %v, got %v", later, after.probeTimestamp)
	}
}

func TestMonitorNodeHealthExternalLiveness(t *testing.T) {
	type pass struct {
		timeToPass    time.Duration
		nodes         []*v1.Node
		lease         *coordv1.Lease
		resetPodReady bool
	}

	t0 := metav1.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	tMinus5m := metav1.NewTime(t0.Add(-5 * time.Minute))
	tMinus2m := metav1.NewTime(t0.Add(-2 * time.Minute))
	tPlus5s := metav1.NewTime(t0.Add(5 * time.Second))
	tPlus9s := metav1.NewTime(t0.Add(9 * time.Second))
	tPlus10s := metav1.NewTime(t0.Add(10 * time.Second))
	tPlus20s := metav1.NewTime(t0.Add(20 * time.Second))
	tPlus30s := metav1.NewTime(t0.Add(30 * time.Second))

	makeNode := func(status v1.ConditionStatus, reason string, lastHeartbeat, lastTransition metav1.Time, taints ...v1.Taint) []*v1.Node {
		return []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					UID:               "node0-uid",
					CreationTimestamp: t0,
				},
				Spec: v1.NodeSpec{
					Taints: taints,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             status,
							Reason:             reason,
							LastHeartbeatTime:  lastHeartbeat,
							LastTransitionTime: lastTransition,
						},
						{
							Type:               v1.NodeDiskPressure,
							Status:             v1.ConditionFalse,
							LastHeartbeatTime:  lastHeartbeat,
							LastTransitionTime: t0,
						},
					},
				},
			},
		}
	}

	tests := []struct {
		name                         string
		livenessSource               string
		passes                       []pass
		updateReactor                func(action testcore.Action) (bool, runtime.Object, error)
		getPodsWrapper               func(c *fake.Clientset) func(string) ([]*v1.Pod, error)
		wantNodeStatusUpdates        int
		wantNodeStatusReadyCondition *v1.ConditionStatus
		wantNodeStatusReadyReason    string
		wantPodStatusUpdates         int
		wantNodeNotReadyEvents       int
		wantUnreachableTaintRemoved  bool
	}{
		{
			name:           "grace period expiry does not mark node Unknown in external mode",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionTrue, "KubeletReady", t0, t0),
				},
				{
					timeToPass: time.Hour,
					nodes:      makeNode(v1.ConditionTrue, "KubeletReady", t0, t0),
				},
			},
			wantNodeStatusUpdates:  0,
			wantPodStatusUpdates:   0,
			wantNodeNotReadyEvents: 0,
		},
		{
			name:           "True to Unknown without heartbeat change marks pods NotReady and emits one NodeNotReady event",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionTrue, "KubeletReady", t0, t0),
				},
				{
					timeToPass: 10 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus10s),
				},
				{
					timeToPass: 20 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus10s),
				},
			},
			wantNodeStatusUpdates:  0,
			wantPodStatusUpdates:   1,
			wantNodeNotReadyEvents: 1,
		},
		{
			name:           "True to Unknown with heartbeat change marks pods NotReady and emits one NodeNotReady event",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionTrue, "KubeletReady", t0, t0),
				},
				{
					timeToPass: 10 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", tPlus10s, tPlus10s),
				},
			},
			wantNodeStatusUpdates:  0,
			wantPodStatusUpdates:   1,
			wantNodeNotReadyEvents: 1,
		},
		{
			name:           "False to Unknown without heartbeat change marks pods NotReady and emits one NodeNotReady event",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionFalse, "KubeletNotReady", t0, t0),
				},
				{
					timeToPass: 10 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus10s),
				},
			},
			wantNodeStatusUpdates:  0,
			wantPodStatusUpdates:   1,
			wantNodeNotReadyEvents: 1,
		},
		{
			name:           "True to False does not mark pods NotReady or emit NodeNotReady event",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionTrue, "KubeletReady", t0, t0),
				},
				{
					timeToPass: 10 * time.Second,
					nodes:      makeNode(v1.ConditionFalse, "KubeletNotReady", tPlus10s, tPlus10s),
				},
				{
					timeToPass: time.Hour,
					nodes:      makeNode(v1.ConditionFalse, "KubeletNotReady", tPlus10s, tPlus10s),
				},
			},
			wantNodeStatusUpdates:  0,
			wantPodStatusUpdates:   0,
			wantNodeNotReadyEvents: 0,
		},
		{
			name:           "recovery back to True clears unreachable taint and allows subsequent Unknown transition",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionTrue, "KubeletReady", t0, t0),
				},
				{
					timeToPass: 10 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus10s),
				},
				{
					timeToPass:    20 * time.Second,
					nodes:         makeNode(v1.ConditionTrue, "KubeletReady", t0, tPlus20s, *UnreachableTaintTemplate),
					resetPodReady: true,
				},
				{
					timeToPass: 30 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus30s),
				},
			},
			wantNodeStatusUpdates:       0,
			wantPodStatusUpdates:        2,
			wantNodeNotReadyEvents:      2,
			wantUnreachableTaintRemoved: true,
		},
		{
			name:           "fast Unknown->True->Unknown flip with newer LastTransitionTime triggers event and marks pods NotReady",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionTrue, "KubeletReady", t0, t0),
				},
				{
					timeToPass: 5 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus5s),
				},
				{
					// Node recovered to True (kubelet reconciled PodReady=True) and flipped back to Unknown before Pass 3.
					timeToPass:    10 * time.Second,
					nodes:         makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus9s),
					resetPodReady: true,
				},
				{
					// Steady Unknown with unchanged LastTransitionTime is a no-op even if a pod became Ready.
					timeToPass:    15 * time.Second,
					nodes:         makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus9s),
					resetPodReady: true,
				},
			},
			wantNodeStatusUpdates:  0,
			wantPodStatusUpdates:   2,
			wantNodeNotReadyEvents: 2,
		},
		{
			name:           "first pass after restart with Ready=Unknown marks pods NotReady without emitting NodeNotReady event",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", tMinus5m, tMinus2m),
				},
				{
					timeToPass: 5 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", tMinus5m, tMinus2m),
				},
			},
			wantNodeStatusUpdates:  0,
			wantPodStatusUpdates:   1,
			wantNodeNotReadyEvents: 0,
		},
		{
			name:           "first pass after restart with Ready=False leaves pods untouched",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionFalse, "KubeletNotReady", tMinus5m, tMinus2m),
				},
				{
					timeToPass: 5 * time.Second,
					nodes:      makeNode(v1.ConditionFalse, "KubeletNotReady", tMinus5m, tMinus2m),
				},
			},
			wantNodeStatusUpdates:  0,
			wantPodStatusUpdates:   0,
			wantNodeNotReadyEvents: 0,
		},
		{
			name:           "never-Ready node is marked Unknown after startup grace period even when Lease is renewed",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			passes: []pass{
				{
					timeToPass: 0,
					nodes: []*v1.Node{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name:              "node0",
								UID:               "node0-uid",
								CreationTimestamp: t0,
							},
						},
					},
					lease: createNodeLease("node0", metav1.NewMicroTime(t0.Time)),
				},
				{
					timeToPass: 70 * time.Second,
					nodes: []*v1.Node{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name:              "node0",
								UID:               "node0-uid",
								CreationTimestamp: t0,
							},
						},
					},
					lease: createNodeLease("node0", metav1.NewMicroTime(t0.Add(70*time.Second))),
				},
			},
			wantNodeStatusUpdates:        1,
			wantNodeStatusReadyCondition: ptr.To(v1.ConditionUnknown),
			wantNodeStatusReadyReason:    "NodeStatusNeverUpdated",
			wantPodStatusUpdates:         1,
			wantNodeNotReadyEvents:       0,
		},
		{
			name:           "pod list error on external transition queues nodesToRetry and marks pods NotReady on next pass",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			getPodsWrapper: func(c *fake.Clientset) func(string) ([]*v1.Pod, error) {
				calls := 0
				base := fakeGetPodsAssignedToNode(c)
				return func(nodeName string) ([]*v1.Pod, error) {
					calls++
					if calls == 2 {
						return nil, fmt.Errorf("injected pod list error")
					}
					return base(nodeName)
				}
			},
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionTrue, "KubeletReady", t0, t0),
				},
				{
					timeToPass: 10 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus10s),
				},
				{
					timeToPass: 15 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus10s),
				},
			},
			wantNodeStatusUpdates:  0,
			wantPodStatusUpdates:   1,
			wantNodeNotReadyEvents: 0,
		},
		{
			name:           "MarkPodsNotReady error on external transition emits one NodeNotReady event and retries on next pass",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceExternal,
			updateReactor: func() func(action testcore.Action) (bool, runtime.Object, error) {
				calls := 0
				return func(action testcore.Action) (bool, runtime.Object, error) {
					if action.GetVerb() == "update" && action.GetResource().Resource == "pods" && action.GetSubresource() == "status" {
						calls++
						if calls == 1 {
							return true, nil, fmt.Errorf("injected pod status update error")
						}
						return false, nil, nil
					}
					return false, nil, nil
				}
			}(),
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionTrue, "KubeletReady", t0, t0),
				},
				{
					timeToPass: 10 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus10s),
				},
				{
					timeToPass: 15 * time.Second,
					nodes:      makeNode(v1.ConditionUnknown, "ExternalDetectorUnreachable", t0, tPlus10s),
				},
			},
			wantNodeStatusUpdates:  0,
			wantPodStatusUpdates:   2,
			wantNodeNotReadyEvents: 1,
		},
		{
			name:           "kubelet mode behavior unchanged when ExternalNodeLivenessDetection gate is enabled",
			livenessSource: nodelifecycleconfig.NodeLivenessSourceKubelet,
			passes: []pass{
				{
					timeToPass: 0,
					nodes:      makeNode(v1.ConditionTrue, "KubeletReady", t0, t0),
				},
				{
					timeToPass: time.Hour,
					nodes:      makeNode(v1.ConditionTrue, "KubeletReady", t0, t0),
				},
			},
			wantNodeStatusUpdates:        1,
			wantNodeStatusReadyCondition: ptr.To(v1.ConditionUnknown),
			wantNodeStatusReadyReason:    "NodeStatusUnknown",
			wantPodStatusUpdates:         1,
			wantNodeNotReadyEvents:       1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExternalNodeLivenessDetection, true)
			tCtx := ktesting.Init(t)

			fakeNodeHandler := &testutil.FakeNodeHandler{
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			}
			if tc.updateReactor != nil {
				fakeNodeHandler.Clientset.PrependReactor("update", "pods", tc.updateReactor)
			}

			nc, err := newNodeLifecycleControllerWithLivenessSource(tCtx, fakeNodeHandler, tc.livenessSource)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			fakeRecorder := testutil.NewFakeRecorder()
			nc.recorder = fakeRecorder
			if tc.getPodsWrapper != nil {
				nc.getPodsAssignedToNode = tc.getPodsWrapper(fakeNodeHandler.Clientset)
			} else {
				nc.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)
			}

			resetPodUpdates := 0
			removedUnreachableTaint := false
			for i, p := range tc.passes {
				if p.resetPodReady {
					readyPod := testutil.NewPod("pod0", "node0")
					if _, err := fakeNodeHandler.Clientset.CoreV1().Pods(readyPod.Namespace).UpdateStatus(tCtx, readyPod, metav1.UpdateOptions{}); err != nil {
						t.Fatalf("pass %d: failed to reset pod ready status: %v", i, err)
					}
					resetPodUpdates++
				}
				nc.now = func() metav1.Time { return metav1.NewTime(t0.Add(p.timeToPass)) }
				fakeNodeHandler.UpdatedNodes = nil
				fakeNodeHandler.Existing = p.nodes
				if err := nc.syncNodeStore(fakeNodeHandler); err != nil {
					t.Fatalf("pass %d: syncNodeStore failed: %v", i, err)
				}
				if p.lease != nil {
					if err := nc.syncLeaseStore(p.lease); err != nil {
						t.Fatalf("pass %d: syncLeaseStore failed: %v", i, err)
					}
				}
				if err := nc.monitorNodeHealth(tCtx); err != nil {
					t.Fatalf("pass %d: monitorNodeHealth failed: %v", i, err)
				}
				if len(fakeNodeHandler.UpdatedNodes) > 0 {
					latestNode := fakeNodeHandler.UpdatedNodes[len(fakeNodeHandler.UpdatedNodes)-1]
					if !taintutils.TaintExists(latestNode.Spec.Taints, UnreachableTaintTemplate) {
						removedUnreachableTaint = true
					}
				}
			}

			if len(fakeNodeHandler.UpdatedNodeStatuses) != tc.wantNodeStatusUpdates {
				t.Errorf("got %d node status updates, want %d", len(fakeNodeHandler.UpdatedNodeStatuses), tc.wantNodeStatusUpdates)
			}
			if tc.wantNodeStatusReadyCondition != nil && len(fakeNodeHandler.UpdatedNodeStatuses) > 0 {
				_, readyCond := controllerutil.GetNodeCondition(&fakeNodeHandler.UpdatedNodeStatuses[len(fakeNodeHandler.UpdatedNodeStatuses)-1].Status, v1.NodeReady)
				if readyCond == nil {
					t.Fatalf("expected NodeReady condition on updated node status, got nil")
				}
				if readyCond.Status != *tc.wantNodeStatusReadyCondition || readyCond.Reason != tc.wantNodeStatusReadyReason {
					t.Errorf("got NodeReady=(%s, %q), want (%s, %q)",
						readyCond.Status, readyCond.Reason, *tc.wantNodeStatusReadyCondition, tc.wantNodeStatusReadyReason)
				}
			}

			if tc.wantUnreachableTaintRemoved && !removedUnreachableTaint {
				t.Errorf("expected unreachable taint to be removed on recovery")
			}

			podStatusUpdates := 0
			for _, action := range fakeNodeHandler.Actions() {
				if action.GetVerb() == "update" && action.GetResource().Resource == "pods" && action.GetSubresource() == "status" {
					podStatusUpdates++
				}
			}
			podStatusUpdates -= resetPodUpdates
			if podStatusUpdates != tc.wantPodStatusUpdates {
				t.Errorf("got %d pod status updates, want %d", podStatusUpdates, tc.wantPodStatusUpdates)
			}

			gotEvents := countNodeScopedEvents(fakeRecorder.Events, "NodeNotReady", t)
			if gotEvents != tc.wantNodeNotReadyEvents {
				t.Errorf("got %d Node-scoped NodeNotReady events, want %d (all events: %+v)", gotEvents, tc.wantNodeNotReadyEvents, fakeRecorder.Events)
			}
		})
	}
}

func TestExternalLivenessBypassesLeaseCircuitBreaker(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExternalNodeLivenessDetection, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeControllerLeaseCircuitBreaker, true)

	tCtx := ktesting.Init(t)
	t0 := metav1.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	createdLongAgo := metav1.NewTime(t0.Add(-time.Hour))

	// node0 has never reported Ready and has a stale Lease in the informer cache (RV 50)
	// vs a fresher Lease in the API server (RV 100). In kubelet mode, node0's startup
	// grace expiry would perform a live Lease GET and trip the controller-wide circuit
	// breaker, stalling node1 on the following pass.
	fakeNodeHandler := &testutil.FakeNodeHandler{
		Existing: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					UID:               "node0-uid",
					CreationTimestamp: createdLongAgo,
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node1",
					UID:               "node1-uid",
					CreationTimestamp: createdLongAgo,
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionTrue,
							LastHeartbeatTime:  t0,
							LastTransitionTime: t0,
						},
					},
				},
			},
		},
		Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod1", "node1")}}),
	}

	liveLease := createNodeLease("node0", metav1.NewMicroTime(t0.Time))
	liveLease.ResourceVersion = "100"
	if _, err := fakeNodeHandler.Clientset.CoordinationV1().Leases(v1.NamespaceNodeLease).Create(tCtx, liveLease, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create live lease: %v", err)
	}
	fakeNodeHandler.Clientset.ClearActions()

	nc, err := newNodeLifecycleControllerWithLivenessSource(tCtx, fakeNodeHandler, nodelifecycleconfig.NodeLivenessSourceExternal)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeRecorder := testutil.NewFakeRecorder()
	nc.recorder = fakeRecorder
	nc.getPodsAssignedToNode = fakeGetPodsAssignedToNode(fakeNodeHandler.Clientset)
	nc.now = func() metav1.Time { return t0 }

	cachedLease := createNodeLease("node0", metav1.NewMicroTime(createdLongAgo.Time))
	cachedLease.ResourceVersion = "50"
	if err := nc.syncLeaseStore(cachedLease); err != nil {
		t.Fatalf("syncLeaseStore failed: %v", err)
	}
	if err := nc.syncNodeStore(fakeNodeHandler); err != nil {
		t.Fatalf("syncNodeStore failed: %v", err)
	}

	// Pass 1: node0 exceeds startup grace period and gets marked Unknown without a live Lease GET.
	if err := nc.monitorNodeHealth(tCtx); err != nil {
		t.Fatalf("pass 1 monitorNodeHealth failed: %v", err)
	}
	if len(fakeNodeHandler.UpdatedNodeStatuses) != 1 || fakeNodeHandler.UpdatedNodeStatuses[0].Name != "node0" {
		t.Fatalf("expected node0 status update to Unknown on pass 1, got %+v", fakeNodeHandler.UpdatedNodeStatuses)
	}

	// Pass 2: node1 transitions externally to Ready=Unknown; processing must not stall.
	tPlus10s := metav1.NewTime(t0.Add(10 * time.Second))
	nc.now = func() metav1.Time { return tPlus10s }
	fakeNodeHandler.Existing[0] = fakeNodeHandler.UpdatedNodeStatuses[0].DeepCopy()
	fakeNodeHandler.Existing[1].Status.Conditions[0].Status = v1.ConditionUnknown
	fakeNodeHandler.Existing[1].Status.Conditions[0].Reason = "ExternalDetectorUnreachable"
	fakeNodeHandler.Existing[1].Status.Conditions[0].LastTransitionTime = tPlus10s
	if err := nc.syncNodeStore(fakeNodeHandler); err != nil {
		t.Fatalf("pass 2 syncNodeStore failed: %v", err)
	}
	if err := nc.monitorNodeHealth(tCtx); err != nil {
		t.Fatalf("pass 2 monitorNodeHealth failed: %v", err)
	}

	leaseGets := 0
	podStatusUpdates := 0
	for _, action := range fakeNodeHandler.Actions() {
		if action.GetVerb() == "get" && action.GetResource().Resource == "leases" {
			leaseGets++
		}
		if action.GetVerb() == "update" && action.GetResource().Resource == "pods" && action.GetSubresource() == "status" {
			podStatusUpdates++
		}
	}
	if leaseGets != 0 {
		t.Errorf("expected 0 live Lease GET calls in external mode, got %d", leaseGets)
	}
	if podStatusUpdates != 1 {
		t.Errorf("expected pod1 to be marked NotReady without circuit-breaker stall, got %d pod status updates", podStatusUpdates)
	}
	if got := countNodeScopedEvents(fakeRecorder.Events, "NodeNotReady", t); got != 1 {
		t.Errorf("expected 1 Node-scoped NodeNotReady event for node1, got %d", got)
	}
}

func TestProcessPodMarkPodNotReadyExternalLiveness(t *testing.T) {
	t0 := metav1.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	tPlus10s := metav1.NewTime(t0.Add(10 * time.Second))

	tests := []struct {
		name               string
		nodeStatuses       []v1.ConditionStatus
		wantPodMarkedUnrdy bool
	}{
		{
			name:               "node health not gathered yet skips pod update",
			nodeStatuses:       nil,
			wantPodMarkedUnrdy: false,
		},
		{
			// Pass 1 sees Ready=True (stored in nodeHealth.status); Pass 2 sees Ready=Unknown
			// without changing LastHeartbeatTime, so nodeHealth.status stays Ready=True while
			// lastReady becomes Ready=Unknown. processPod must read lastReady and mark pod NotReady.
			name:               "marks newly assigned pod NotReady when lastReady is Unknown despite stale Ready=True in status",
			nodeStatuses:       []v1.ConditionStatus{v1.ConditionTrue, v1.ConditionUnknown},
			wantPodMarkedUnrdy: true,
		},
		{
			// Pass 1 sees Ready=Unknown (stored in nodeHealth.status); Pass 2 sees Ready=True
			// without changing LastHeartbeatTime, so nodeHealth.status stays Ready=Unknown while
			// lastReady becomes Ready=True. processPod must read lastReady and keep pod Ready.
			name:               "does not mark newly assigned pod NotReady when lastReady is True despite stale Ready=Unknown in status",
			nodeStatuses:       []v1.ConditionStatus{v1.ConditionUnknown, v1.ConditionTrue},
			wantPodMarkedUnrdy: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExternalNodeLivenessDetection, true)
			tCtx := ktesting.Init(t)

			newPod := testutil.NewPod("pod-new", "node0")
			fakeNodeHandler := &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:              "node0",
							CreationTimestamp: t0,
						},
						Status: v1.NodeStatus{
							Conditions: []v1.NodeCondition{
								{
									Type:               v1.NodeReady,
									Status:             v1.ConditionFalse,
									LastHeartbeatTime:  t0,
									LastTransitionTime: t0,
								},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*newPod}}),
			}

			nc, err := newNodeLifecycleControllerWithLivenessSource(tCtx, fakeNodeHandler, nodelifecycleconfig.NodeLivenessSourceExternal)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			nc.now = func() metav1.Time { return t0 }
			nc.recorder = testutil.NewFakeRecorder()
			// Keep monitorNodeHealth from touching pod-new so we isolate processPod.
			nc.getPodsAssignedToNode = func(string) ([]*v1.Pod, error) { return nil, nil }

			if err := nc.syncNodeStore(fakeNodeHandler); err != nil {
				t.Fatalf("syncNodeStore failed: %v", err)
			}
			for i, st := range tc.nodeStatuses {
				transitionTime := t0
				if i > 0 {
					transitionTime = tPlus10s
				}
				fakeNodeHandler.Existing[0].Status.Conditions[0].Status = st
				// Keep LastHeartbeatTime constant at t0 across passes so nodeHealth.status is never refreshed.
				fakeNodeHandler.Existing[0].Status.Conditions[0].LastHeartbeatTime = t0
				fakeNodeHandler.Existing[0].Status.Conditions[0].LastTransitionTime = transitionTime
				if err := nc.syncNodeStore(fakeNodeHandler); err != nil {
					t.Fatalf("pass %d syncNodeStore failed: %v", i, err)
				}
				if err := nc.monitorNodeHealth(tCtx); err != nil {
					t.Fatalf("pass %d monitorNodeHealth failed: %v", i, err)
				}
			}
			fakeNodeHandler.Clientset.ClearActions()

			if err := nc.syncPodStore(newPod); err != nil {
				t.Fatalf("syncPodStore failed: %v", err)
			}
			nc.podUpdated(nil, newPod)
			nc.processPod(tCtx, podUpdateItem{name: newPod.Name, namespace: newPod.Namespace})

			podStatusUpdated := false
			for _, action := range fakeNodeHandler.Actions() {
				if action.GetVerb() == "update" && action.GetResource().Resource == "pods" && action.GetSubresource() == "status" {
					podStatusUpdated = true
				}
			}
			if podStatusUpdated != tc.wantPodMarkedUnrdy {
				t.Errorf("got podStatusUpdated=%v, want %v", podStatusUpdated, tc.wantPodMarkedUnrdy)
			}
		})
	}
}
