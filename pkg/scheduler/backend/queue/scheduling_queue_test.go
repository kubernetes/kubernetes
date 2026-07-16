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

package queue

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	plfeature "k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/schedulinggates"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	testingclock "k8s.io/utils/clock/testing"
)

const queueMetricMetadata = `
		# HELP scheduler_queue_incoming_pods_total [STABLE] Number of pods added to scheduling queues by event and queue type.
		# TYPE scheduler_queue_incoming_pods_total counter
	`

var (
	// nodeAdd is the event when a new node is added to the cluster.
	nodeAdd = fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add}
	// pvAdd is the event when a persistent volume is added in the cluster.
	pvAdd = fwk.ClusterEvent{Resource: fwk.PersistentVolume, ActionType: fwk.Add}
	// pvUpdate is the event when a persistent volume is updated in the cluster.
	pvUpdate = fwk.ClusterEvent{Resource: fwk.PersistentVolume, ActionType: fwk.Update}
	// pvcAdd is the event when a persistent volume claim is added in the cluster.
	pvcAdd = fwk.ClusterEvent{Resource: fwk.PersistentVolumeClaim, ActionType: fwk.Add}
	// csiNodeUpdate is the event when a CSI node is updated in the cluster.
	csiNodeUpdate = fwk.ClusterEvent{Resource: fwk.CSINode, ActionType: fwk.Update}

	lowPriority, midPriority, highPriority = int32(0), int32(100), int32(1000)
	mediumPriority                         = (lowPriority + highPriority) / 2

	highPriorityPodInfo = mustNewPodInfo(
		st.MakePod().Name("hpp").Namespace("ns1").UID("hppns1").Priority(highPriority).Obj(),
	)
	highPriNominatedPodInfo = mustNewPodInfo(
		st.MakePod().Name("hpp").Namespace("ns1").UID("hppns1").Priority(highPriority).NominatedNodeName("node1").Obj(),
	)
	medPriorityPodInfo = mustNewPodInfo(
		st.MakePod().Name("mpp").Namespace("ns2").UID("mppns2").Annotation("annot2", "val2").Priority(mediumPriority).NominatedNodeName("node1").Obj(),
	)
	unschedulablePodInfo = mustNewPodInfo(
		st.MakePod().Name("up").Namespace("ns1").UID("upns1").Annotation("annot2", "val2").Priority(lowPriority).NominatedNodeName("node1").Condition(v1.PodScheduled, v1.ConditionFalse, v1.PodReasonUnschedulable).Obj(),
	)
	nonExistentPodInfo = mustNewPodInfo(
		st.MakePod().Name("ne").Namespace("ns1").UID("nens1").Obj(),
	)
	scheduledPodInfo = mustNewPodInfo(
		st.MakePod().Name("sp").Namespace("ns1").UID("spns1").Node("foo").Obj(),
	)

	nominatorCmpOpts = []cmp.Option{
		cmp.AllowUnexported(nominator{}, podRef{}),
		cmpopts.IgnoreFields(nominator{}, "podLister", "nLock"),
	}

	queueHintReturnQueue = func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
		return fwk.Queue, nil
	}
	queueHintReturnSkip = func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
		return fwk.QueueSkip, nil
	}
)

func init() {
	metrics.Register()
}

func setQueuedPodInfoGated(queuedPodInfo *framework.QueuedPodInfo, gatingPlugin string, gatingPluginEvents []fwk.ClusterEvent) *framework.QueuedPodInfo {
	queuedPodInfo.GatingPlugin = gatingPlugin
	// GatingPlugin should also be registered in UnschedulablePlugins.
	queuedPodInfo.UnschedulablePlugins = sets.New(gatingPlugin)
	queuedPodInfo.GatingPluginEvents = gatingPluginEvents
	return queuedPodInfo
}

func getUnschedulablePod(p *PriorityQueue, pod *v1.Pod) *v1.Pod {
	pInfo := p.unschedulableEntities.get(newQueuedPodInfoForLookup(pod))
	if pInfo != nil {
		return pInfo.(*framework.QueuedPodInfo).Pod
	}
	return nil
}

// makeEmptyQueueingHintMapPerProfile initializes an empty QueueingHintMapPerProfile for "" profile name.
func makeEmptyQueueingHintMapPerProfile() QueueingHintMapPerProfile {
	m := make(QueueingHintMapPerProfile)
	m[""] = make(QueueingHintMap)
	return m
}

func withPodGroupName(pod *v1.Pod, podGroupName string) *v1.Pod {
	pod = pod.DeepCopy()
	pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: &podGroupName}
	return pod
}

func TestPriorityQueue_Add(t *testing.T) {
	tests := []struct {
		name                   string
		usePodGroups           bool
		genericWorkloadEnabled bool
	}{
		{
			name:                   "individual pod with GenericWorkload gate disabled",
			usePodGroups:           false,
			genericWorkloadEnabled: false,
		},
		{
			name:                   "individual pod with GenericWorkload gate enabled",
			usePodGroups:           false,
			genericWorkloadEnabled: true,
		},
		{
			name:                   "pod group member with GenericWorkload gate enabled",
			usePodGroups:           true,
			genericWorkloadEnabled: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, tt.genericWorkloadEnabled)
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			var medPod, unschedPod, highPod = medPriorityPodInfo.Pod, unschedulablePodInfo.Pod, highPriorityPodInfo.Pod
			if tt.usePodGroups {
				medPod = withPodGroupName(medPod, "pg-med")
				unschedPod = withPodGroupName(unschedPod, "pg-unsched")
				highPod = withPodGroupName(highPod, "pg-high")
			}

			objs := []runtime.Object{medPod, unschedPod, highPod}
			q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), objs)
			if tt.usePodGroups {
				podGroups := []*schedulingv1alpha3.PodGroup{
					st.MakePodGroup().Name("pg-med").Namespace(medPod.Namespace).Priority(midPriority).Obj(),
					st.MakePodGroup().Name("pg-unsched").Namespace(unschedPod.Namespace).Priority(lowPriority).Obj(),
					st.MakePodGroup().Name("pg-high").Namespace(highPod.Namespace).Priority(highPriority).Obj(),
				}
				for _, podGroup := range podGroups {
					q.AddPodGroup(logger, podGroup)
				}
			}
			q.Add(ctx, medPod)
			q.Add(ctx, unschedPod)
			q.Add(ctx, highPod)
			expectedNominatedPods := &nominator{
				nominatedPodToNode: map[types.UID]string{
					medPod.UID:     "node1",
					unschedPod.UID: "node1",
				},
				nominatedPods: map[string][]podRef{
					"node1": {podToRef(medPod), podToRef(unschedPod)},
				},
			}
			if diff := cmp.Diff(q.nominator, expectedNominatedPods, nominatorCmpOpts...); diff != "" {
				t.Errorf("Unexpected diff after adding pods (-want, +got):\n%s", diff)
			}

			getPod := func(entity framework.QueuedEntityInfo) *v1.Pod {
				if tt.usePodGroups {
					return entity.(*framework.QueuedPodGroupInfo).QueuedPodInfos[0].Pod
				}
				return entity.(*framework.QueuedPodInfo).Pod
			}

			if entity, err := q.Pop(logger); err != nil {
				t.Errorf("Pop failed: %v", err)
			} else if diff := cmp.Diff(highPod, getPod(entity)); diff != "" {
				t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
			}
			if entity, err := q.Pop(logger); err != nil {
				t.Errorf("Pop failed: %v", err)
			} else if diff := cmp.Diff(medPod, getPod(entity)); diff != "" {
				t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
			}
			if entity, err := q.Pop(logger); err != nil {
				t.Errorf("Pop failed: %v", err)
			} else if diff := cmp.Diff(unschedPod, getPod(entity)); diff != "" {
				t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
			}
			if len(q.nominator.nominatedPods["node1"]) != 2 {
				t.Errorf("Expected medPod and unschedPod to be still present in nominatedPods: %v", q.nominator.nominatedPods["node1"])
			}
		})
	}
}

func TestPriorityQueue_AddNominatedGatedPod(t *testing.T) {
	gatedPod := st.MakePod().Name("pod-gated").Namespace("ns1").UID("pod-gated").NominatedNodeName("node1").Obj()
	objs := []runtime.Object{gatedPod}
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	plugin := &preEnqueuePlugin{allowlists: []string{"allow"}}
	m := map[string]map[string]fwk.PreEnqueuePlugin{
		"": {
			"preEnqueuePlugin": plugin,
		},
	}
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), objs, WithPreEnqueuePluginMap(m))
	q.Add(ctx, gatedPod)

	// Verify the pod is gated
	pInfo := q.unschedulableEntities.get(newQueuedPodInfoForLookup(gatedPod))
	if pInfo == nil || !pInfo.Gated() {
		t.Fatalf("Expected pod to be gated in unschedulableEntities")
	}

	// Verify the pod is added to nominator
	if len(q.nominator.nominatedPods["node1"]) != 1 {
		t.Errorf("Expected pod-gated in nominatedPods")
	}
	if q.nominator.nominatedPodToNode[gatedPod.UID] != "node1" {
		t.Errorf("Expected pod-gated in nominatedPodToNode")
	}
}

func newDefaultQueueSort() fwk.LessFunc {
	sort := &queuesort.PrioritySort{}
	return sort.Less
}

func TestPriorityQueue_AddWithReversePriorityLessFunc(t *testing.T) {
	objs := []runtime.Object{medPriorityPodInfo.Pod, highPriorityPodInfo.Pod}
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), objs)
	q.Add(ctx, medPriorityPodInfo.Pod)
	q.Add(ctx, highPriorityPodInfo.Pod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(highPriorityPodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(medPriorityPodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
}

func Test_InFlightPods(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	pod1 := st.MakePod().Name("targetpod").UID("pod1").Obj()
	pod2 := st.MakePod().Name("targetpod2").UID("pod2").Obj()
	pod3 := st.MakePod().Name("targetpod3").UID("pod3").Obj()

	pgName := "pg-test"
	pgPod1 := st.MakePod().Name("pgpod1").UID("pgpod1").PodGroupName(pgName).Obj()
	pgPod2 := st.MakePod().Name("pgpod2").UID("pgpod2").PodGroupName(pgName).Obj()

	var poppedPod, poppedPod2 *framework.QueuedPodInfo

	type action struct {
		// ONLY ONE of the following should be set.
		eventHappens *fwk.ClusterEvent
		podPopped    *v1.Pod
		// podCreated is the Pod that is created and inserted into the activeQ.
		podCreated *v1.Pod
		// podEnqueued is the Pod that is enqueued back to activeQ.
		podEnqueued *framework.QueuedPodInfo
		// podGroupAttempted is the PodGroup that was attempted to schedule.
		podGroupAttempted *framework.QueuedPodGroupInfo
		callback          func(t *testing.T, q *PriorityQueue)
	}

	tests := []struct {
		name            string
		queueingHintMap QueueingHintMapPerProfile
		// initialPods is the initial Pods in the activeQ.
		initialPods                []*v1.Pod
		actions                    []action
		genericWorkloadEnabled     []bool
		wantInFlightPods           []*v1.Pod
		wantInFlightEvents         []interface{}
		wantActiveQPodNames        []string
		wantBackoffQPodNames       []string
		wantUnschedPodPoolPodNames []string
	}{
		{
			name:        "Pod and interested events are registered in inFlightPods/inFlightEvents",
			initialPods: []*v1.Pod{pod1},
			actions: []action{
				// This won't be added to inFlightEvents because no inFlightPods at this point.
				{eventHappens: &pvcAdd},
				{podPopped: pod1},
				// This gets added for the pod.
				{eventHappens: &pvAdd},
				// This doesn't get added because no plugin is interested in PvUpdate.
				{eventHappens: &pvUpdate},
			},
			wantInFlightPods:   []*v1.Pod{pod1},
			wantInFlightEvents: []interface{}{pod1, pvAdd},
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					pvAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:        "Pod, registered in inFlightPods, is enqueued back to backoffQ",
			initialPods: []*v1.Pod{pod1, pod2},
			actions: []action{
				// This won't be added to inFlightEvents because no inFlightPods at this point.
				{eventHappens: &pvcAdd},
				{podPopped: pod1},
				{eventHappens: &pvAdd},
				{podPopped: pod2},
				{eventHappens: &nodeAdd},
				// This pod will be requeued to backoffQ immediately because no plugin is registered as unschedulable plugin,
				// which means the pod encountered an unexpected error (e.g., a network error).
				{podEnqueued: newQueuedPodInfoForLookup(pod1)},
			},
			wantBackoffQPodNames: []string{"targetpod"},
			wantInFlightPods:     []*v1.Pod{pod2}, // only pod2 is registered because pod is already enqueued back.
			wantInFlightEvents:   []interface{}{pod2, nodeAdd},
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					pvAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					nodeAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					pvcAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:        "All Pods registered in inFlightPods are enqueued back to activeQ",
			initialPods: []*v1.Pod{pod1, pod2},
			actions: []action{
				// This won't be added to inFlightEvents because no inFlightPods at this point.
				{eventHappens: &pvcAdd},
				{podPopped: pod1},
				{eventHappens: &pvAdd},
				{podPopped: pod2},
				{eventHappens: &nodeAdd},
				// This pod will be requeued to backoffQ immediately because no plugin is registered as unschedulable plugin,
				// which means the pod encountered an unexpected error (e.g., a network error).
				{podEnqueued: newQueuedPodInfoForLookup(pod1)},
				{eventHappens: &csiNodeUpdate},
				// This pod will be requeued to backoffQ immediately because no plugin is registered as unschedulable plugin,
				// which means the pod encountered an unexpected error (e.g., a network error).
				{podEnqueued: newQueuedPodInfoForLookup(pod2)},
			},
			wantBackoffQPodNames: []string{"targetpod", "targetpod2"},
			wantInFlightPods:     nil, // empty
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					pvAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					nodeAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					pvcAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					csiNodeUpdate: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:        "One intermediate Pod registered in inFlightPods is enqueued back to activeQ",
			initialPods: []*v1.Pod{pod1, pod2, pod3},
			actions: []action{
				// This won't be added to inFlightEvents because no inFlightPods at this point.
				{eventHappens: &pvcAdd},
				{podPopped: pod1},
				{eventHappens: &pvAdd},
				{podPopped: pod2},
				{eventHappens: &nodeAdd},
				// This Pod won't be requeued again.
				{podPopped: pod3},
				{eventHappens: &framework.EventAssignedPodAdd},
				// This pod will be requeued to backoffQ immediately because no plugin is registered as unschedulable plugin,
				// which means the pod encountered an unexpected error (e.g., a network error).
				{podEnqueued: newQueuedPodInfoForLookup(pod2)},
			},
			wantBackoffQPodNames: []string{"targetpod2"},
			wantInFlightPods:     []*v1.Pod{pod1, pod3},
			wantInFlightEvents:   []interface{}{pod1, pvAdd, nodeAdd, pod3, framework.EventAssignedPodAdd},
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					pvAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					nodeAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					framework.EventAssignedPodAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:        "events before popping Pod are ignored when Pod is enqueued back to queue",
			initialPods: []*v1.Pod{pod1},
			actions: []action{
				{eventHappens: &framework.EventUnschedulableTimeout},
				{podPopped: pod1},
				{eventHappens: &framework.EventAssignedPodAdd},
				// This Pod won't be requeued to activeQ/backoffQ because fooPlugin1 returns QueueSkip.
				{podEnqueued: newQueuedPodInfoForLookup(pod1, "fooPlugin1")},
			},
			wantUnschedPodPoolPodNames: []string{"targetpod"},
			wantInFlightPods:           nil,
			wantInFlightEvents:         nil,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					// fooPlugin1 has a queueing hint function for framework.AssignedPodAdd,
					// but hint fn tells that this event doesn't make a Pod scheudlable.
					framework.EventAssignedPodAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnSkip,
						},
					},
				},
			},
		},
		{
			name:        "pod is enqueued to backoff if no failed plugin",
			initialPods: []*v1.Pod{pod1},
			actions: []action{
				{podPopped: pod1},
				{eventHappens: &framework.EventAssignedPodAdd},
				// This pod will be requeued to backoffQ immediately because no plugin is registered as unschedulable plugin,
				// which means the pod encountered an unexpected error (e.g., a network error).
				{podEnqueued: newQueuedPodInfoForLookup(pod1)},
			},
			wantBackoffQPodNames: []string{"targetpod"},
			wantInFlightPods:     nil,
			wantInFlightEvents:   nil,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					// It will be ignored because no failed plugin.
					framework.EventAssignedPodAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:        "pod is enqueued to unschedulable pod pool if no events that can make the pod schedulable",
			initialPods: []*v1.Pod{pod1},
			actions: []action{
				{podPopped: pod1},
				{eventHappens: &nodeAdd},
				{podEnqueued: newQueuedPodInfoForLookup(pod1, "fooPlugin1")},
			},
			wantUnschedPodPoolPodNames: []string{"targetpod"},
			wantInFlightPods:           nil,
			wantInFlightEvents:         nil,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					// fooPlugin1 has no queueing hint function for NodeAdd.
					framework.EventAssignedPodAdd: {
						{
							// It will be ignored because the event is not NodeAdd.
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:        "pod is enqueued to unschedulable pod pool because the failed plugin has a hint fn but it returns Skip",
			initialPods: []*v1.Pod{pod1},
			actions: []action{
				{podPopped: pod1},
				{eventHappens: &framework.EventAssignedPodAdd},
				{podEnqueued: newQueuedPodInfoForLookup(pod1, "fooPlugin1")},
			},
			wantUnschedPodPoolPodNames: []string{"targetpod"},
			wantInFlightPods:           nil,
			wantInFlightEvents:         nil,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					// fooPlugin1 has a queueing hint function for framework.AssignedPodAdd,
					// but hint fn tells that this event doesn't make a Pod scheudlable.
					framework.EventAssignedPodAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnSkip,
						},
					},
				},
			},
		},
		{
			name:        "pod is enqueued to activeQ because the Pending plugins has a hint fn and it returns Queue",
			initialPods: []*v1.Pod{pod1},
			actions: []action{
				{podPopped: pod1},
				{eventHappens: &framework.EventAssignedPodAdd},
				{podEnqueued: &framework.QueuedPodInfo{
					PodInfo: mustNewPodInfo(pod1),
					QueueingParams: framework.QueueingParams{
						UnschedulablePlugins: sets.New("fooPlugin2", "fooPlugin3"),
						PendingPlugins:       sets.New("fooPlugin1"),
					},
				}},
			},
			wantActiveQPodNames: []string{"targetpod"},
			wantInFlightPods:    nil,
			wantInFlightEvents:  nil,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					framework.EventAssignedPodAdd: {
						{
							PluginName:     "fooPlugin3",
							QueueingHintFn: queueHintReturnSkip,
						},
						{
							PluginName:     "fooPlugin2",
							QueueingHintFn: queueHintReturnQueue,
						},
						{
							// The hint fn tells that this event makes a Pod scheudlable immediately.
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:        "pod is enqueued to backoffQ because the failed plugin has a hint fn and it returns Queue",
			initialPods: []*v1.Pod{pod1},
			actions: []action{
				{podPopped: pod1},
				{eventHappens: &framework.EventAssignedPodAdd},
				{podEnqueued: newQueuedPodInfoForLookup(pod1, "fooPlugin1", "fooPlugin2")},
			},
			wantBackoffQPodNames: []string{"targetpod"},
			wantInFlightPods:     nil,
			wantInFlightEvents:   nil,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					framework.EventAssignedPodAdd: {
						{
							// it will be ignored because the hint fn returns Skip that is weaker than queueHintReturnQueue from fooPlugin1.
							PluginName:     "fooPlugin2",
							QueueingHintFn: queueHintReturnSkip,
						},
						{
							// The hint fn tells that this event makes a Pod schedulable.
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:        "pod is enqueued to activeQ because the pending plugin has a hint fn and it returns Queue for a concurrent event that was received while some other pod was in flight",
			initialPods: []*v1.Pod{pod1, pod2},
			actions: []action{
				{callback: func(t *testing.T, q *PriorityQueue) { poppedPod = popPod(t, logger, q, pod1) }},
				{eventHappens: &nodeAdd},
				{callback: func(t *testing.T, q *PriorityQueue) { poppedPod2 = popPod(t, logger, q, pod2) }},
				{eventHappens: &framework.EventAssignedPodAdd},
				{callback: func(t *testing.T, q *PriorityQueue) {
					logger, _ := ktesting.NewTestContext(t)
					// This pod will be requeued to backoffQ immediately because no plugin is registered as unschedulable plugin,
					// which means the pod encountered an unexpected error (e.g., a network error).
					err := q.AddUnschedulablePodIfNotPresent(logger, poppedPod, q.SchedulingCycle())
					if err != nil {
						t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
					}
				}},
				{callback: func(t *testing.T, q *PriorityQueue) {
					logger, _ := ktesting.NewTestContext(t)
					poppedPod2.UnschedulablePlugins = sets.New("fooPlugin2", "fooPlugin3")
					poppedPod2.PendingPlugins = sets.New("fooPlugin1")
					err := q.AddUnschedulablePodIfNotPresent(logger, poppedPod2, q.SchedulingCycle())
					if err != nil {
						t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
					}
				}},
			},
			wantActiveQPodNames:  []string{pod2.Name},
			wantBackoffQPodNames: []string{pod1.Name},
			wantInFlightPods:     nil,
			wantInFlightEvents:   nil,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					framework.EventAssignedPodAdd: {
						{
							// it will be ignored because the hint fn returns QueueSkip that is weaker than queueHintReturnQueueImmediately from fooPlugin1.
							PluginName:     "fooPlugin3",
							QueueingHintFn: queueHintReturnSkip,
						},
						{
							// it will be ignored because the fooPlugin2 is registered in UnschedulablePlugins and it's interpret as Queue that is weaker than QueueImmediately from fooPlugin1.
							PluginName:     "fooPlugin2",
							QueueingHintFn: queueHintReturnQueue,
						},
						{
							// The hint fn tells that this event makes a Pod scheudlable.
							// Given fooPlugin1 is registered as Pendings, we interpret Queue as queueImmediately.
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:        "popped pod preserves UnschedulablePlugins and PendingPlugins",
			initialPods: []*v1.Pod{pod1},
			actions: []action{
				{callback: func(t *testing.T, q *PriorityQueue) { poppedPod = popPod(t, logger, q, pod1) }},
				{callback: func(t *testing.T, q *PriorityQueue) {
					logger, _ := ktesting.NewTestContext(t)
					// Unschedulable due to PendingPlugins.
					poppedPod.PendingPlugins = sets.New("fooPlugin1")
					poppedPod.UnschedulablePlugins = sets.New("fooPlugin2")
					if err := q.AddUnschedulablePodIfNotPresent(logger, poppedPod, q.SchedulingCycle()); err != nil {
						t.Errorf("Unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
					}
				}},
				{eventHappens: &pvAdd}, // Active again.
				{callback: func(t *testing.T, q *PriorityQueue) {
					poppedPod = popPod(t, logger, q, pod1)
					// UnschedulablePlugins should be preserved for logging/debugging
					if !poppedPod.GetUnschedulablePlugins().Equal(sets.New("fooPlugin2")) {
						t.Errorf("QueuedPodInfo from Pop should preserve UnschedulablePlugins, expected fooPlugin2, got: %+v", poppedPod.GetUnschedulablePlugins())
					}
					// PendingPlugins are preserved after Pop() for logging
					if !poppedPod.PendingPlugins.Equal(sets.New("fooPlugin1")) {
						t.Errorf("QueuedPodInfo from Pop should preserve PendingPlugins, expected fooPlugin1, got: %+v", poppedPod.PendingPlugins)
					}
				}},
				{callback: func(t *testing.T, q *PriorityQueue) {
					logger, _ := ktesting.NewTestContext(t)
					// Failed (i.e. no UnschedulablePlugins). Should go to backoff.
					if err := q.AddUnschedulablePodIfNotPresent(logger, poppedPod, q.SchedulingCycle()); err != nil {
						t.Errorf("Unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
					}
				}},
			},
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					pvAdd: {
						{
							// The hint fn tells that this event makes a Pod scheudlable immediately.
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			// This scenario shouldn't happen unless we make the similar bug like https://github.com/kubernetes/kubernetes/issues/118226.
			// But, given the bug could make a serious memory leak and likely would be hard to detect,
			// we should have a safe guard from the same bug so that, at least, we can prevent the memory leak.
			name:        "Pop is made twice for the same Pod, but the cleanup still happen correctly",
			initialPods: []*v1.Pod{pod1, pod2},
			actions: []action{
				// This won't be added to inFlightEvents because no inFlightPods at this point.
				{eventHappens: &pvcAdd},
				{podPopped: pod1},
				{eventHappens: &pvAdd},
				{podPopped: pod2},
				// Simulate a bug, putting pod into activeQ, while pod is being scheduled.
				{callback: func(t *testing.T, q *PriorityQueue) {
					q.activeQ.add(logger, newQueuedPodInfoForLookup(pod1), framework.EventUnscheduledPodAdd.Label())
				}},
				// At this point, in the activeQ, we have pod1 and pod3 in this order.
				{podCreated: pod3},
				// pod3 is poped, not pod1.
				// In detail, this Pop() first tries to pop pod1, but it's already being scheduled and hence discarded.
				// Then, it pops the next pod, pod3.
				{podPopped: pod3},
				{callback: func(t *testing.T, q *PriorityQueue) {
					// Make sure that pod1 is discarded and hence no pod in activeQ.
					if len(q.activeQ.list()) != 0 {
						t.Fatalf("activeQ should be empty, but got: %v", q.activeQ.list())
					}
				}},
				{eventHappens: &nodeAdd},
				// This pod will be requeued to backoffQ immediately because no plugin is registered as unschedulable plugin,
				// which means the pod encountered an unexpected error (e.g., a network error).
				{podEnqueued: newQueuedPodInfoForLookup(pod1)},
				{eventHappens: &csiNodeUpdate},
				// This pod will be requeued to backoffQ immediately because no plugin is registered as unschedulable plugin,
				// which means the pod encountered an unexpected error (e.g., a network error).
				{podEnqueued: newQueuedPodInfoForLookup(pod2)},
				// This pod will be requeued to backoffQ immediately because no plugin is registered as unschedulable plugin,
				// which means the pod encountered an unexpected error (e.g., a network error).
				{podEnqueued: newQueuedPodInfoForLookup(pod3)},
			},
			wantBackoffQPodNames: []string{"targetpod", "targetpod2", "targetpod3"},
			wantInFlightPods:     nil, // should be empty
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					pvAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					nodeAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					pvcAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					csiNodeUpdate: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:                   "Pod group members and interested events are registered in inFlightPods/inFlightEvents",
			genericWorkloadEnabled: []bool{true},
			initialPods:            []*v1.Pod{pgPod1, pgPod2},
			actions: []action{
				{podPopped: pgPod1}, // Pops group, so pgPod1 and pgPod2 are inFlight
				{eventHappens: &pvAdd},
			},
			wantInFlightPods:   []*v1.Pod{pgPod1, pgPod2},
			wantInFlightEvents: []any{pgPod1, pgPod2, pvAdd},
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					pvAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:                   "Pop is made twice for all pods in the group, but the cleanup still happens correctly",
			genericWorkloadEnabled: []bool{true},
			initialPods:            []*v1.Pod{pgPod1, pgPod2},
			actions: []action{
				// This won't be added to inFlightEvents because no inFlightPods at this point.
				{eventHappens: &pvcAdd},
				// Pop group, so pgPod1 and pgPod2 are inFlight.
				{podPopped: pgPod1},
				{eventHappens: &pvAdd},
				{podGroupAttempted: newQueuedPodGroupInfoForLookup(pgPod1)},
				// Simulate a bug: add pgPod1 and pgPod2 back to activeQ while pod group is in-flight.
				{podCreated: pgPod1},
				{podCreated: pgPod2},
				// At this point, in the activeQ, we have pod group (with pgPod1 and pgPod2) and pod3 in this order.
				{podCreated: pod3},
				// pod3 is poped, not pgPod1.
				// In detail, this Pop() first tries to pop pod group with pgPod1 and pgPod2, but it's already being scheduled and hence discarded.
				// Then, it pops the next pod, pod3.
				{podPopped: pod3},
				{callback: func(t *testing.T, q *PriorityQueue) {
					// Make sure that pod group is discarded and hence no entity in activeQ.
					if len(q.activeQ.list()) != 0 {
						t.Fatalf("activeQ should be empty, but got: %v", q.activeQ.list())
					}
				}},
				{eventHappens: &nodeAdd},
				// This pod will be requeued to activeQ because it's a pod group member and the queued pod group itself is missing.
				{podEnqueued: newQueuedPodInfoForLookup(pgPod1)},
				{eventHappens: &csiNodeUpdate},
				// This pod will be requeued to activeQ because it's a pod group member and the queued pod group itself is missing.
				{podEnqueued: newQueuedPodInfoForLookup(pgPod2)},
				// This pod will be requeued to backoffQ immediately because no plugin is registered as unschedulable plugin,
				// which means the pod encountered an unexpected error (e.g., a network error).
				{podEnqueued: newQueuedPodInfoForLookup(pod3)},
			},
			wantActiveQPodNames:  []string{"pgpod1", "pgpod2"},
			wantBackoffQPodNames: []string{"targetpod3"},
			wantInFlightPods:     nil, // should be empty
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					pvAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					nodeAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					pvcAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					csiNodeUpdate: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:                   "Pop is made twice for a subset (one) pod in the group, but the cleanup still happens correctly",
			genericWorkloadEnabled: []bool{true},
			initialPods:            []*v1.Pod{pgPod1},
			actions: []action{
				// This won't be added to inFlightEvents because no inFlightPods at this point.
				{eventHappens: &pvcAdd},
				// Pop group, so pgPod1 is inFlight.
				{podPopped: pgPod1},
				{eventHappens: &pvAdd},
				{podGroupAttempted: newQueuedPodGroupInfoForLookup(pgPod1)},
				// Simulate a bug: add pgPod1 back to activeQ while pod group is in-flight.
				{podCreated: pgPod1},
				// Add a new, pgPod2 to activeQ.
				{podCreated: pgPod2},
				// At this point, in the activeQ, we have pod group (with pgPod1 and pgPod2) and pod3 in this order.
				{podCreated: pod3},
				// pgPod2 is popped, while pgPod1 is discarded.
				{podPopped: pgPod2},
				// pod3 is poped.
				{podPopped: pod3},
				{callback: func(t *testing.T, q *PriorityQueue) {
					// Make sure that pgPod1 was discarded and hence no entity in activeQ.
					if len(q.activeQ.list()) != 0 {
						t.Fatalf("activeQ should be empty, but got: %v", q.activeQ.list())
					}
				}},
				{eventHappens: &nodeAdd},
				// This pod will be requeued to activeQ because it's a pod group member and the queued pod group itself is missing.
				{podEnqueued: newQueuedPodInfoForLookup(pgPod1)},
				{eventHappens: &csiNodeUpdate},
				// This pod will be requeued to activeQ because it's a pod group member and the queued pod group itself is missing.
				{podEnqueued: newQueuedPodInfoForLookup(pgPod2)},
				// This pod will be requeued to backoffQ immediately because no plugin is registered as unschedulable plugin,
				// which means the pod encountered an unexpected error (e.g., a network error).
				{podEnqueued: newQueuedPodInfoForLookup(pod3)},
			},
			wantActiveQPodNames: []string{"pgpod1", "pgpod2"},
			wantInFlightPods:    nil, // should be empty
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					pvAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					nodeAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					pvcAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					csiNodeUpdate: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name:                   "pod group is enqueued to backoff if no failed plugin",
			genericWorkloadEnabled: []bool{true},
			initialPods:            []*v1.Pod{pgPod1, pgPod2},
			actions: []action{
				{podPopped: pgPod1}, // Pops group, so pgPod1 and pgPod2 are inFlight
				// Requeue them sequentially with no plugins (unexpected errors)
				{podEnqueued: &framework.QueuedPodInfo{
					PodInfo: mustNewPodInfo(pgPod1),
				}},
				{podEnqueued: &framework.QueuedPodInfo{
					PodInfo: mustNewPodInfo(pgPod2),
				}},
				{podGroupAttempted: newQueuedPodGroupInfoForLookup(pgPod1)},
			},
			wantBackoffQPodNames: []string{"pgpod1", "pgpod2"},
		},
		{
			name:                   "pod group is enqueued to backoff even if there were no events that can make the pod group schedulable",
			genericWorkloadEnabled: []bool{true},
			initialPods:            []*v1.Pod{pgPod1, pgPod2},
			actions: []action{
				{podPopped: pgPod1}, // Pops group, so pgPod1 and pgPod2 are inFlight
				// Requeue them sequentially with failed plugins
				{podEnqueued: &framework.QueuedPodInfo{
					PodInfo: mustNewPodInfo(pgPod1),
					QueueingParams: framework.QueueingParams{
						UnschedulablePlugins: sets.New("fooPlugin1"),
					},
				}},
				{podEnqueued: &framework.QueuedPodInfo{
					PodInfo: mustNewPodInfo(pgPod2),
					QueueingParams: framework.QueueingParams{
						UnschedulablePlugins: sets.New("fooPlugin1"),
					},
				}},
				{podGroupAttempted: newQueuedPodGroupInfoForLookup(pgPod1)},
			},
			wantBackoffQPodNames: []string{"pgpod1", "pgpod2"},
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					pvAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		for _, genericWorkloadEnabled := range test.genericWorkloadEnabled {
			t.Run(fmt.Sprintf("%s (genericWorkloadEnabled: %v)", test.name, genericWorkloadEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.GenericWorkload: genericWorkloadEnabled,
				})
				logger, ctx := ktesting.NewTestContext(t)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()
				obj := make([]runtime.Object, 0, len(test.initialPods))
				for _, p := range test.initialPods {
					obj = append(obj, p)
				}
				fakeClock := testingclock.NewFakeClock(time.Now())
				q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), obj, WithQueueingHintMapPerProfile(test.queueingHintMap), WithClock(fakeClock))
				sortOpt := cmpopts.SortSlices(func(a, b string) bool { return a < b })
				if genericWorkloadEnabled {
					podGroup := st.MakePodGroup().Name(pgName).Namespace(pgPod1.Namespace).Obj()
					q.AddPodGroup(logger, podGroup)
				}

				// When a Pod is added to the queue, the QueuedPodInfo will have a new timestamp.
				// On Windows, time.Now() is not as precise, 2 consecutive calls may return the same timestamp.
				// Thus, all the QueuedPodInfos can have the same timestamps, which can be an issue
				// when we're expecting them to be popped in a certain order (the Less function
				// sorts them by Timestamps if they have the same Pod Priority).
				// Using a fake clock for the queue and incrementing it after each added Pod will
				// solve this issue on Windows unit test runs.
				// For more details on the Windows clock resolution issue, see: https://github.com/golang/go/issues/8687
				for _, p := range test.initialPods {
					q.Add(ctx, p)
					fakeClock.Step(time.Second)
				}

				for _, action := range test.actions {
					switch {
					case action.podCreated != nil:
						q.Add(ctx, action.podCreated)
					case action.podPopped != nil:
						popPod(t, logger, q, action.podPopped)
					case action.eventHappens != nil:
						q.MoveAllToActiveOrBackoffQueue(logger, *action.eventHappens, nil, nil, nil)
					case action.podEnqueued != nil:
						err := q.AddUnschedulablePodIfNotPresent(logger, action.podEnqueued, q.SchedulingCycle())
						if err != nil {
							t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
						}
					case action.podGroupAttempted != nil:
						err := q.AddAttemptedPodGroupIfNeeded(logger, action.podGroupAttempted, q.SchedulingCycle(), fwk.NewStatus(fwk.Error))
						if err != nil {
							t.Fatalf("unexpected error from AddAttemptedPodGroupIfNeeded: %v", err)
						}
					case action.callback != nil:
						action.callback(t, q)
					}
				}

				actualInFlightPods := make(map[types.UID]*v1.Pod)
				for _, pod := range q.activeQ.listInFlightPods() {
					actualInFlightPods[pod.UID] = pod
				}
				wantInFlightPods := make(map[types.UID]*v1.Pod)
				for _, pod := range test.wantInFlightPods {
					wantInFlightPods[pod.UID] = pod
				}
				if diff := cmp.Diff(wantInFlightPods, actualInFlightPods); diff != "" {
					t.Errorf("Unexpected diff in inFlightPods (-want, +got):\n%s", diff)
				}

				var wantInFlightEvents []interface{}
				for _, value := range test.wantInFlightEvents {
					if event, ok := value.(fwk.ClusterEvent); ok {
						value = &clusterEvent{event: event}
					}
					wantInFlightEvents = append(wantInFlightEvents, value)
				}
				if diff := cmp.Diff(wantInFlightEvents, q.activeQ.listInFlightEvents(), cmp.AllowUnexported(clusterEvent{}), cmpopts.EquateComparable(fwk.ClusterEvent{})); diff != "" {
					t.Errorf("Unexpected diff in inFlightEvents (-want, +got):\n%s", diff)
				}

				if test.wantActiveQPodNames != nil {
					pods := q.activeQ.list()
					var podNames []string
					for _, pod := range pods {
						podNames = append(podNames, pod.Name)
					}
					if diff := cmp.Diff(test.wantActiveQPodNames, podNames, sortOpt); diff != "" {
						t.Fatalf("Unexpected diff of activeQ pod names (-want, +got):\n%s", diff)
					}

					wantPodNames := sets.New(test.wantActiveQPodNames...)
					for _, pod := range pods {
						if !wantPodNames.Has(pod.Name) {
							t.Fatalf("Pod %v was not expected to be in the activeQ.", pod.Name)
						}
					}
				}

				if test.wantBackoffQPodNames != nil {
					pods := q.backoffQ.list()
					var podNames []string
					for _, pod := range pods {
						podNames = append(podNames, pod.Name)
					}
					if diff := cmp.Diff(test.wantBackoffQPodNames, podNames, sortOpt); diff != "" {
						t.Fatalf("Unexpected diff of backoffQ pod names (-want, +got):\n%s", diff)
					}

					wantPodNames := sets.New(test.wantBackoffQPodNames...)
					for _, podGotFromBackoffQ := range pods {
						if !wantPodNames.Has(podGotFromBackoffQ.Name) {
							t.Fatalf("Pod %v was not expected to be in the backoffQ.", podGotFromBackoffQ.Name)
						}
					}
				}

				for _, podName := range test.wantUnschedPodPoolPodNames {
					p := getUnschedulablePod(q, &st.MakePod().Name(podName).Pod)
					if p == nil {
						t.Fatalf("Pod %v was not found in the unschedulableEntities.", podName)
					}
				}
			})
		}
	}
}

func popPod(t *testing.T, logger klog.Logger, q *PriorityQueue, pod *v1.Pod) *framework.QueuedPodInfo {
	entity, err := q.Pop(logger)
	if err != nil {
		t.Fatalf("Pop failed: %v", err)
	}
	var pInfo *framework.QueuedPodInfo
	switch specificEntity := entity.(type) {
	case *framework.QueuedPodInfo:
		pInfo = specificEntity
	case *framework.QueuedPodGroupInfo:
		for _, pi := range specificEntity.QueuedPodInfos {
			if pi.Pod.UID == pod.UID {
				pInfo = pi
				break
			}
		}
	default:
		t.Fatalf("unexpected popped entity type: %T", entity)
	}
	if pInfo == nil {
		t.Fatalf("Pod %s was not found in the popped entity", pod.UID)
	}
	if pInfo.Pod.UID != pod.UID {
		t.Errorf("Unexpected popped pod: expected %s, got %s", pod.UID, pInfo.Pod.UID)
	}
	return pInfo
}

func TestPop(t *testing.T) {
	pod := st.MakePod().Name("targetpod").UID("pod1").Obj()
	queueingHintMap := QueueingHintMapPerProfile{
		"": {
			pvAdd: {
				{
					// The hint fn tells that this event makes a Pod scheudlable.
					PluginName:     "fooPlugin1",
					QueueingHintFn: queueHintReturnQueue,
				},
			},
		},
	}
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod}, WithQueueingHintMapPerProfile(queueingHintMap))
	q.Add(ctx, pod)

	// Simulate failed attempt that makes the pod unschedulable.
	poppedPod := popPod(t, logger, q, pod)
	// We put register the plugin to PendingPlugins so that it's interpreted as queueImmediately and skip backoff.
	poppedPod.PendingPlugins = sets.New("fooPlugin1")
	if err := q.AddUnschedulablePodIfNotPresent(logger, poppedPod, q.SchedulingCycle()); err != nil {
		t.Errorf("Unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}

	// Activate it again.
	q.MoveAllToActiveOrBackoffQueue(logger, pvAdd, nil, nil, nil)

	// Now check result of Pop.
	poppedPod = popPod(t, logger, q, pod)
	// PendingPlugins are preserved after Pop() so they can be logged if scheduling
	// succeeds, or cleared in handleSchedulingFailure() if it fails.
	if !poppedPod.PendingPlugins.Equal(sets.New("fooPlugin1")) {
		t.Errorf("QueuedPodInfo from Pop should preserve PendingPlugins, expected fooPlugin1, got instead: %+v", poppedPod)
	}
}

func TestPriorityQueue_AddUnschedulablePodIfNotPresent(t *testing.T) {
	objs := []runtime.Object{highPriNominatedPodInfo.Pod, unschedulablePodInfo.Pod}
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), objs)
	// insert unschedulablePodInfo and pop right after that
	// because the scheduling queue records unschedulablePod as in-flight Pod.
	q.Add(ctx, unschedulablePodInfo.Pod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(unschedulablePodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}

	q.Add(ctx, highPriNominatedPodInfo.Pod)
	err := q.AddUnschedulablePodIfNotPresent(logger, newQueuedPodInfoForLookup(unschedulablePodInfo.Pod, "plugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(highPriNominatedPodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	if len(q.nominator.nominatedPods) != 1 {
		t.Errorf("Expected nominatedPods to have one element: %v", q.nominator)
	}
	// unschedulablePodInfo is inserted to unschedulable pod pool because no events happened during scheduling.
	if diff := cmp.Diff(unschedulablePodInfo.Pod, getUnschedulablePod(q, unschedulablePodInfo.Pod)); diff != "" {
		t.Errorf("Unexpected pod in unschedulableEntities (-want, +got):\n%s", diff)
	}
}

// TestPriorityQueue_AddUnschedulablePodIfNotPresent_Backoff tests the scenarios when
// AddUnschedulablePodIfNotPresent is called asynchronously.
// Pods in and before current scheduling cycle will be put back to activeQueue
// if we were trying to schedule them when we received move request.
func TestPriorityQueue_AddUnschedulablePodIfNotPresent_Backoff(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(testingclock.NewFakeClock(time.Now())))
	totalNum := 10
	expectedPods := make([]v1.Pod, 0, totalNum)
	for i := 0; i < totalNum; i++ {
		priority := int32(i)
		p := st.MakePod().Name(fmt.Sprintf("pod%d", i)).Namespace(fmt.Sprintf("ns%d", i)).UID(fmt.Sprintf("upns%d", i)).Priority(priority).Obj()
		expectedPods = append(expectedPods, *p)
		// priority is to make pods ordered in the PriorityQueue
		q.Add(ctx, p)
	}

	// Pop all pods except for the first one
	for i := totalNum - 1; i > 0; i-- {
		p, _ := q.Pop(logger)
		if diff := cmp.Diff(&expectedPods[i], p.(*framework.QueuedPodInfo).Pod); diff != "" {
			t.Errorf("Unexpected pod (-want, +got):\n%s", diff)
		}
	}

	// move all pods to active queue when we were trying to schedule them
	q.MoveAllToActiveOrBackoffQueue(logger, framework.EventUnschedulableTimeout, nil, nil, nil)
	oldCycle := q.SchedulingCycle()

	item, _ := q.Pop(logger)
	firstPod := item.(*framework.QueuedPodInfo)
	if diff := cmp.Diff(&expectedPods[0], firstPod.Pod); diff != "" {
		t.Errorf("Unexpected pod (-want, +got):\n%s", diff)
	}

	// mark pods[1] ~ pods[totalNum-1] as unschedulable and add them back
	for i := 1; i < totalNum; i++ {
		unschedulablePod := expectedPods[i].DeepCopy()
		unschedulablePod.Status = v1.PodStatus{
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodScheduled,
					Status: v1.ConditionFalse,
					Reason: v1.PodReasonUnschedulable,
				},
			},
		}

		err := q.AddUnschedulablePodIfNotPresent(logger, newQueuedPodInfoForLookup(unschedulablePod, "plugin"), oldCycle)
		if err != nil {
			t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
		}
	}

	// Since there was a move request at the same cycle as "oldCycle", these pods
	// should be in the backoff queue.
	for i := 1; i < totalNum; i++ {
		if !q.backoffQ.has(newQueuedPodInfoForLookup(&expectedPods[i])) {
			t.Errorf("Expected %v to be added to backoffQ.", expectedPods[i].Name)
		}
	}
}

// tryPop tries to pop one entity from the queue and returns it.
// It waits 5 seconds before timing out, assuming the queue is then empty.
func tryPop(t *testing.T, logger klog.Logger, q *PriorityQueue) framework.QueuedEntityInfo {
	t.Helper()

	var gotEntity framework.QueuedEntityInfo
	popped := make(chan struct{}, 1)
	go func() {
		entity, err := q.Pop(logger)
		if err != nil {
			t.Errorf("Failed to pop entity from scheduling queue: %s", err)
		}
		if entity != nil {
			gotEntity = entity
		}
		popped <- struct{}{}
	}()

	timer := time.NewTimer(5 * time.Second)
	select {
	case <-timer.C:
		q.Close()
	case <-popped:
		timer.Stop()
	}
	return gotEntity
}

func TestPriorityQueue_Pop(t *testing.T) {
	highPriorityPodInfo2 := mustNewPodInfo(
		st.MakePod().Name("hpp2").Namespace("ns1").UID("hpp2ns1").Priority(highPriority).Obj(),
	)
	tests := []struct {
		name                   string
		popFromBackoffQEnabled bool
		genericWorkloadEnabled bool
		usePodGroups           bool
		wantPods               []string
	}{
		{
			name:                   "individual pod, GenericWorkload enabled, PopFromBackoffQ enabled, pops from both activeQ and backoffQ",
			popFromBackoffQEnabled: true,
			genericWorkloadEnabled: true,
			usePodGroups:           false,
			wantPods:               []string{medPriorityPodInfo.Pod.Name, highPriorityPodInfo.Pod.Name},
		},
		{
			name:                   "individual pod, GenericWorkload enabled, PopFromBackoffQ disabled, pops only from activeQ",
			popFromBackoffQEnabled: false,
			genericWorkloadEnabled: true,
			usePodGroups:           false,
			wantPods:               []string{medPriorityPodInfo.Pod.Name},
		},
		{
			name:                   "individual pod, GenericWorkload disabled, PopFromBackoffQ enabled, pops from both activeQ and backoffQ",
			popFromBackoffQEnabled: true,
			genericWorkloadEnabled: false,
			usePodGroups:           false,
			wantPods:               []string{medPriorityPodInfo.Pod.Name, highPriorityPodInfo.Pod.Name},
		},
		{
			name:                   "individual pod, GenericWorkload disabled, PopFromBackoffQ disabled, pops only from activeQ",
			popFromBackoffQEnabled: false,
			genericWorkloadEnabled: false,
			usePodGroups:           false,
			wantPods:               []string{medPriorityPodInfo.Pod.Name},
		},
		{
			name:                   "pod group, PopFromBackoffQ enabled, pops from both activeQ and backoffQ",
			popFromBackoffQEnabled: true,
			genericWorkloadEnabled: true,
			usePodGroups:           true,
			wantPods:               []string{medPriorityPodInfo.Pod.Name, highPriorityPodInfo.Pod.Name},
		},
		{
			name:                   "pod group, PopFromBackoffQ disabled, pops only from activeQ",
			popFromBackoffQEnabled: false,
			genericWorkloadEnabled: true,
			usePodGroups:           true,
			wantPods:               []string{medPriorityPodInfo.Pod.Name},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:          tt.genericWorkloadEnabled,
				features.SchedulerPopFromBackoffQ: tt.popFromBackoffQEnabled,
			})

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			medPod, backoffPod, errorBackoffPod, unschedPod := medPriorityPodInfo.Pod, highPriorityPodInfo.Pod, highPriorityPodInfo2.Pod, unschedulablePodInfo.Pod
			if tt.usePodGroups {
				medPod = withPodGroupName(medPriorityPodInfo.Pod, "pg-med")
				backoffPod = withPodGroupName(highPriorityPodInfo.Pod, "pg-backoff")
				errorBackoffPod = withPodGroupName(highPriorityPodInfo2.Pod, "pg-errbackoff")
				unschedPod = withPodGroupName(unschedulablePodInfo.Pod, "pg-unsched")
			}

			objs := []runtime.Object{medPod, backoffPod, errorBackoffPod, unschedPod}
			q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), objs)

			var medEntity, backoffEntity, errorBackoffEntity, unschedEntity framework.QueuedEntityInfo
			if tt.usePodGroups {
				medEntity = q.newQueuedPodGroupInfo(q.newQueuedPodInfo(ctx, medPod), nil)
				backoffPodGroup := q.newQueuedPodGroupInfo(q.newQueuedPodInfo(ctx, backoffPod, "plugin"), nil)
				backoffPodGroup.UnschedulablePlugins = sets.New("plugin")
				backoffEntity = backoffPodGroup
				errorBackoffEntity = q.newQueuedPodGroupInfo(q.newQueuedPodInfo(ctx, errorBackoffPod), nil)
				unschedPodGroup := q.newQueuedPodGroupInfo(q.newQueuedPodInfo(ctx, unschedPod, "plugin"), nil)
				unschedPodGroup.UnschedulablePlugins = sets.New("plugin")
				unschedEntity = unschedPodGroup
			} else {
				medEntity = q.newQueuedPodInfo(ctx, medPod)
				backoffEntity = q.newQueuedPodInfo(ctx, backoffPod, "plugin")
				errorBackoffEntity = q.newQueuedPodInfo(ctx, errorBackoffPod)
				unschedEntity = q.newQueuedPodInfo(ctx, unschedPod, "plugin")
			}

			// Add medium priority entity to the activeQ
			q.activeQ.add(logger, medEntity, framework.EventUnscheduledPodAdd.Label())
			// Add high priority entity to the backoffQ
			q.backoffQ.add(logger, backoffEntity, framework.EventUnscheduledPodAdd.Label())
			// Add high priority entity to the errorBackoffQ
			q.backoffQ.add(logger, errorBackoffEntity, framework.EventUnscheduledPodAdd.Label())
			// Add entity to the unschedulableEntities
			q.unschedulableEntities.addOrUpdate(unschedEntity, false, framework.EventUnscheduledPodAdd.Label())

			var gotPods []string
			for i := 0; i < len(tt.wantPods)+1; i++ {
				gotEntity := tryPop(t, logger, q)
				if gotEntity == nil {
					break
				}
				if _, isPodGroup := gotEntity.(*framework.QueuedPodGroupInfo); isPodGroup != tt.usePodGroups {
					t.Errorf("Expected queued pod group: %v, got: %v", tt.usePodGroups, isPodGroup)
				}
				gotEntity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
					gotPods = append(gotPods, pInfo.Pod.Name)
					return true
				})
			}

			if diff := cmp.Diff(tt.wantPods, gotPods); diff != "" {
				t.Errorf("Unexpected popped pods (-want, +got): %s", diff)
			}
		})
	}
}

func TestPriorityQueue_Update(t *testing.T) {
	c := testingclock.NewFakeClock(time.Now())

	queuePlugin := "queuePlugin"
	skipPlugin := "skipPlugin"
	queueingHintMap := QueueingHintMapPerProfile{
		"": {
			framework.EventTargetPodUpdate: {
				{
					PluginName:     queuePlugin,
					QueueingHintFn: queueHintReturnQueue,
				},
				{
					PluginName:     skipPlugin,
					QueueingHintFn: queueHintReturnSkip,
				},
			},
		},
	}

	withGate := func(p *v1.Pod) *v1.Pod {
		newPod := p.DeepCopy()
		newPod.Labels = map[string]string{"deny": "true"}
		return newPod
	}

	notInAnyQueue := "NotInAnyQueue"
	tests := []struct {
		name  string
		wantQ string
		// wantAddedToNominated is whether a Pod from the test case should be registered as a nominated Pod in the nominator.
		wantAddedToNominated bool
		// prepareFunc is the function called to prepare pods in the queue before the test case calls Update().
		// This function returns three values;
		// - oldPod/newPod: each test will call Update() with these oldPod and newPod.
		prepareFunc func(tCtx ktesting.TContext, q *PriorityQueue) (oldPod, newPod *v1.Pod)
	}{
		{
			name:  "Update pod that didn't exist in the queue",
			wantQ: activeQ,
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) (oldPod, newPod *v1.Pod) {
				updatedPod := medPriorityPodInfo.Pod.DeepCopy()
				updatedPod.Annotations["foo"] = "test"
				return medPriorityPodInfo.Pod, updatedPod
			},
		},
		{
			name:  "Update gated pod that didn't exist in the queue",
			wantQ: unschedulableQ,
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) (oldPod, newPod *v1.Pod) {
				updatedPod := withGate(medPriorityPodInfo.Pod)
				updatedPod.Annotations["foo"] = "test"
				return withGate(medPriorityPodInfo.Pod), updatedPod
			},
		},
		{
			name:                 "Update non-existent highPriorityPodInfo and add a nominatedNodeName to it",
			wantQ:                activeQ,
			wantAddedToNominated: true,
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) (oldPod, newPod *v1.Pod) {
				return highPriorityPodInfo.Pod, highPriNominatedPodInfo.Pod
			},
		},
		{
			name:                 "Update non-existent gated highPriorityPodInfo and add a nominatedNodeName to it",
			wantQ:                unschedulableQ,
			wantAddedToNominated: true,
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) (oldPod, newPod *v1.Pod) {
				return withGate(highPriorityPodInfo.Pod), withGate(highPriNominatedPodInfo.Pod)
			},
		},
		{
			name:  "When updating a pod that is already in activeQ, the pod should remain in activeQ after Update()",
			wantQ: activeQ,
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) (oldPod, newPod *v1.Pod) {
				q.Add(tCtx, highPriorityPodInfo.Pod)
				return highPriorityPodInfo.Pod, highPriorityPodInfo.Pod
			},
		},
		{
			name:  "When updating a pod that is in backoff queue and is still backing off, it will be updated in backoff queue",
			wantQ: backoffQ,
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) (oldPod, newPod *v1.Pod) {
				podInfo := q.newQueuedPodInfo(tCtx, medPriorityPodInfo.Pod)
				q.backoffQ.add(klog.FromContext(tCtx), podInfo, framework.EventUnscheduledPodAdd.Label())
				return podInfo.Pod, podInfo.Pod
			},
		},
		{
			name:  "when updating a pod in unschedulableEntities, if its backoff timer has not yet expired, it moves to backoffQ",
			wantQ: backoffQ,
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) (oldPod, newPod *v1.Pod) {
				pInfo := q.newQueuedPodInfo(tCtx, medPriorityPodInfo.Pod, queuePlugin)
				// needs to increment to make the pod backing off
				pInfo.UnschedulableCount++
				q.unschedulableEntities.addOrUpdate(pInfo, false, framework.EventUnscheduledPodAdd.Label())
				updatedPod := medPriorityPodInfo.Pod.DeepCopy()
				updatedPod.Annotations["foo"] = "test"
				return medPriorityPodInfo.Pod, updatedPod
			},
		},
		{
			name:  "when updating a pod in unschedulableEntities, if its backoff timer has expired, it moves to activeQ",
			wantQ: activeQ,
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) (oldPod, newPod *v1.Pod) {
				pInfo := q.newQueuedPodInfo(tCtx, medPriorityPodInfo.Pod, queuePlugin)
				// needs to increment to make the pod backing off
				pInfo.UnschedulableCount++
				q.unschedulableEntities.addOrUpdate(pInfo, false, framework.EventUnscheduledPodAdd.Label())
				updatedPod := medPriorityPodInfo.Pod.DeepCopy()
				updatedPod.Annotations["foo"] = "test1"
				// Move clock by podMaxBackoffDuration, so that pods in the unschedulableEntities would pass the backing off,
				// and the pods will be moved into activeQ.
				c.Step(q.backoffQ.podMaxBackoffDuration())
				return medPriorityPodInfo.Pod, updatedPod
			},
		},
		{
			name:  "when updating a pod in unschedulableEntities, if the scheduling hint returns QueueSkip, it remains in unschedulableEntities",
			wantQ: unschedulableQ,
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) (oldPod, newPod *v1.Pod) {
				q.unschedulableEntities.addOrUpdate(q.newQueuedPodInfo(tCtx, medPriorityPodInfo.Pod, skipPlugin), false, framework.EventUnscheduledPodAdd.Label())
				updatedPod := medPriorityPodInfo.Pod.DeepCopy()
				updatedPod.Annotations["foo"] = "test1"
				return medPriorityPodInfo.Pod, updatedPod
			},
		},
		{
			name:  "when updating a pod which is in flightPods, the pod will not be added to any queue",
			wantQ: notInAnyQueue,
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) (oldPod, newPod *v1.Pod) {
				// We need to once add this Pod to activeQ and Pop() it so that this Pod is registered correctly in inFlightPods.
				q.Add(tCtx, medPriorityPodInfo.Pod)
				if p, err := q.Pop(klog.FromContext(tCtx)); err != nil {
					t.Errorf("Pop failed: %v", err)
				} else if diff := cmp.Diff(medPriorityPodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
					t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
				}
				updatedPod := medPriorityPodInfo.Pod.DeepCopy()
				updatedPod.Annotations["foo"] = "bar"
				return medPriorityPodInfo.Pod, updatedPod
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			objs := []runtime.Object{highPriorityPodInfo.Pod, unschedulablePodInfo.Pod, medPriorityPodInfo.Pod}
			plugin := &denyingPreEnqueuePlugin{denylists: []string{"deny"}}
			m := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"denyingPreEnqueuePlugin": plugin,
				},
			}
			q := NewTestQueueWithObjects(tCtx, newDefaultQueueSort(), objs, WithClock(c), WithQueueingHintMapPerProfile(queueingHintMap), WithPreEnqueuePluginMap(m))

			oldPod, newPod := tt.prepareFunc(tCtx, q)

			q.Update(tCtx, oldPod, newPod)

			var pInfo *framework.QueuedPodInfo

			// validate expected queue
			if pInfoFromBackoff, exists := q.backoffQ.get(newQueuedPodInfoForLookup(newPod)); exists {
				if tt.wantQ != backoffQ {
					t.Errorf("expected pod %s not to be queued to backoffQ, but it was", newPod.Name)
				}
				pInfo = pInfoFromBackoff.(*framework.QueuedPodInfo)
			}

			if pInfoFromActive, exists := q.activeQ.get(newQueuedPodInfoForLookup(newPod)); exists {
				if tt.wantQ != activeQ {
					t.Errorf("expected pod %s not to be queued to activeQ, but it was", newPod.Name)
				}
				pInfo = pInfoFromActive.(*framework.QueuedPodInfo)
			}

			if pInfoFromUnsched := q.unschedulableEntities.get(newQueuedPodInfoForLookup(newPod)); pInfoFromUnsched != nil {
				if tt.wantQ != unschedulableQ {
					t.Errorf("expected pod %s to not be queued to unschedulableEntities, but it was", newPod.Name)
				}
				pInfo = pInfoFromUnsched.(*framework.QueuedPodInfo)
			}

			if tt.wantQ == notInAnyQueue {
				// skip the rest of the test if pod is not expected to be in any of the queues.
				return
			}

			if diff := cmp.Diff(newPod, pInfo.PodInfo.Pod); diff != "" {
				t.Errorf("Unexpected updated pod diff (-want, +got): %s", diff)
			}

			if tt.wantAddedToNominated && len(q.nominator.nominatedPods) != 1 {
				t.Errorf("Expected one item in nominatedPods map: %v", q.nominator.nominatedPods)
			}

		})
	}
}

// TestPriorityQueue_UpdateWhenInflight ensures to requeue a Pod back to activeQ/backoffQ
// if it actually got an update that may make it schedulable while being scheduled.
// See https://github.com/kubernetes/kubernetes/pull/125578#discussion_r1648338033 for more context.
func TestPriorityQueue_UpdateWhenInflight(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	m := makeEmptyQueueingHintMapPerProfile()
	// fakePlugin could change its scheduling result by any updates in Pods.
	m[""][framework.EventTargetPodUpdate] = []*QueueingHintFunction{
		{
			PluginName:     "fakePlugin",
			QueueingHintFn: queueHintReturnQueue,
		},
	}
	c := testingclock.NewFakeClock(time.Now())
	q := NewTestQueue(ctx, newDefaultQueueSort(), WithQueueingHintMapPerProfile(m), WithClock(c))

	// test-pod is created and popped out from the queue
	testPod := st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").Obj()
	q.Add(ctx, testPod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(testPod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}

	// testPod is updated while being scheduled.
	updatedPod := testPod.DeepCopy()
	updatedPod.Spec.Tolerations = []v1.Toleration{
		{
			Key:    "foo",
			Effect: v1.TaintEffectNoSchedule,
		},
	}

	q.Update(ctx, testPod, updatedPod)
	// test-pod got rejected by fakePlugin,
	// but the update event that it just got may change this scheduling result,
	// and hence we should put this pod to activeQ/backoffQ.
	err := q.AddUnschedulablePodIfNotPresent(logger, newQueuedPodInfoForLookup(updatedPod, "fakePlugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}

	item, exists := q.backoffQ.get(newQueuedPodInfoForLookup(updatedPod))
	if !exists {
		t.Fatalf("expected pod %s to be queued to backoffQ, but it wasn't.", updatedPod.Name)
	}
	pInfo := item.(*framework.QueuedPodInfo)
	if diff := cmp.Diff(updatedPod, pInfo.PodInfo.Pod); diff != "" {
		t.Errorf("Unexpected updated pod diff (-want, +got): %s", diff)
	}
}

func TestPriorityQueue_Delete(t *testing.T) {
	metrics.Register()
	timestamp := time.Now()

	pod1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Obj()
	pInfo1 := &framework.QueuedPodInfo{
		PodInfo: mustNewTestPodInfo(t, pod1),
		QueueingParams: framework.QueueingParams{
			Timestamp:            timestamp,
			UnschedulablePlugins: sets.New("fakePlugin"),
			PendingPlugins:       sets.New("fakePendingPlugin"),
		},
	}
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Obj()
	pInfo2 := &framework.QueuedPodInfo{
		PodInfo: mustNewTestPodInfo(t, pod2),
		QueueingParams: framework.QueueingParams{
			Timestamp:            timestamp,
			UnschedulablePlugins: sets.New("fakePlugin2"),
		},
	}

	tests := []struct {
		name                string
		operations          []operation
		operands            []*framework.QueuedPodInfo
		podToDelete         *v1.Pod
		expectedAbsentPods  []*v1.Pod
		expectedPresentPods []*v1.Pod
		expectedMetrics     map[string]int
	}{
		{
			name: "Delete pod from activeQ",
			operations: []operation{
				add,
				add,
			},
			operands:            []*framework.QueuedPodInfo{pInfo1, pInfo2},
			podToDelete:         pod1,
			expectedAbsentPods:  []*v1.Pod{pod1},
			expectedPresentPods: []*v1.Pod{pod2},
			expectedMetrics: map[string]int{
				"fakePlugin":        0,
				"fakePendingPlugin": 0,
				"fakePlugin2":       0,
			},
		},
		{
			name: "Delete pod from backoffQ",
			operations: []operation{
				popAndRequeueAsBackoff,
				add,
			},
			operands:            []*framework.QueuedPodInfo{pInfo1, pInfo2},
			podToDelete:         pod1,
			expectedAbsentPods:  []*v1.Pod{pod1},
			expectedPresentPods: []*v1.Pod{pod2},
			expectedMetrics: map[string]int{
				"fakePlugin":        0,
				"fakePendingPlugin": 0,
				"fakePlugin2":       0,
			},
		},
		{
			name: "Delete pod from unschedulablePods",
			operations: []operation{
				popAndRequeueAsUnschedulable,
				popAndRequeueAsUnschedulable,
			},
			operands:            []*framework.QueuedPodInfo{pInfo1, pInfo2},
			podToDelete:         pod1,
			expectedAbsentPods:  []*v1.Pod{pod1},
			expectedPresentPods: []*v1.Pod{pod2},
			expectedMetrics: map[string]int{
				"fakePlugin":        0,
				"fakePendingPlugin": 0,
				"fakePlugin2":       1,
			},
		},
		{
			name: "Delete nominated pod from activeQ",
			operations: []operation{
				add,
			},
			operands:           []*framework.QueuedPodInfo{{PodInfo: highPriNominatedPodInfo}},
			podToDelete:        highPriNominatedPodInfo.Pod,
			expectedAbsentPods: []*v1.Pod{highPriNominatedPodInfo.Pod},
			expectedMetrics: map[string]int{
				"fakePlugin":        0,
				"fakePendingPlugin": 0,
				"fakePlugin2":       0,
			},
		},
		{
			name: "Delete non-existing pod",
			operations: []operation{
				popAndRequeueAsUnschedulable,
			},
			operands:            []*framework.QueuedPodInfo{pInfo1},
			podToDelete:         pod2,
			expectedAbsentPods:  []*v1.Pod{pod2},
			expectedPresentPods: []*v1.Pod{pod1},
			expectedMetrics: map[string]int{
				"fakePlugin":        1,
				"fakePendingPlugin": 1,
				"fakePlugin2":       0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			q := NewTestQueue(tCtx, newDefaultQueueSort(), WithClock(testingclock.NewFakeClock(timestamp)))

			// Reset metrics for plugins used in the test
			allPlugins := sets.New("fakePlugin", "fakePendingPlugin", "fakePlugin2")
			for plugin := range allPlugins {
				metrics.UnschedulableReason(plugin, pod1.Spec.SchedulerName).Set(0)
			}

			// Execute operations
			for i, op := range tt.operations {
				op(tCtx, q, tt.operands[i])
			}

			q.Delete(tCtx.Logger(), tt.podToDelete)

			// Verification
			for _, pod := range tt.expectedAbsentPods {
				pInfoLookup := newQueuedPodInfoForLookup(pod)
				if q.activeQ.has(pInfoLookup) || q.backoffQ.has(pInfoLookup) || q.unschedulableEntities.get(pInfoLookup) != nil {
					t.Errorf("Expected pod %v to be absent, but it is present", pod.Name)
				}
			}

			for _, pod := range tt.expectedPresentPods {
				pInfoLookup := newQueuedPodInfoForLookup(pod)
				if !q.activeQ.has(pInfoLookup) && !q.backoffQ.has(pInfoLookup) && q.unschedulableEntities.get(pInfoLookup) == nil {
					t.Errorf("Expected pod %v to be present, but it is absent", pod.Name)
				}
			}

			for plugin, expectedVal := range tt.expectedMetrics {
				val, _ := testutil.GetGaugeMetricValue(metrics.UnschedulableReason(plugin, tt.podToDelete.Spec.SchedulerName))
				if diff := cmp.Diff(float64(expectedVal), val); diff != "" {
					t.Errorf("Unexpected metric value for plugin %v after delete (-want, +got):\n%s", plugin, diff)
				}
			}

			if len(q.nominator.nominatedPods) != 0 || len(q.nominator.nominatedPodToNode) != 0 {
				t.Errorf("Expected nominatedPods and nominatedPodToNode to be empty, but got %v and %v", q.nominator.nominatedPods, q.nominator.nominatedPodToNode)
			}
		})
	}
}

func TestPriorityQueue_Activate(t *testing.T) {
	metrics.Register()
	tests := []struct {
		name                            string
		qPodInfoInUnschedulableEntities []*framework.QueuedPodInfo
		qPodInfoInBackoffQ              []*framework.QueuedPodInfo
		qPodInActiveQ                   []*v1.Pod
		qPodInfoToActivate              *framework.QueuedPodInfo
		qPodInInFlightPod               *v1.Pod
		expectedInFlightEvent           *clusterEvent
		want                            []*framework.QueuedPodInfo
	}{
		{
			name:               "pod already in activeQ",
			qPodInActiveQ:      []*v1.Pod{highPriNominatedPodInfo.Pod},
			qPodInfoToActivate: &framework.QueuedPodInfo{PodInfo: highPriNominatedPodInfo},
			want:               []*framework.QueuedPodInfo{{PodInfo: highPriNominatedPodInfo}}, // 1 already active
		},
		{
			name:               "pod not in unschedulableEntities/backoffQ",
			qPodInfoToActivate: &framework.QueuedPodInfo{PodInfo: highPriNominatedPodInfo},
			want:               []*framework.QueuedPodInfo{},
		},
		{
			name:                  "pod not in unschedulableEntities/backoffQ but in-flight",
			qPodInfoToActivate:    &framework.QueuedPodInfo{PodInfo: highPriNominatedPodInfo},
			qPodInInFlightPod:     highPriNominatedPodInfo.Pod,
			expectedInFlightEvent: &clusterEvent{oldObj: (*v1.Pod)(nil), newObj: highPriNominatedPodInfo.Pod, event: framework.EventForceActivate},
			want:                  []*framework.QueuedPodInfo{},
		},
		{
			name:               "pod not in unschedulableEntities/backoffQ and not in-flight",
			qPodInfoToActivate: &framework.QueuedPodInfo{PodInfo: highPriNominatedPodInfo},
			qPodInInFlightPod:  medPriorityPodInfo.Pod, // different pod is in-flight
			want:               []*framework.QueuedPodInfo{},
		},
		{
			name:                            "pod in unschedulableEntities",
			qPodInfoInUnschedulableEntities: []*framework.QueuedPodInfo{{PodInfo: highPriNominatedPodInfo}},
			qPodInfoToActivate:              &framework.QueuedPodInfo{PodInfo: highPriNominatedPodInfo},
			want:                            []*framework.QueuedPodInfo{{PodInfo: highPriNominatedPodInfo}},
		},
		{
			name:               "pod in backoffQ",
			qPodInfoInBackoffQ: []*framework.QueuedPodInfo{{PodInfo: highPriNominatedPodInfo}},
			qPodInfoToActivate: &framework.QueuedPodInfo{PodInfo: highPriNominatedPodInfo},
			want:               []*framework.QueuedPodInfo{{PodInfo: highPriNominatedPodInfo}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			var objs []runtime.Object
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), objs)

			if tt.qPodInInFlightPod != nil {
				// Put -> Pop the Pod to make it registered in inFlightPods.
				q.activeQ.add(logger, newQueuedPodInfoForLookup(tt.qPodInInFlightPod), framework.EventUnscheduledPodAdd.Label())
				p, err := q.activeQ.pop(logger)
				if err != nil {
					t.Fatalf("Pop failed: %v", err)
				}
				if p.(*framework.QueuedPodInfo).Pod.Name != tt.qPodInInFlightPod.Name {
					t.Errorf("Unexpected popped pod: %v", p.(*framework.QueuedPodInfo).Pod.Name)
				}
				if len(q.activeQ.listInFlightEvents()) != 1 {
					t.Fatal("Expected the pod to be recorded in in-flight events, but it doesn't")
				}
			}

			// Prepare activeQ/unschedulableEntities/backoffQ according to the table
			for _, qPod := range tt.qPodInActiveQ {
				q.Add(ctx, qPod)
			}

			for _, qPodInfo := range tt.qPodInfoInUnschedulableEntities {
				q.unschedulableEntities.addOrUpdate(qPodInfo, false, framework.EventUnscheduledPodAdd.Label())
			}

			for _, qPodInfo := range tt.qPodInfoInBackoffQ {
				q.backoffQ.add(logger, qPodInfo, framework.EventUnscheduledPodAdd.Label())
			}

			// Activate specific pod according to the table
			q.Activate(logger, map[string]*v1.Pod{"test_pod": tt.qPodInfoToActivate.PodInfo.Pod})

			// Check the result after activation by the length of activeQ
			if wantLen := len(tt.want); q.activeQ.len() != wantLen {
				t.Fatalf("length compare: want %v, got %v", wantLen, q.activeQ.len())
			}

			if tt.expectedInFlightEvent != nil {
				if len(q.activeQ.listInFlightEvents()) != 2 {
					t.Fatalf("Expected two in-flight event to be recorded, but got %v events", len(q.activeQ.listInFlightEvents()))
				}
				found := false
				for _, e := range q.activeQ.listInFlightEvents() {
					event, ok := e.(*clusterEvent)
					if !ok {
						continue
					}

					if d := cmp.Diff(tt.expectedInFlightEvent, event, cmpopts.EquateComparable(clusterEvent{})); d != "" {
						t.Fatalf("Unexpected in-flight event (-want, +got):\n%s", d)
					}
					found = true
				}

				if !found {
					t.Fatalf("Expected in-flight event to be recorded, but it wasn't.")
				}
			}

			// Check if the specific pod exists in activeQ
			for _, want := range tt.want {
				if !q.activeQ.has(newQueuedPodInfoForLookup(want.PodInfo.Pod)) {
					t.Errorf("podInfo not exist in activeQ: want %v", want.PodInfo.Pod.Name)
				}
			}
		})
	}
}

type preEnqueuePlugin struct {
	allowlists []string
	name       string
}

func (pl *preEnqueuePlugin) Name() string {
	if pl.name != "" {
		return pl.name
	}
	return "preEnqueuePlugin"
}

func (pl *preEnqueuePlugin) PreEnqueue(ctx context.Context, p *v1.Pod) *fwk.Status {
	for _, allowed := range pl.allowlists {
		for label := range p.Labels {
			if label == allowed {
				return nil
			}
		}
	}
	return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "pod label not in allowlists")
}

type denyingPreEnqueuePlugin struct {
	denylists []string
}

func (pl *denyingPreEnqueuePlugin) Name() string {
	return "denyingPreEnqueuePlugin"
}

func (pl *denyingPreEnqueuePlugin) PreEnqueue(ctx context.Context, p *v1.Pod) *fwk.Status {
	for _, denied := range pl.denylists {
		for label := range p.Labels {
			if label == denied {
				return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "pod label in denylists")
			}
		}
	}
	return nil
}

func TestPriorityQueue_moveToActiveQ(t *testing.T) {
	tests := []struct {
		name                   string
		plugins                []fwk.PreEnqueuePlugin
		pod                    *v1.Pod
		event                  string
		movesFromBackoffQ      bool
		popFromBackoffQEnabled []bool
		wantUnschedulablePods  int
		wantSuccess            bool
	}{
		{
			name:                  "no plugins registered",
			pod:                   st.MakePod().Name("p").Label("p", "").Obj(),
			event:                 framework.EventUnscheduledPodAdd.Label(),
			wantUnschedulablePods: 0,
			wantSuccess:           true,
		},
		{
			name:                  "preEnqueue plugin registered, pod name not in allowlists",
			plugins:               []fwk.PreEnqueuePlugin{&preEnqueuePlugin{}, &preEnqueuePlugin{}},
			pod:                   st.MakePod().Name("p").Label("p", "").Obj(),
			event:                 framework.EventUnscheduledPodAdd.Label(),
			wantUnschedulablePods: 1,
			wantSuccess:           false,
		},
		{
			name: "preEnqueue plugin registered, pod failed one preEnqueue plugin",
			plugins: []fwk.PreEnqueuePlugin{
				&preEnqueuePlugin{allowlists: []string{"foo", "bar"}},
				&preEnqueuePlugin{allowlists: []string{"foo"}},
			},
			pod:                   st.MakePod().Name("bar").Label("bar", "").Obj(),
			event:                 framework.EventUnscheduledPodAdd.Label(),
			wantUnschedulablePods: 1,
			wantSuccess:           false,
		},
		{
			name: "preEnqueue plugin registered, preEnqueue rejects the pod, even if it is after backoff",
			plugins: []fwk.PreEnqueuePlugin{
				&preEnqueuePlugin{allowlists: []string{"foo", "bar"}},
				&preEnqueuePlugin{allowlists: []string{"foo"}},
			},
			pod:                    st.MakePod().Name("bar").Label("bar", "").Obj(),
			event:                  framework.BackoffComplete,
			popFromBackoffQEnabled: []bool{false},
			wantUnschedulablePods:  1,
			wantSuccess:            false,
		},
		{
			// With SchedulerPopFromBackoffQ enabled, the queue assumes the pod has already passed PreEnqueue,
			// and it doesn't run PreEnqueue again, always puts the pod to activeQ.
			name: "preEnqueue plugin registered, pod would fail one preEnqueue plugin, but it is moved from backoffQ after completing backoff, so preEnqueue is not executed",
			plugins: []fwk.PreEnqueuePlugin{
				&preEnqueuePlugin{allowlists: []string{"foo", "bar"}},
				&preEnqueuePlugin{allowlists: []string{"foo"}},
			},
			pod:                    st.MakePod().Name("bar").Label("bar", "").Obj(),
			event:                  framework.BackoffComplete,
			movesFromBackoffQ:      true,
			popFromBackoffQEnabled: []bool{true},
			wantUnschedulablePods:  0,
			wantSuccess:            true,
		},
		{
			name: "preEnqueue plugin registered, pod failed one preEnqueue plugin when activated from unschedulableEntities",
			plugins: []fwk.PreEnqueuePlugin{
				&preEnqueuePlugin{allowlists: []string{"foo", "bar"}},
				&preEnqueuePlugin{allowlists: []string{"foo"}},
			},
			pod:                    st.MakePod().Name("bar").Label("bar", "").Obj(),
			event:                  framework.ForceActivate,
			movesFromBackoffQ:      false,
			popFromBackoffQEnabled: []bool{true},
			wantUnschedulablePods:  1,
			wantSuccess:            false,
		},
		{
			name: "preEnqueue plugin registered, pod would fail one preEnqueue plugin, but was activated from backoffQ, so preEnqueue is not executed",
			plugins: []fwk.PreEnqueuePlugin{
				&preEnqueuePlugin{allowlists: []string{"foo", "bar"}},
				&preEnqueuePlugin{allowlists: []string{"foo"}},
			},
			pod:                    st.MakePod().Name("bar").Label("bar", "").Obj(),
			event:                  framework.ForceActivate,
			movesFromBackoffQ:      true,
			popFromBackoffQEnabled: []bool{true},
			wantUnschedulablePods:  0,
			wantSuccess:            true,
		},
		{
			name: "preEnqueue plugin registered, pod passed all preEnqueue plugins",
			plugins: []fwk.PreEnqueuePlugin{
				&preEnqueuePlugin{allowlists: []string{"foo", "bar"}},
				&preEnqueuePlugin{allowlists: []string{"bar"}},
			},
			pod:                   st.MakePod().Name("bar").Label("bar", "").Obj(),
			event:                 framework.EventUnscheduledPodAdd.Label(),
			wantUnschedulablePods: 0,
			wantSuccess:           true,
		},
	}

	for _, tt := range tests {
		if tt.popFromBackoffQEnabled == nil {
			tt.popFromBackoffQEnabled = []bool{true, false}
		}
		for _, popFromBackoffQEnabled := range tt.popFromBackoffQEnabled {
			t.Run(fmt.Sprintf("%s popFromBackoffQEnabled(%v)", tt.name, popFromBackoffQEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerPopFromBackoffQ, popFromBackoffQEnabled)
				logger, ctx := ktesting.NewTestContext(t)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()

				m := map[string]map[string]fwk.PreEnqueuePlugin{"": make(map[string]fwk.PreEnqueuePlugin, len(tt.plugins))}
				for _, plugin := range tt.plugins {
					m[""][plugin.Name()] = plugin
				}
				q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{tt.pod}, WithPreEnqueuePluginMap(m),
					WithPodInitialBackoffDuration(time.Second*30), WithPodMaxBackoffDuration(time.Second*60))
				got := q.moveToActiveQ(logger, q.newQueuedPodInfo(ctx, tt.pod), tt.event, tt.movesFromBackoffQ)
				if got != tt.wantSuccess {
					t.Errorf("Unexpected result: want %v, but got %v", tt.wantSuccess, got)
				}
				if tt.wantUnschedulablePods != len(q.unschedulableEntities.entityInfoMap) {
					t.Errorf("Unexpected unschedulableEntities: want %v, but got %v", tt.wantUnschedulablePods, len(q.unschedulableEntities.entityInfoMap))
				}

				// Simulate an update event.
				clone := tt.pod.DeepCopy()
				metav1.SetMetaDataAnnotation(&clone.ObjectMeta, "foo", "")
				q.Update(ctx, tt.pod, clone)
				// Ensure the pod is still located in unschedulableEntities.
				if tt.wantUnschedulablePods != len(q.unschedulableEntities.entityInfoMap) {
					t.Errorf("Unexpected unschedulableEntities: want %v, but got %v", tt.wantUnschedulablePods, len(q.unschedulableEntities.entityInfoMap))
				}
			})
		}
	}
}

func TestPriorityQueue_moveToBackoffQ(t *testing.T) {
	tests := []struct {
		name                   string
		plugins                []fwk.PreEnqueuePlugin
		pod                    *v1.Pod
		popFromBackoffQEnabled []bool
		wantSuccess            bool
	}{
		{
			name:        "no plugins registered",
			pod:         st.MakePod().Name("p").Label("p", "").Obj(),
			wantSuccess: true,
		},
		{
			name:                   "preEnqueue plugin registered, pod name would not be in allowlists",
			plugins:                []fwk.PreEnqueuePlugin{&preEnqueuePlugin{}, &preEnqueuePlugin{}},
			pod:                    st.MakePod().Name("p").Label("p", "").Obj(),
			popFromBackoffQEnabled: []bool{false},
			wantSuccess:            true,
		},
		{
			name:                   "preEnqueue plugin registered, pod name not in allowlists",
			plugins:                []fwk.PreEnqueuePlugin{&preEnqueuePlugin{}, &preEnqueuePlugin{}},
			pod:                    st.MakePod().Name("p").Label("p", "").Obj(),
			popFromBackoffQEnabled: []bool{true},
			wantSuccess:            false,
		},
		{
			name: "preEnqueue plugin registered, preEnqueue plugin would reject the pod, but isn't run",
			plugins: []fwk.PreEnqueuePlugin{
				&preEnqueuePlugin{allowlists: []string{"foo", "bar"}},
				&preEnqueuePlugin{allowlists: []string{"foo"}},
			},
			pod:                    st.MakePod().Name("bar").Label("bar", "").Obj(),
			popFromBackoffQEnabled: []bool{false},
			wantSuccess:            true,
		},
		{
			name: "preEnqueue plugin registered, pod failed one preEnqueue plugin",
			plugins: []fwk.PreEnqueuePlugin{
				&preEnqueuePlugin{allowlists: []string{"foo", "bar"}},
				&preEnqueuePlugin{allowlists: []string{"foo"}},
			},
			pod:                    st.MakePod().Name("bar").Label("bar", "").Obj(),
			popFromBackoffQEnabled: []bool{true},
			wantSuccess:            false,
		},
		{
			name: "preEnqueue plugin registered, pod passed all preEnqueue plugins",
			plugins: []fwk.PreEnqueuePlugin{
				&preEnqueuePlugin{allowlists: []string{"foo", "bar"}},
				&preEnqueuePlugin{allowlists: []string{"bar"}},
			},
			pod:         st.MakePod().Name("bar").Label("bar", "").Obj(),
			wantSuccess: true,
		},
	}

	for _, tt := range tests {
		if tt.popFromBackoffQEnabled == nil {
			tt.popFromBackoffQEnabled = []bool{true, false}
		}
		for _, popFromBackoffQEnabled := range tt.popFromBackoffQEnabled {
			t.Run(fmt.Sprintf("%s popFromBackoffQEnabled(%v)", tt.name, popFromBackoffQEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerPopFromBackoffQ, popFromBackoffQEnabled)
				logger, ctx := ktesting.NewTestContext(t)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()

				m := map[string]map[string]fwk.PreEnqueuePlugin{"": make(map[string]fwk.PreEnqueuePlugin, len(tt.plugins))}
				for _, plugin := range tt.plugins {
					m[""][plugin.Name()] = plugin
				}
				q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{tt.pod}, WithPreEnqueuePluginMap(m),
					WithPodInitialBackoffDuration(time.Second*30), WithPodMaxBackoffDuration(time.Second*60))
				pInfo := q.newQueuedPodInfo(ctx, tt.pod)
				got := q.moveToBackoffQ(logger, pInfo, framework.EventUnscheduledPodAdd.Label())
				if got != tt.wantSuccess {
					t.Errorf("Unexpected result: want %v, but got %v", tt.wantSuccess, got)
				}
				if tt.wantSuccess {
					if !q.backoffQ.has(pInfo) {
						t.Errorf("Expected pod to be in backoffQ, but it isn't")
					}
					if q.unschedulableEntities.get(pInfo) != nil {
						t.Errorf("Expected pod not to be in unschedulableEntities, but it is")
					}
				} else {
					if q.backoffQ.has(pInfo) {
						t.Errorf("Expected pod not to be in backoffQ, but it is")
					}
					if q.unschedulableEntities.get(pInfo) == nil {
						t.Errorf("Expected pod to be in unschedulableEntities, but it isn't")
					}
				}
			})
		}
	}
}

func BenchmarkMoveAllToActiveOrBackoffQueue(b *testing.B) {
	tests := []struct {
		name      string
		moveEvent fwk.ClusterEvent
	}{
		{
			name:      "baseline",
			moveEvent: framework.EventUnschedulableTimeout,
		},
		{
			name:      "worst",
			moveEvent: nodeAdd,
		},
		{
			name: "random",
			// leave "moveEvent" unspecified
		},
	}

	podTemplates := []*v1.Pod{
		highPriorityPodInfo.Pod, highPriNominatedPodInfo.Pod,
		medPriorityPodInfo.Pod, unschedulablePodInfo.Pod,
	}

	events := []fwk.ClusterEvent{
		{Resource: fwk.Node, ActionType: fwk.Add},
		{Resource: fwk.Node, ActionType: fwk.UpdateNodeTaint},
		{Resource: fwk.Node, ActionType: fwk.UpdateNodeAllocatable},
		{Resource: fwk.Node, ActionType: fwk.UpdateNodeCondition},
		{Resource: fwk.Node, ActionType: fwk.UpdateNodeLabel},
		{Resource: fwk.Node, ActionType: fwk.UpdateNodeAnnotation},
		{Resource: fwk.PersistentVolumeClaim, ActionType: fwk.Add},
		{Resource: fwk.PersistentVolumeClaim, ActionType: fwk.Update},
		{Resource: fwk.PersistentVolume, ActionType: fwk.Add},
		{Resource: fwk.PersistentVolume, ActionType: fwk.Update},
		{Resource: fwk.StorageClass, ActionType: fwk.Add},
		{Resource: fwk.StorageClass, ActionType: fwk.Update},
		{Resource: fwk.CSINode, ActionType: fwk.Add},
		{Resource: fwk.CSINode, ActionType: fwk.Update},
		{Resource: fwk.CSIDriver, ActionType: fwk.Add},
		{Resource: fwk.CSIDriver, ActionType: fwk.Update},
		{Resource: fwk.CSIStorageCapacity, ActionType: fwk.Add},
		{Resource: fwk.CSIStorageCapacity, ActionType: fwk.Update},
	}

	pluginNum := 20
	var plugins []string
	// Mimic that we have 20 plugins loaded in runtime.
	for i := 0; i < pluginNum; i++ {
		plugins = append(plugins, fmt.Sprintf("fake-plugin-%v", i))
	}

	for _, tt := range tests {
		for _, podsInUnschedulablePods := range []int{1000, 5000} {
			b.Run(fmt.Sprintf("%v-%v", tt.name, podsInUnschedulablePods), func(b *testing.B) {
				logger, ctx := ktesting.NewTestContext(b)
				for i := 0; i < b.N; i++ {
					b.StopTimer()
					c := testingclock.NewFakeClock(time.Now())

					m := makeEmptyQueueingHintMapPerProfile()
					// - All plugins registered for events[0], which is NodeAdd.
					// - 1/2 of plugins registered for events[1]
					// - 1/3 of plugins registered for events[2]
					// - ...
					for j := 0; j < len(events); j++ {
						for k := 0; k < len(plugins); k++ {
							if (k+1)%(j+1) == 0 {
								m[""][events[j]] = append(m[""][events[j]], &QueueingHintFunction{
									PluginName:     plugins[k],
									QueueingHintFn: queueHintReturnQueue,
								})
							}
						}
					}

					ctx, cancel := context.WithCancel(ctx)
					defer cancel()
					q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(c), WithQueueingHintMapPerProfile(m))

					// Init pods in unschedulableEntities.
					for j := 0; j < podsInUnschedulablePods; j++ {
						p := podTemplates[j%len(podTemplates)].DeepCopy()
						p.Name, p.UID = fmt.Sprintf("%v-%v", p.Name, j), types.UID(fmt.Sprintf("%v-%v", p.UID, j))
						var podInfo *framework.QueuedPodInfo
						// The ultimate goal of composing each PodInfo is to cover the path that intersects
						// (unschedulable) plugin names with the plugins that register the moveEvent,
						// here the rational is:
						// - in baseline case, don't inject unschedulable plugin names, so podMatchesEvent()
						//   never gets executed.
						// - in worst case, make both ends (of the intersection) a big number,i.e.,
						//   M intersected with N instead of M with 1 (or 1 with N)
						// - in random case, each pod failed by a random plugin, and also the moveEvent
						//   is randomized.
						if tt.name == "baseline" {
							podInfo = q.newQueuedPodInfo(ctx, p)
						} else if tt.name == "worst" {
							// Each pod failed by all plugins.
							podInfo = q.newQueuedPodInfo(ctx, p, plugins...)
						} else {
							// Random case.
							podInfo = q.newQueuedPodInfo(ctx, p, plugins[j%len(plugins)])
						}
						err := q.AddUnschedulablePodIfNotPresent(logger, podInfo, q.SchedulingCycle())
						if err != nil {
							b.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
						}
					}

					b.StartTimer()
					if tt.moveEvent.Resource != "" {
						q.MoveAllToActiveOrBackoffQueue(logger, tt.moveEvent, nil, nil, nil)
					} else {
						// Random case.
						q.MoveAllToActiveOrBackoffQueue(logger, events[i%len(events)], nil, nil, nil)
					}
				}
			})
		}
	}
}

func TestPriorityQueue_MoveAllToActiveOrBackoffQueueWithQueueingHint(t *testing.T) {
	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	p := st.MakePod().Name("pod1").Namespace("ns1").UID("1").Label("foo", "bar").Obj()
	gatedPod := st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()
	tests := []struct {
		name    string
		podInfo *framework.QueuedPodInfo
		hint    fwk.QueueingHintFn
		// duration is the duration that the Pod has been in the unschedulable queue.
		duration time.Duration
		// triggerEvent is the event to trigger the move. If unset, defaults to nodeAdd.
		triggerEvent *fwk.ClusterEvent
		// expectedQ is the queue name (activeQ, backoffQ, or unschedulableEntities) that this Pod should be queued to.
		expectedQ string
	}{
		{
			name: "Queue queues pod to activeQ",
			// This pod has PendingPlugins and hence will be pushed directly to activeQ
			podInfo: &framework.QueuedPodInfo{
				PodInfo: mustNewPodInfo(p),
				QueueingParams: framework.QueueingParams{
					PendingPlugins: sets.New("foo"),
				},
			},
			hint:      queueHintReturnQueue,
			expectedQ: activeQ,
		},
		{
			name: "Queue queues pod to backoffQ if Pod is backing off",
			// needs UnschedulableCount to make it backing off.
			podInfo: &framework.QueuedPodInfo{
				PodInfo: mustNewPodInfo(p),
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("foo"),
				},
			},
			hint:      queueHintReturnQueue,
			expectedQ: backoffQ,
		},
		{
			name: "Queue queues pod to activeQ if Pod is not backing off",
			podInfo: &framework.QueuedPodInfo{
				PodInfo: mustNewPodInfo(p),
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("foo"),
				},
			},
			hint: queueHintReturnQueue,
			// The pod is assumed to failed the scheduling cycle once, which would get DefaultPodInitialBackoffDuration as the penalty.
			// To finish the backoff, waiting for DefaultPodInitialBackoffDuration isn't enough, need to wait for +1
			// because the pod is determined to be still backing off if `{backoff expiration time} == trancate({current time})`
			duration:  DefaultPodInitialBackoffDuration + time.Second,
			expectedQ: activeQ,
		},
		{
			name: "Skip queues pod to unschedulableEntities",
			podInfo: &framework.QueuedPodInfo{
				PodInfo: mustNewPodInfo(p),
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("foo"),
				},
			},
			hint:      queueHintReturnSkip,
			expectedQ: unschedulableQ,
		},
		{
			name: "Queue queues pod to backoffQ if Pod is not gated and the event is wildcard",
			podInfo: &framework.QueuedPodInfo{
				PodInfo: mustNewPodInfo(p),
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("foo"),
				},
			},
			triggerEvent: &framework.EventUnschedulableTimeout,
			hint: func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
				return fwk.Queue, fmt.Errorf("QueueingHintFn should not be called as trigger event is wildcard")
			},
			expectedQ: backoffQ,
		},
		{
			name:         "Queue queues pod to backoffQ when Pod is no longer gated and the event is wildcard",
			podInfo:      setQueuedPodInfoGated(&framework.QueuedPodInfo{PodInfo: mustNewPodInfo(p)}, "foo", []fwk.ClusterEvent{framework.EventUnscheduledPodUpdate}),
			triggerEvent: &framework.EventUnschedulableTimeout,
			hint: func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
				return fwk.Queue, fmt.Errorf("QueueingHintFn should not be called as trigger event is wildcard")
			},
			expectedQ: backoffQ,
		},
		{
			name:         "Queue queues pod to unschedulableQ when Pod is gated and the event is wildcard",
			podInfo:      setQueuedPodInfoGated(&framework.QueuedPodInfo{PodInfo: mustNewPodInfo(gatedPod)}, "foo", []fwk.ClusterEvent{framework.EventUnscheduledPodUpdate}),
			triggerEvent: &framework.EventUnschedulableTimeout,
			hint: func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
				return fwk.Queue, fmt.Errorf("QueueingHintFn should not be called as trigger event is wildcard")
			},
			expectedQ: unschedulableQ,
		},
		{
			name:    "QueueHintFunction is not called when Pod is gated by the plugin that isn't interested in the event",
			podInfo: setQueuedPodInfoGated(&framework.QueuedPodInfo{PodInfo: mustNewPodInfo(p)}, names.SchedulingGates, []fwk.ClusterEvent{framework.EventUnscheduledPodUpdate}),
			// The hintFn should not be called as the pod is gated by SchedulingGates plugin,
			// the scheduling gate isn't interested in the node add event,
			// and the queue should keep this Pod in the unschedQ without calling the hintFn.
			hint: func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
				return fwk.Queue, fmt.Errorf("QueueingHintFn should not be called as pod is gated")
			},
			expectedQ: unschedulableQ,
		},
		{
			name:    "QueueHintFunction is called when Pod is gated by the plugin that is interested in the event",
			podInfo: setQueuedPodInfoGated(&framework.QueuedPodInfo{PodInfo: mustNewPodInfo(p)}, "foo", []fwk.ClusterEvent{nodeAdd}),
			// In this case, the hintFn should be called as the pod is gated by foo plugin that is interested in the NodeAdd event.
			hint: queueHintReturnQueue,
			// and, as a result, this pod should be queued to backoffQ.
			expectedQ: backoffQ,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			m := makeEmptyQueueingHintMapPerProfile()
			m[""][nodeAdd] = []*QueueingHintFunction{
				{
					PluginName:     "foo",
					QueueingHintFn: test.hint,
				},
			}
			m[""][framework.EventUnscheduledPodUpdate] = []*QueueingHintFunction{
				{
					PluginName:     names.SchedulingGates,
					QueueingHintFn: queueHintReturnQueue,
				},
			}
			cl := testingclock.NewFakeClock(now)
			plugin, _ := schedulinggates.New(ctx, nil, nil, plfeature.Features{})
			preEnqM := map[string]map[string]fwk.PreEnqueuePlugin{"": {
				names.SchedulingGates: plugin.(fwk.PreEnqueuePlugin),
				"foo":                 &preEnqueuePlugin{allowlists: []string{"foo"}},
			}}
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithQueueingHintMapPerProfile(m), WithClock(cl), WithPreEnqueuePluginMap(preEnqM))
			q.Add(ctx, test.podInfo.Pod)
			if q.activeQ.len() > 0 {
				if p, err := q.Pop(logger); err != nil {
					t.Errorf("Pop failed: %v", err)
				} else if diff := cmp.Diff(test.podInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
					t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
				}
				// add to unsched pod pool
				err := q.AddUnschedulablePodIfNotPresent(logger, test.podInfo, q.SchedulingCycle())
				if err != nil {
					t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
				}
			} else {
				// The pod was already moved to unschedulableEntities because it's gated.
				// Update it with the test's configured podInfo to ensure custom test fields are set.
				q.unschedulableEntities.addOrUpdate(test.podInfo, test.podInfo.Gated(), "test-setup")
			}
			cl.Step(test.duration)

			event := nodeAdd
			if test.triggerEvent != nil {
				event = *test.triggerEvent
			}
			q.MoveAllToActiveOrBackoffQueue(logger, event, nil, nil, nil)

			if q.backoffQ.len() == 0 && test.expectedQ == backoffQ {
				t.Fatalf("expected pod to be queued to backoffQ, but it was not")
			}

			if q.activeQ.len() == 0 && test.expectedQ == activeQ {
				t.Fatalf("expected pod to be queued to activeQ, but it was not")
			}

			if q.unschedulableEntities.get(test.podInfo) == nil && test.expectedQ == unschedulableQ {
				t.Fatalf("expected pod to be queued to unschedulableEntities, but it was not")
			}
		})
	}
}

func TestPriorityQueue_MoveAllToActiveOrBackoffQueue(t *testing.T) {
	c := testingclock.NewFakeClock(time.Now())
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	m := makeEmptyQueueingHintMapPerProfile()

	m[""][nodeAdd] = []*QueueingHintFunction{
		{
			PluginName:     "fooPlugin",
			QueueingHintFn: queueHintReturnQueue,
		},
	}
	q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(c), WithQueueingHintMapPerProfile(m))
	// To simulate the pod is failed in scheduling in the real world, Pop() the pod from activeQ before AddUnschedulablePodIfNotPresent()s below.
	q.Add(ctx, unschedulablePodInfo.Pod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(unschedulablePodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	expectInFlightPods(t, q, unschedulablePodInfo.Pod.UID)
	q.Add(ctx, highPriorityPodInfo.Pod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(highPriorityPodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	expectInFlightPods(t, q, unschedulablePodInfo.Pod.UID, highPriorityPodInfo.Pod.UID)
	err := q.AddUnschedulablePodIfNotPresent(logger, q.newQueuedPodInfo(ctx, unschedulablePodInfo.Pod, "fooPlugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	err = q.AddUnschedulablePodIfNotPresent(logger, q.newQueuedPodInfo(ctx, highPriorityPodInfo.Pod, "fooPlugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	expectInFlightPods(t, q)
	// Construct a Pod, but don't associate its scheduler failure to any plugin
	hpp1 := clonePod(highPriorityPodInfo.Pod, "hpp1")
	q.Add(ctx, hpp1)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(hpp1, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	expectInFlightPods(t, q, hpp1.UID)
	// This Pod will go to backoffQ because no failure plugin is associated with it.
	hpp1PodInfo := q.newQueuedPodInfo(ctx, hpp1)
	hpp1PodInfo.UnschedulableCount++
	err = q.AddUnschedulablePodIfNotPresent(logger, hpp1PodInfo, q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	expectInFlightPods(t, q)
	// Construct another Pod, and associate its scheduler failure to plugin "barPlugin".
	hpp2 := clonePod(highPriorityPodInfo.Pod, "hpp2")
	q.Add(ctx, hpp2)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(hpp2, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	expectInFlightPods(t, q, hpp2.UID)
	// This Pod will go to the unschedulable Pod pool.
	err = q.AddUnschedulablePodIfNotPresent(logger, q.newQueuedPodInfo(ctx, hpp2, "barPlugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	expectInFlightPods(t, q)
	// This NodeAdd event moves unschedulablePodInfo and highPriorityPodInfo to the backoffQ,
	// because of the queueing hint function registered for NodeAdd/fooPlugin.
	q.MoveAllToActiveOrBackoffQueue(logger, nodeAdd, nil, nil, nil)
	q.Add(ctx, medPriorityPodInfo.Pod)
	if q.activeQ.len() != 1 {
		t.Errorf("Expected 1 item to be in activeQ, but got: %v", q.activeQ.len())
	}
	// Pop out the medPriorityPodInfo in activeQ.
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(medPriorityPodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	expectInFlightPods(t, q, medPriorityPodInfo.Pod.UID)
	// hpp2 won't be moved.
	if q.backoffQ.len() != 3 {
		t.Fatalf("Expected 3 items to be in backoffQ, but got: %v", q.backoffQ.len())
	}

	// pop out the pods in the backoffQ.
	// This doesn't make them in-flight pods.
	c.Step(q.backoffQ.podMaxBackoffDuration())
	_ = q.backoffQ.popAllBackoffCompleted(logger)
	expectInFlightPods(t, q, medPriorityPodInfo.Pod.UID)

	q.Add(ctx, unschedulablePodInfo.Pod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(unschedulablePodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	expectInFlightPods(t, q, medPriorityPodInfo.Pod.UID, unschedulablePodInfo.Pod.UID)
	q.Add(ctx, highPriorityPodInfo.Pod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(highPriorityPodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	expectInFlightPods(t, q, medPriorityPodInfo.Pod.UID, unschedulablePodInfo.Pod.UID, highPriorityPodInfo.Pod.UID)
	q.Add(ctx, hpp1)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(hpp1, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	unschedulableQueuedPodInfo := q.newQueuedPodInfo(ctx, unschedulablePodInfo.Pod, "fooPlugin")
	highPriorityQueuedPodInfo := q.newQueuedPodInfo(ctx, highPriorityPodInfo.Pod, "fooPlugin")
	hpp1QueuedPodInfo := q.newQueuedPodInfo(ctx, hpp1)
	expectInFlightPods(t, q, medPriorityPodInfo.Pod.UID, unschedulablePodInfo.Pod.UID, highPriorityPodInfo.Pod.UID, hpp1.UID)
	err = q.AddUnschedulablePodIfNotPresent(logger, unschedulableQueuedPodInfo, q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	expectInFlightPods(t, q, medPriorityPodInfo.Pod.UID, highPriorityPodInfo.Pod.UID, hpp1.UID)
	err = q.AddUnschedulablePodIfNotPresent(logger, highPriorityQueuedPodInfo, q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	expectInFlightPods(t, q, medPriorityPodInfo.Pod.UID, hpp1.UID)
	err = q.AddUnschedulablePodIfNotPresent(logger, hpp1QueuedPodInfo, q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	expectInFlightPods(t, q, medPriorityPodInfo.Pod.UID)
	q.Add(ctx, medPriorityPodInfo.Pod)
	// hpp1 will go to backoffQ because no failure plugin is associated with it.
	// All plugins other than hpp1 are enqueued to the unschedulable Pod pool.
	for _, pod := range []*v1.Pod{unschedulablePodInfo.Pod, highPriorityPodInfo.Pod, hpp2} {
		if q.unschedulableEntities.get(newQueuedPodInfoForLookup(pod)) == nil {
			t.Errorf("Expected %v in the unschedulableEntities", pod.Name)
		}
	}
	if !q.backoffQ.has(hpp1QueuedPodInfo) {
		t.Errorf("Expected %v in the backoffQ", hpp1.Name)
	}
	// all the remaining Pods should be in activeQ.
	if q.activeQ.len() != 1 {
		t.Errorf("Expected %v in the activeQ", medPriorityPodInfo.Pod.Name)
	}

	// Move clock by podMaxBackoffDuration, so that pods in the unschedulableEntities would pass the backing off,
	// and the pods will be moved into activeQ.
	c.Step(q.backoffQ.podMaxBackoffDuration())
	q.flushBackoffQCompleted(logger) // flush the completed backoffQ to move hpp1 to activeQ.
	q.MoveAllToActiveOrBackoffQueue(logger, nodeAdd, nil, nil, nil)
	if q.activeQ.len() != 4 {
		t.Errorf("Expected 4 items to be in activeQ, but got: %v", q.activeQ.len())
	}
	if q.backoffQ.len() != 0 {
		t.Errorf("Expected 0 item to be in backoffQ, but got: %v", q.backoffQ.len())
	}
	expectInFlightPods(t, q, medPriorityPodInfo.Pod.UID)
	if len(q.unschedulableEntities.entityInfoMap) != 1 {
		// hpp2 won't be moved regardless of its backoff timer.
		t.Errorf("Expected 1 item to be in unschedulableEntities, but got: %v", len(q.unschedulableEntities.entityInfoMap))
	}
}

func clonePod(pod *v1.Pod, newName string) *v1.Pod {
	pod = pod.DeepCopy()
	pod.Name = newName
	pod.UID = types.UID(pod.Name + pod.Namespace)
	return pod
}

func expectInFlightPods(t *testing.T, q *PriorityQueue, uids ...types.UID) {
	t.Helper()
	var actualUIDs []types.UID
	for _, pod := range q.activeQ.listInFlightPods() {
		actualUIDs = append(actualUIDs, pod.UID)
	}
	sortUIDs := cmpopts.SortSlices(func(a, b types.UID) bool { return a < b })
	if diff := cmp.Diff(uids, actualUIDs, sortUIDs); diff != "" {
		t.Fatalf("Unexpected content of inFlightPods (-want, +have):\n%s", diff)
	}
	actualUIDs = nil
	events := q.activeQ.listInFlightEvents()
	for _, e := range events {
		if pod, ok := e.(*v1.Pod); ok {
			actualUIDs = append(actualUIDs, pod.UID)
		}
	}
	if diff := cmp.Diff(uids, actualUIDs, sortUIDs); diff != "" {
		t.Fatalf("Unexpected pods in inFlightEvents (-want, +have):\n%s", diff)
	}
}

func TestPriorityQueue_NominatedPodsForNode(t *testing.T) {
	objs := []runtime.Object{medPriorityPodInfo.Pod, unschedulablePodInfo.Pod, highPriorityPodInfo.Pod}
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), objs)
	q.Add(ctx, medPriorityPodInfo.Pod)
	q.Add(ctx, unschedulablePodInfo.Pod)
	q.Add(ctx, highPriorityPodInfo.Pod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(highPriorityPodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	expectedList := []fwk.PodInfo{medPriorityPodInfo, unschedulablePodInfo}
	podInfos := q.NominatedPodsForNode("node1")
	if diff := cmp.Diff(expectedList, podInfos, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
		t.Errorf("Unexpected list of nominated Pods for node: (-want, +got):\n%s", diff)
	}
	podInfos[0].GetPod().Name = "not mpp"
	if diff := cmp.Diff(podInfos, q.NominatedPodsForNode("node1"), cmpopts.IgnoreUnexported(framework.PodInfo{})); diff == "" {
		t.Error("Expected list of nominated Pods for node2 is different from podInfos")
	}
	if len(q.NominatedPodsForNode("node2")) != 0 {
		t.Error("Expected list of nominated Pods for node2 to be empty.")
	}
}

func TestPriorityQueue_NominatedPodDeleted(t *testing.T) {
	tests := []struct {
		name      string
		podInfo   *framework.PodInfo
		deletePod bool
		wantLen   int
	}{
		{
			name:    "alive pod gets added into PodNominator",
			podInfo: medPriorityPodInfo,
			wantLen: 1,
		},
		{
			name:      "deleted pod shouldn't be added into PodNominator",
			podInfo:   highPriNominatedPodInfo,
			deletePod: true,
			wantLen:   0,
		},
		{
			name:    "pod without .status.nominatedPodName specified shouldn't be added into PodNominator",
			podInfo: highPriorityPodInfo,
			wantLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			cs := fake.NewClientset(tt.podInfo.Pod)
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			podLister := informerFactory.Core().V1().Pods().Lister()

			// Build a PriorityQueue.
			q := NewPriorityQueue(newDefaultQueueSort(), informerFactory, WithPodLister(podLister))
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			if tt.deletePod {
				// Simulate that the test pod gets deleted physically.
				informerFactory.Core().V1().Pods().Informer().GetStore().Delete(tt.podInfo.Pod)
			}

			q.AddNominatedPod(logger, tt.podInfo, nil)

			if got := len(q.NominatedPodsForNode(tt.podInfo.Pod.Status.NominatedNodeName)); got != tt.wantLen {
				t.Errorf("Expected %v nominated pods for node, but got %v", tt.wantLen, got)
			}
		})
	}
}

// TestPriorityQueue_NominatedNodeNameEmptyNodeKey ensures an empty NominatedNodeName does not
// store pods under nominatedPods[""] (#138267), including ModeNoop behavior.
func TestPriorityQueue_NominatedNodeNameEmptyNodeKey(t *testing.T) {
	tests := []struct {
		name                 string
		initialNominatedNode string // if set, nominate the pod to this node before the test call
		podNominatedNode     string // pod.Status.NominatedNodeName
		nominatingMode       fwk.NominatingMode
		nominatingNodeName   string // NominatedNodeName in nominatingInfo
		wantNominatedNode    string // expected node after the call; empty means no nomination
	}{
		{
			name:           "ModeOverride empty without prior nomination",
			nominatingMode: fwk.ModeOverride,
		},
		{
			name:                 "nominated then cleared with ModeOverride empty",
			initialNominatedNode: "node1",
			nominatingMode:       fwk.ModeOverride,
		},
		{
			name:           "ModeNoop empty without prior nomination",
			nominatingMode: fwk.ModeNoop,
		},
		{
			name:                 "nominated then cleared with ModeNoop empty",
			initialNominatedNode: "node1",
			nominatingMode:       fwk.ModeNoop,
		},
		{
			name:               "ModeNoop with non-empty nominatingInfo NNN ignored when pod NNN empty",
			nominatingMode:     fwk.ModeNoop,
			nominatingNodeName: "node1",
		},
		{
			name:               "ModeNoop with non-empty nominatingInfo NNN uses pod NNN",
			nominatingMode:     fwk.ModeNoop,
			podNominatedNode:   "node2",
			nominatingNodeName: "node1",
			wantNominatedNode:  "node2",
		},
		{
			name:              "ModeNoop empty nominatingInfo NNN with non-empty pod NNN",
			nominatingMode:    fwk.ModeNoop,
			podNominatedNode:  "node2",
			wantNominatedNode: "node2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podBuilder := st.MakePod().Name("hpp").Namespace("ns1").UID("hppns1").Priority(highPriority)
			if tt.podNominatedNode != "" {
				podBuilder = podBuilder.NominatedNodeName(tt.podNominatedNode)
			}
			podInfo := mustNewTestPodInfo(t, podBuilder.Obj())

			logger, ctx := ktesting.NewTestContext(t)
			cs := fake.NewClientset(podInfo.Pod)
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			q := NewPriorityQueue(newDefaultQueueSort(), informerFactory, WithPodLister(informerFactory.Core().V1().Pods().Lister()))
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			if tt.initialNominatedNode != "" {
				q.AddNominatedPod(logger, podInfo, &fwk.NominatingInfo{
					NominatingMode:    fwk.ModeOverride,
					NominatedNodeName: tt.initialNominatedNode,
				})
				if got := len(q.nominator.nominatedPods[tt.initialNominatedNode]); got != 1 {
					t.Fatalf("expected 1 nominated pod on %s before clear, got %d", tt.initialNominatedNode, got)
				}
			}

			q.AddNominatedPod(logger, podInfo, &fwk.NominatingInfo{
				NominatingMode:    tt.nominatingMode,
				NominatedNodeName: tt.nominatingNodeName,
			})

			if len(q.nominator.nominatedPods[""]) != 0 {
				t.Errorf("expected no pods under nominatedPods[\"\"], got %v", q.nominator.nominatedPods[""])
			}
			if tt.wantNominatedNode == "" {
				if len(q.nominator.nominatedPods) != 0 {
					t.Errorf("expected nominatedPods empty, got %v", q.nominator.nominatedPods)
				}
				if len(q.nominator.nominatedPodToNode) != 0 {
					t.Errorf("expected nominatedPodToNode empty, got %v", q.nominator.nominatedPodToNode)
				}
			} else {
				if got := len(q.nominator.nominatedPods[tt.wantNominatedNode]); got != 1 {
					t.Errorf("expected 1 nominated pod on %s, got %d", tt.wantNominatedNode, got)
				}
				if got := q.nominator.nominatedPodToNode[podInfo.Pod.UID]; got != tt.wantNominatedNode {
					t.Errorf("expected nominatedPodToNode[%s]=%s, got %s", podInfo.Pod.UID, tt.wantNominatedNode, got)
				}
			}
		})
	}
}

func TestPriorityQueue_PendingPods(t *testing.T) {
	makeSet := func(pods []*v1.Pod) map[*v1.Pod]struct{} {
		pendingSet := map[*v1.Pod]struct{}{}
		for _, p := range pods {
			pendingSet[p] = struct{}{}
		}
		return pendingSet
	}

	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueue(ctx, newDefaultQueueSort())
	// To simulate the pod is failed in scheduling in the real world, Pop() the pod from activeQ before AddUnschedulablePodIfNotPresent()s below.
	q.Add(ctx, unschedulablePodInfo.Pod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(unschedulablePodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	q.Add(ctx, highPriorityPodInfo.Pod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(highPriorityPodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	q.Add(ctx, medPriorityPodInfo.Pod)
	err := q.AddUnschedulablePodIfNotPresent(logger, q.newQueuedPodInfo(ctx, unschedulablePodInfo.Pod, "plugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	err = q.AddUnschedulablePodIfNotPresent(logger, q.newQueuedPodInfo(ctx, highPriorityPodInfo.Pod, "plugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}

	expectedSet := makeSet([]*v1.Pod{medPriorityPodInfo.Pod, unschedulablePodInfo.Pod, highPriorityPodInfo.Pod})
	gotPods, gotSummary := q.PendingPods()
	if diff := cmp.Diff(expectedSet, makeSet(gotPods)); diff != "" {
		t.Errorf("Unexpected list of pending Pods (-want, +got):\n%s", diff)
	}
	if wantSummary := fmt.Sprintf(pendingPodsSummary, 1, 0, 2); wantSummary != gotSummary {
		t.Errorf("Unexpected pending pods summary: want %v, but got %v.", wantSummary, gotSummary)
	}
	// Move all to active queue. We should still see the same set of pods.
	q.MoveAllToActiveOrBackoffQueue(logger, framework.EventUnschedulableTimeout, nil, nil, nil)
	gotPods, gotSummary = q.PendingPods()
	if diff := cmp.Diff(expectedSet, makeSet(gotPods)); diff != "" {
		t.Errorf("Unexpected list of pending Pods (-want, +got):\n%s", diff)
	}
	if wantSummary := fmt.Sprintf(pendingPodsSummary, 1, 2, 0); wantSummary != gotSummary {
		t.Errorf("Unexpected pending pods summary: want %v, but got %v.", wantSummary, gotSummary)
	}
}

func TestPriorityQueue_UpdateNominatedPodForNode(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	objs := []runtime.Object{medPriorityPodInfo.Pod, unschedulablePodInfo.Pod, highPriorityPodInfo.Pod, scheduledPodInfo.Pod}
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), objs)
	q.Add(ctx, medPriorityPodInfo.Pod)
	// Update unschedulablePodInfo on a different node than specified in the pod.
	q.AddNominatedPod(logger, mustNewTestPodInfo(t, unschedulablePodInfo.Pod),
		&fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node5"})

	// Update nominated node name of a pod on a node that is not specified in the pod object.
	q.AddNominatedPod(logger, mustNewTestPodInfo(t, highPriorityPodInfo.Pod),
		&fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node2"})
	expectedNominatedPods := &nominator{
		nominatedPodToNode: map[types.UID]string{
			medPriorityPodInfo.Pod.UID:   "node1",
			highPriorityPodInfo.Pod.UID:  "node2",
			unschedulablePodInfo.Pod.UID: "node5",
		},
		nominatedPods: map[string][]podRef{
			"node1": {podToRef(medPriorityPodInfo.Pod)},
			"node2": {podToRef(highPriorityPodInfo.Pod)},
			"node5": {podToRef(unschedulablePodInfo.Pod)},
		},
	}
	if diff := cmp.Diff(q.nominator, expectedNominatedPods, nominatorCmpOpts...); diff != "" {
		t.Errorf("Unexpected diff after adding pods (-want, +got):\n%s", diff)
	}
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(medPriorityPodInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	// List of nominated pods shouldn't change after popping them from the queue.
	if diff := cmp.Diff(q.nominator, expectedNominatedPods, nominatorCmpOpts...); diff != "" {
		t.Errorf("Unexpected diff after popping pods (-want, +got):\n%s", diff)
	}
	// Update one of the nominated pods that doesn't have nominatedNodeName in the
	// pod object. It should be updated correctly.
	q.AddNominatedPod(logger, highPriorityPodInfo, &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node4"})
	expectedNominatedPods = &nominator{
		nominatedPodToNode: map[types.UID]string{
			medPriorityPodInfo.Pod.UID:   "node1",
			highPriorityPodInfo.Pod.UID:  "node4",
			unschedulablePodInfo.Pod.UID: "node5",
		},
		nominatedPods: map[string][]podRef{
			"node1": {podToRef(medPriorityPodInfo.Pod)},
			"node4": {podToRef(highPriorityPodInfo.Pod)},
			"node5": {podToRef(unschedulablePodInfo.Pod)},
		},
	}
	if diff := cmp.Diff(q.nominator, expectedNominatedPods, nominatorCmpOpts...); diff != "" {
		t.Errorf("Unexpected diff after updating pods (-want, +got):\n%s", diff)
	}

	// Attempt to nominate a pod that was deleted from the informer cache.
	// Nothing should change.
	q.AddNominatedPod(logger, nonExistentPodInfo, &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node1"})
	if diff := cmp.Diff(q.nominator, expectedNominatedPods, nominatorCmpOpts...); diff != "" {
		t.Errorf("Unexpected diff after nominating a deleted pod (-want, +got):\n%s", diff)
	}
	// Attempt to nominate a pod that was already scheduled in the informer cache.
	// Nothing should change.
	scheduledPodCopy := scheduledPodInfo.Pod.DeepCopy()
	scheduledPodInfo.Pod.Spec.NodeName = ""
	q.AddNominatedPod(logger, mustNewTestPodInfo(t, scheduledPodCopy), &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node1"})
	if diff := cmp.Diff(q.nominator, expectedNominatedPods, nominatorCmpOpts...); diff != "" {
		t.Errorf("Unexpected diff after nominating a scheduled pod (-want, +got):\n%s", diff)
	}

	// Delete a nominated pod that doesn't have nominatedNodeName in the pod
	// object. It should be deleted.
	q.DeleteNominatedPodIfExists(highPriorityPodInfo.Pod)
	expectedNominatedPods = &nominator{
		nominatedPodToNode: map[types.UID]string{
			medPriorityPodInfo.Pod.UID:   "node1",
			unschedulablePodInfo.Pod.UID: "node5",
		},
		nominatedPods: map[string][]podRef{
			"node1": {podToRef(medPriorityPodInfo.Pod)},
			"node5": {podToRef(unschedulablePodInfo.Pod)},
		},
	}
	if diff := cmp.Diff(q.nominator, expectedNominatedPods, nominatorCmpOpts...); diff != "" {
		t.Errorf("Unexpected diff after deleting pods (-want, +got):\n%s", diff)
	}
}

func TestPriorityQueue_NewWithOptions(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueue(ctx,
		newDefaultQueueSort(),
		WithPodInitialBackoffDuration(2*time.Second),
		WithPodMaxBackoffDuration(20*time.Second),
	)

	if q.backoffQ.podInitialBackoffDuration() != 2*time.Second {
		t.Errorf("Unexpected pod backoff initial duration. Expected: %v, got: %v", 2*time.Second, q.backoffQ.podInitialBackoffDuration())
	}

	if q.backoffQ.podMaxBackoffDuration() != 20*time.Second {
		t.Errorf("Unexpected pod backoff max duration. Expected: %v, got: %v", 2*time.Second, q.backoffQ.podMaxBackoffDuration())
	}
}

func TestSchedulingQueue_Close(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueue(ctx, newDefaultQueueSort())
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		entity, err := q.Pop(logger)
		if err != nil {
			t.Errorf("Expected nil err from Pop() if queue is closed, but got %q", err.Error())
		}
		if entity != nil {
			t.Errorf("Expected nil item from Pop() if queue is closed, but got: %v", entity)
		}
	}()
	q.Close()
	wg.Wait()
}

// TestRecentlyTriedPodsGoBack tests that pods which are recently tried and are
// unschedulable go behind other pods with the same priority. This behavior
// ensures that an unschedulable pod does not block head of the queue when there
// are frequent events that move pods to the active queue.
func TestRecentlyTriedPodsGoBack(t *testing.T) {
	c := testingclock.NewFakeClock(time.Now())
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(c))
	// Add a few pods to priority queue.
	for i := 0; i < 5; i++ {
		p := st.MakePod().Name(fmt.Sprintf("test-pod-%v", i)).Namespace("ns1").UID(fmt.Sprintf("tp00%v", i)).Priority(highPriority).Node("node1").NominatedNodeName("node1").Obj()
		q.Add(ctx, p)
	}
	c.Step(time.Microsecond)
	// Simulate a pod being popped by the scheduler, determined unschedulable, and
	// then moved back to the active queue.
	entity, err := q.Pop(logger)
	if err != nil {
		t.Errorf("Error while popping the head of the queue: %v", err)
	}
	p1 := entity.(*framework.QueuedPodInfo)
	// Update pod condition to unschedulable.
	podutil.UpdatePodCondition(&p1.PodInfo.Pod.Status, &v1.PodCondition{
		Type:          v1.PodScheduled,
		Status:        v1.ConditionFalse,
		Reason:        v1.PodReasonUnschedulable,
		Message:       "fake scheduling failure",
		LastProbeTime: metav1.Now(),
	})
	p1.UnschedulablePlugins = sets.New("plugin")
	// Put in the unschedulable queue.
	err = q.AddUnschedulablePodIfNotPresent(logger, p1, q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	c.Step(q.backoffQ.podMaxBackoffDuration())
	// Move all unschedulable pods to the active queue.
	q.MoveAllToActiveOrBackoffQueue(logger, framework.EventUnschedulableTimeout, nil, nil, nil)
	// Simulation is over. Now let's pop all pods. The pod popped first should be
	// the last one we pop here.
	for i := 0; i < 5; i++ {
		entity, err := q.Pop(logger)
		if err != nil {
			t.Errorf("Error while popping pods from the queue: %v", err)
		}
		p := entity.(*framework.QueuedPodInfo)
		if (i == 4) != (p1 == p) {
			t.Errorf("A pod tried before is not the last pod popped: i: %v, pod name: %v", i, p.PodInfo.Pod.Name)
		}
	}
}

// TestPodFailedSchedulingMultipleTimesDoesNotBlockNewerPod tests
// that a pod determined as unschedulable multiple times doesn't block any newer pod.
// This behavior ensures that an unschedulable pod does not block head of the queue when there
// are frequent events that move pods to the active queue.
func TestPodFailedSchedulingMultipleTimesDoesNotBlockNewerPod(t *testing.T) {
	c := testingclock.NewFakeClock(time.Now())
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(c))

	// Add an unschedulable pod to a priority queue.
	// This makes a situation that the pod was tried to schedule
	// and had been determined unschedulable so far
	unschedulablePod := st.MakePod().Name("test-pod-unscheduled").Namespace("ns1").UID("tp001").Priority(highPriority).NominatedNodeName("node1").Obj()

	// Update pod condition to unschedulable.
	podutil.UpdatePodCondition(&unschedulablePod.Status, &v1.PodCondition{
		Type:    v1.PodScheduled,
		Status:  v1.ConditionFalse,
		Reason:  v1.PodReasonUnschedulable,
		Message: "fake scheduling failure",
	})

	// To simulate the pod is failed in scheduling in the real world, Pop() the pod from activeQ before AddUnschedulablePodIfNotPresent() below.
	q.Add(ctx, unschedulablePod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(unschedulablePod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	// Put in the unschedulable queue
	err := q.AddUnschedulablePodIfNotPresent(logger, newQueuedPodInfoForLookup(unschedulablePod, "plugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	// Move clock to make the unschedulable pods complete backoff.
	c.Step(DefaultPodInitialBackoffDuration + time.Second)
	// Move all unschedulable pods to the active queue.
	q.MoveAllToActiveOrBackoffQueue(logger, framework.EventUnschedulableTimeout, nil, nil, nil)

	// Simulate a pod being popped by the scheduler,
	// At this time, unschedulable pod should be popped.
	entity, err := q.Pop(logger)
	if err != nil {
		t.Errorf("Error while popping the head of the queue: %v", err)
	}
	p1 := entity.(*framework.QueuedPodInfo)
	if diff := cmp.Diff(unschedulablePod, p1.Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}

	// Assume newer pod was added just after unschedulable pod
	// being popped and before being pushed back to the queue.
	newerPod := st.MakePod().Name("test-newer-pod").Namespace("ns1").UID("tp002").CreationTimestamp(metav1.Now()).Priority(highPriority).NominatedNodeName("node1").Obj()
	q.Add(ctx, newerPod)

	// And then unschedulablePodInfo was determined as unschedulable AGAIN.
	podutil.UpdatePodCondition(&unschedulablePod.Status, &v1.PodCondition{
		Type:    v1.PodScheduled,
		Status:  v1.ConditionFalse,
		Reason:  v1.PodReasonUnschedulable,
		Message: "fake scheduling failure",
	})

	// And then, put unschedulable pod to the unschedulable queue
	err = q.AddUnschedulablePodIfNotPresent(logger, newQueuedPodInfoForLookup(unschedulablePod, "plugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	// Move clock to make the unschedulable pods complete backoff.
	c.Step(DefaultPodInitialBackoffDuration + time.Second)
	// Move all unschedulable pods to the active queue.
	q.MoveAllToActiveOrBackoffQueue(logger, framework.EventUnschedulableTimeout, nil, nil, nil)

	// At this time, newerPod should be popped
	// because it is the oldest tried pod.
	item2, err2 := q.Pop(logger)
	if err2 != nil {
		t.Errorf("Error while popping the head of the queue: %v", err2)
	} else {
		p2 := item2.(*framework.QueuedPodInfo)
		if diff := cmp.Diff(newerPod, p2.Pod); diff != "" {
			t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
		}
	}
}

// TestHighPriorityBackoff tests that a high priority pod does not block
// other pods if it is unschedulable
func TestHighPriorityBackoff(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueue(ctx, newDefaultQueueSort())

	midPod := st.MakePod().Name("test-midpod").Namespace("ns1").UID("tp-mid").Priority(midPriority).NominatedNodeName("node1").Obj()
	highPod := st.MakePod().Name("test-highpod").Namespace("ns1").UID("tp-high").Priority(highPriority).NominatedNodeName("node1").Obj()
	q.Add(ctx, midPod)
	q.Add(ctx, highPod)
	// Simulate a pod being popped by the scheduler, determined unschedulable, and
	// then moved back to the active queue.
	entity, err := q.Pop(logger)
	if err != nil {
		t.Errorf("Error while popping the head of the queue: %v", err)
	}
	p := entity.(*framework.QueuedPodInfo)
	if diff := cmp.Diff(highPod, p.Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	// Update pod condition to unschedulable.
	podutil.UpdatePodCondition(&p.Pod.Status, &v1.PodCondition{
		Type:    v1.PodScheduled,
		Status:  v1.ConditionFalse,
		Reason:  v1.PodReasonUnschedulable,
		Message: "fake scheduling failure",
	})
	// Put in the unschedulable queue.
	err = q.AddUnschedulablePodIfNotPresent(logger, newQueuedPodInfoForLookup(p.Pod, "fooPlugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	// Move all unschedulable pods to the active/backoff queue.
	q.MoveAllToActiveOrBackoffQueue(logger, framework.EventUnschedulableTimeout, nil, nil, nil)

	entity, err = q.Pop(logger)
	p = entity.(*framework.QueuedPodInfo)
	if err != nil {
		t.Errorf("Error while popping the head of the queue: %v", err)
	} else if diff := cmp.Diff(midPod, p.Pod); diff != "" {
		// high pod should be in backoffQ
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
}

// TestHighPriorityFlushUnschedulableEntitiesLeftover tests that entities will be moved to
// activeQ after one minutes if it is in unschedulableEntities.
func TestHighPriorityFlushUnschedulableEntitiesLeftover(t *testing.T) {
	c := testingclock.NewFakeClock(time.Now())
	m := makeEmptyQueueingHintMapPerProfile()
	m[""][nodeAdd] = []*QueueingHintFunction{
		{
			PluginName:     "fakePlugin",
			QueueingHintFn: queueHintReturnQueue,
		},
	}
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(c), WithQueueingHintMapPerProfile(m))
	midPod := st.MakePod().Name("test-midpod").Namespace("ns1").UID("tp-mid").Priority(midPriority).NominatedNodeName("node1").Obj()
	highPod := st.MakePod().Name("test-highpod").Namespace("ns1").UID("tp-high").Priority(highPriority).NominatedNodeName("node1").Obj()

	// Update pod condition to highPod.
	podutil.UpdatePodCondition(&highPod.Status, &v1.PodCondition{
		Type:    v1.PodScheduled,
		Status:  v1.ConditionFalse,
		Reason:  v1.PodReasonUnschedulable,
		Message: "fake scheduling failure",
	})

	// Update pod condition to midPod.
	podutil.UpdatePodCondition(&midPod.Status, &v1.PodCondition{
		Type:    v1.PodScheduled,
		Status:  v1.ConditionFalse,
		Reason:  v1.PodReasonUnschedulable,
		Message: "fake scheduling failure",
	})

	// To simulate the pod is failed in scheduling in the real world, Pop() the pod from activeQ before AddUnschedulablePodIfNotPresent()s below.
	q.Add(ctx, highPod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(highPod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	q.Add(ctx, midPod)
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(midPod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	err := q.AddUnschedulablePodIfNotPresent(logger, q.newQueuedPodInfo(ctx, highPod, "fakePlugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	err = q.AddUnschedulablePodIfNotPresent(logger, q.newQueuedPodInfo(ctx, midPod, "fakePlugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}
	c.Step(DefaultPodMaxInUnschedulablePodsDuration + time.Second)
	q.flushUnschedulableEntitiesLeftover(logger)

	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(highPod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
	if p, err := q.Pop(logger); err != nil {
		t.Errorf("Pop failed: %v", err)
	} else if diff := cmp.Diff(midPod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
		t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
	}
}

// TestFlushUnschedulableEntitiesLeftoverSetsFlag verifies that the WasFlushedFromUnschedulable
// flag is correctly set when entities are flushed and cleared when they return to the queue.
func TestFlushUnschedulableEntitiesLeftoverSetsFlag(t *testing.T) {
	c := testingclock.NewFakeClock(time.Now())
	m := makeEmptyQueueingHintMapPerProfile()
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(c), WithQueueingHintMapPerProfile(m))

	pod := st.MakePod().Name("test-pod").Namespace("ns1").UID("tp-1").Priority(midPriority).NominatedNodeName("node1").Obj()

	// Add pod to activeQ and pop it to simulate a scheduling attempt
	q.Add(ctx, pod)
	entity, err := q.Pop(logger)
	if err != nil {
		t.Fatalf("Unexpected error from Pop: %v", err)
	}
	pInfo := entity.(*framework.QueuedPodInfo)

	// Verify flag is initially false
	if pInfo.WasFlushedFromUnschedulable {
		t.Errorf("Expected WasFlushedFromUnschedulable to be false initially, but got true")
	}

	// Add pod to unschedulableEntities (simulating failed scheduling)
	err = q.AddUnschedulablePodIfNotPresent(logger, q.newQueuedPodInfo(ctx, pod, "fakePlugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("Unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}

	// Advance time past the flush duration and flush
	c.Step(DefaultPodMaxInUnschedulablePodsDuration + time.Second)
	q.flushUnschedulableEntitiesLeftover(logger)

	// Pop the pod and verify flag is now true
	entity, err = q.Pop(logger)
	pInfo = entity.(*framework.QueuedPodInfo)
	if err != nil {
		t.Fatalf("Unexpected error from Pop after flush: %v", err)
	}
	if !pInfo.WasFlushedFromUnschedulable {
		t.Errorf("Expected WasFlushedFromUnschedulable to be true after flush, but got false")
	}

	// Simulate pod failing to schedule again and returning to queue
	err = q.AddUnschedulablePodIfNotPresent(logger, q.newQueuedPodInfo(ctx, pInfo.Pod, "fakePlugin"), q.SchedulingCycle())
	if err != nil {
		t.Fatalf("Unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
	}

	// Verify flag is cleared when pod returns to queue
	internalPInfo := q.unschedulableEntities.get(newQueuedPodInfoForLookup(pod))
	if internalPInfo == nil {
		t.Fatalf("pod should be in unschedulableEntities")
	}
	if internalPInfo.(*framework.QueuedPodInfo).WasFlushedFromUnschedulable {
		t.Errorf("Expected WasFlushedFromUnschedulable to be cleared (false) after returning to queue, but got true")
	}
}

func TestFlushUnschedulablePodsLeftoverSetsFlag_GatedPod(t *testing.T) {
	tests := []struct {
		name                            string
		gatedBeforeFlush                bool
		gatedAfterFlush                 bool
		backingOff                      bool
		wantWasFlushedFromUnschedulable bool
		wantQ                           string
	}{
		{
			name:                            "flag is set when pod is not gated",
			wantWasFlushedFromUnschedulable: true,
			wantQ:                           activeQ,
		},
		{
			name:                            "flag is set when pod is no longer gated",
			gatedBeforeFlush:                true,
			wantWasFlushedFromUnschedulable: true,
			wantQ:                           activeQ,
		},
		{
			name:                            "flag is unset when pod is newly gated",
			gatedAfterFlush:                 true,
			wantWasFlushedFromUnschedulable: false,
			wantQ:                           unschedulableQ,
		},
		{
			name:                            "flag is unset when pod is still gated",
			gatedBeforeFlush:                true,
			gatedAfterFlush:                 true,
			wantWasFlushedFromUnschedulable: false,
			wantQ:                           unschedulableQ,
		},
		{
			name:                            "flag is set when pod is not gated and backoff is not complete",
			backingOff:                      true,
			wantWasFlushedFromUnschedulable: true,
			wantQ:                           backoffQ,
		},
		{
			name:                            "flag is set when pod is no longer gated and backoff is not complete",
			gatedBeforeFlush:                true,
			backingOff:                      true,
			wantWasFlushedFromUnschedulable: true,
			wantQ:                           backoffQ,
		},
		{
			name:                            "flag is unset when pod is newly gated and backoff is not complete",
			gatedAfterFlush:                 true,
			backingOff:                      true,
			wantWasFlushedFromUnschedulable: false,
			wantQ:                           unschedulableQ,
		},
		{
			name:                            "flag is unset when pod is still gated and backoff is not complete",
			gatedBeforeFlush:                true,
			gatedAfterFlush:                 true,
			backingOff:                      true,
			wantWasFlushedFromUnschedulable: false,
			wantQ:                           unschedulableQ,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const allowedLabel = "allow"
			const preEnqueuePluginName = "preEnqueuePlugin"

			var backoffDuration time.Duration
			if tt.backingOff {
				backoffDuration = DefaultPodMaxInUnschedulablePodsDuration + time.Minute
			}

			c := testingclock.NewFakeClock(time.Now())
			m := makeEmptyQueueingHintMapPerProfile()
			preEnqM := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					preEnqueuePluginName: &preEnqueuePlugin{allowlists: []string{allowedLabel}},
				},
			}
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(c), WithQueueingHintMapPerProfile(m), WithPreEnqueuePluginMap(preEnqM),
				WithPodInitialBackoffDuration(backoffDuration), WithPodMaxBackoffDuration(backoffDuration))

			pod := st.MakePod().Name("pod1").Namespace("ns1").UID("1").Label(allowedLabel, "").Obj()
			podInfo := &framework.QueuedPodInfo{PodInfo: mustNewPodInfo(pod), QueueingParams: framework.QueueingParams{UnschedulablePlugins: sets.New("foo")}}
			if tt.gatedBeforeFlush {
				podInfo = setQueuedPodInfoGated(podInfo, preEnqueuePluginName, []fwk.ClusterEvent{})
			}

			q.Add(ctx, podInfo.Pod)
			_, err := q.Pop(logger)
			if err != nil {
				t.Fatalf("Failed to pop from active queue: %v", err)
			}

			if tt.gatedAfterFlush {
				delete(pod.Labels, allowedLabel)
			}

			err = q.AddUnschedulablePodIfNotPresent(logger, podInfo, q.SchedulingCycle())
			if err != nil {
				t.Fatalf("Failed to add pod to unschedulable: %v", err)
			}

			// Advance time past the flush duration and flush
			c.Step(DefaultPodMaxInUnschedulablePodsDuration + time.Second)
			q.flushUnschedulableEntitiesLeftover(logger)

			queueSizes := map[string]int{
				unschedulableQ: len(q.UnschedulablePods()),
				backoffQ:       len(q.PodsInBackoffQ()),
				activeQ:        len(q.PodsInActiveQ()),
			}

			if queueSizes[tt.wantQ] == 0 {
				t.Errorf("Pod not found in %s", tt.wantQ)
			}
			actualPod, ok := q.GetPod(podInfo.Pod.Name, podInfo.Pod.Namespace, nil)
			if !ok {
				t.Fatalf("Pod not found in scheduling queue")
			}
			if actualPod.WasFlushedFromUnschedulable != tt.wantWasFlushedFromUnschedulable {
				t.Errorf("Unexpected WasFlushedFromUnschedulable value: %v", actualPod.WasFlushedFromUnschedulable)
			}
		})
	}
}

// TestGatedPodFlushFrequency verifies that a gated pod is only flushed once every
// podMaxInUnschedulablePodsDuration, and not on every periodic flush execution.
func TestGatedPodFlushFrequency(t *testing.T) {
	gatedPod := mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj())

	tests := []struct {
		name       string
		entityInfo framework.QueuedEntityInfo
	}{
		{
			name: "queued pod",
			entityInfo: &framework.QueuedPodInfo{
				PodInfo:        gatedPod,
				QueueingParams: framework.QueueingParams{UnschedulablePlugins: sets.New("foo")},
			},
		},
		{
			name: "queued pod group",
			entityInfo: &framework.QueuedPodGroupInfo{
				PodGroupInfo:   &framework.PodGroupInfo{Namespace: gatedPod.GetNamespace(), Name: "pg", UnscheduledPods: []*v1.Pod{gatedPod.Pod}},
				QueuedPodInfos: []*framework.QueuedPodInfo{{PodInfo: gatedPod, QueueingParams: framework.QueueingParams{UnschedulablePlugins: sets.New("foo")}}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := testingclock.NewFakeClock(time.Now())
			m := makeEmptyQueueingHintMapPerProfile()
			preEnqueuePluginName := "preEnqueuePlugin"
			preEnqM := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					preEnqueuePluginName: &preEnqueuePlugin{}, // Empty allowlist, gates all pods
				},
			}
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			// Use a user-defined 5-minute duration for clarity
			flushDuration := 5 * time.Minute
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(c), WithQueueingHintMapPerProfile(m), WithPreEnqueuePluginMap(preEnqM),
				WithPodMaxInUnschedulablePodsDuration(flushDuration))

			getEntityFromAnyQueue := func() framework.QueuedEntityInfo {
				var entity framework.QueuedEntityInfo
				q.activeQ.underRLock(func(unlockedActiveQ unlockedActiveQueueReader) {
					entity = q.getEntityFromAnyQueue(unlockedActiveQ, tt.entityInfo)
				})
				if entity == nil {
					t.Fatalf("Failed to find entity in any queue")
				}
				return entity
			}

			// Add gated pod directly to unschedulableEntities
			q.unschedulableEntities.addOrUpdate(tt.entityInfo, false, "test-setup")

			// Step clock past the flush duration and trigger flush
			// T=5:01
			c.Step(flushDuration + time.Second)

			q.flushUnschedulableEntitiesLeftover(logger)

			actualEntity := getEntityFromAnyQueue()
			// Verify that flush happened
			firstFlushTime := actualEntity.GetFlushTimestamp()
			if firstFlushTime.IsZero() {
				t.Errorf("Expected FlushTimestamp to be set after the first leftover flush")
			}

			// Step clock by less than the flush duration and trigger flush
			// T=9:01
			c.Step(4 * time.Minute)
			q.flushUnschedulableEntitiesLeftover(logger)

			actualEntity = getEntityFromAnyQueue()
			// Verify that flush didn't happen
			if actualEntity.GetFlushTimestamp() != firstFlushTime {
				t.Errorf("Expected FlushTimestamp to remain %v, but was updated to %v (pod was flushed prematurely)", firstFlushTime, actualEntity.GetFlushTimestamp())
			}

			// Step clock past the duration since the last flush
			// T=10:02
			c.Step(time.Minute + time.Second)
			q.flushUnschedulableEntitiesLeftover(logger)

			actualEntity = getEntityFromAnyQueue()
			// Verify that flush happened
			if !actualEntity.GetFlushTimestamp().After(firstFlushTime) {
				t.Errorf("Expected FlushTimestamp to be updated to a newer time after 5 minutes elapsed, but remained %v", actualEntity.GetFlushTimestamp())
			}
		})
	}
}

// TestAddAttemptedPodGroupIfNeeded verifies that AddAttemptedPodGroupIfNeeded
// correctly handles pod groups with or without failed plugins.
func TestAddAttemptedPodGroupIfNeeded(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)

	pgName := "test-pg"
	pod1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Priority(midPriority).PodGroupName(pgName).Obj()
	pod2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Priority(midPriority).PodGroupName(pgName).Obj()

	tests := []struct {
		name                            string
		setup                           func(t ktesting.TContext, q *PriorityQueue, pgInfo *framework.QueuedPodGroupInfo)
		disableBackoff                  bool
		blockOnPreEnqueue               bool
		skipAddPodGroup                 bool
		status                          *fwk.Status
		expectedUnschedulableCount      int
		expectedConsecutiveErrorsCount  int
		expectedInActiveQ               bool
		expectedInBackoffQ              bool
		expectedInUnschedulableEntities bool
		expectedInIncomplete            []*v1.Pod
		expectedPodsInGroup             int
		expectPreservedTimestamp        bool
	}{
		{
			name: "No pending pods, pod group is not enqueued",
			setup: func(tCtx ktesting.TContext, q *PriorityQueue, pgInfo *framework.QueuedPodGroupInfo) {
				// No setup needed, pendingPodGroupPods is empty
			},
			expectedPodsInGroup: 1,
		},
		{
			name: "Pods present, pod group scheduling had an error, and pod group not backing off",
			setup: func(tCtx ktesting.TContext, q *PriorityQueue, pgInfo *framework.QueuedPodGroupInfo) {
				pInfo1 := q.newQueuedPodInfo(tCtx, pod1)
				pInfo2 := q.newQueuedPodInfo(tCtx, pod2)
				q.pendingPodGroupPods.add(pInfo1)
				q.pendingPodGroupPods.add(pInfo2)
			},
			status:                         fwk.NewStatus(fwk.Error),
			disableBackoff:                 true,
			expectedConsecutiveErrorsCount: 1,
			expectedInActiveQ:              true,
			expectedPodsInGroup:            2,
		},
		{
			name: "Pods present, pod group scheduling result is unschedulable, and pod group not backing off",
			setup: func(tCtx ktesting.TContext, q *PriorityQueue, pgInfo *framework.QueuedPodGroupInfo) {
				pInfo1 := q.newQueuedPodInfo(tCtx, pod1, "fakePlugin")
				pInfo2 := q.newQueuedPodInfo(tCtx, pod2)
				q.pendingPodGroupPods.add(pInfo1)
				q.pendingPodGroupPods.add(pInfo2)
				pgInfo.ConsecutiveErrorsCount = 5 // Set to non-zero to verify reset
			},
			status:                     fwk.NewStatus(fwk.Unschedulable),
			disableBackoff:             true,
			expectedUnschedulableCount: 1,
			expectedInActiveQ:          true,
			expectedPodsInGroup:        2,
		},
		{
			name: "Pods present, pod group scheduling result is unschedulable, and pod group not backing off, but blocked on PreEnqueue",
			setup: func(tCtx ktesting.TContext, q *PriorityQueue, pgInfo *framework.QueuedPodGroupInfo) {
				pInfo1 := q.newQueuedPodInfo(tCtx, pod1, "fakePlugin")
				pInfo2 := q.newQueuedPodInfo(tCtx, pod2)
				q.pendingPodGroupPods.add(pInfo1)
				q.pendingPodGroupPods.add(pInfo2)
			},
			status:                          fwk.NewStatus(fwk.Unschedulable),
			disableBackoff:                  true,
			blockOnPreEnqueue:               true,
			expectedUnschedulableCount:      1,
			expectedInUnschedulableEntities: true,
			expectedPodsInGroup:             2,
		},
		{
			name: "Pods present, pod group scheduling result is Error, and pod group backing off",
			setup: func(tCtx ktesting.TContext, q *PriorityQueue, pgInfo *framework.QueuedPodGroupInfo) {
				pInfo1 := q.newQueuedPodInfo(tCtx, pod1)
				pInfo2 := q.newQueuedPodInfo(tCtx, pod2)
				q.pendingPodGroupPods.add(pInfo1)
				q.pendingPodGroupPods.add(pInfo2)
			},
			status:                         fwk.NewStatus(fwk.Error),
			expectedConsecutiveErrorsCount: 1,
			expectedInBackoffQ:             true,
			expectedPodsInGroup:            2,
		},
		{
			name: "Pods present, pod group scheduling result is Unschedulable, and pod group backing off",
			setup: func(tCtx ktesting.TContext, q *PriorityQueue, pgInfo *framework.QueuedPodGroupInfo) {
				pInfo1 := q.newQueuedPodInfo(tCtx, pod1, "fakePlugin")
				pInfo2 := q.newQueuedPodInfo(tCtx, pod2)
				q.pendingPodGroupPods.add(pInfo1)
				q.pendingPodGroupPods.add(pInfo2)
				pgInfo.ConsecutiveErrorsCount = 5 // Set to non-zero to verify reset
			},
			status:                     fwk.NewStatus(fwk.Unschedulable),
			expectedUnschedulableCount: 1,
			expectedInBackoffQ:         true,
			expectedPodsInGroup:        2,
		},
		{
			name: "Pods present, pod group scheduling result is Error, and pod group backing off, but blocked on PreEnqueue",
			setup: func(tCtx ktesting.TContext, q *PriorityQueue, pgInfo *framework.QueuedPodGroupInfo) {
				pInfo1 := q.newQueuedPodInfo(tCtx, pod1)
				pInfo2 := q.newQueuedPodInfo(tCtx, pod2)
				q.pendingPodGroupPods.add(pInfo1)
				q.pendingPodGroupPods.add(pInfo2)
			},
			status:                          fwk.NewStatus(fwk.Error),
			blockOnPreEnqueue:               true,
			expectedConsecutiveErrorsCount:  1,
			expectedInUnschedulableEntities: true,
			expectedPodsInGroup:             2,
		},
		{
			name: "Pods present, but pod group is not observed, pods move to incompletePodGroupPods",
			setup: func(tCtx ktesting.TContext, q *PriorityQueue, pgInfo *framework.QueuedPodGroupInfo) {
				pInfo1 := q.newQueuedPodInfo(tCtx, pod1)
				pInfo2 := q.newQueuedPodInfo(tCtx, pod2)
				q.pendingPodGroupPods.add(pInfo1)
				q.pendingPodGroupPods.add(pInfo2)
			},
			status:               fwk.NewStatus(fwk.Error),
			skipAddPodGroup:      true,
			expectedInIncomplete: []*v1.Pod{pod1, pod2},
			expectedPodsInGroup:  2,
		},
		{
			name: "Unschedulable pods are present but pod group scheduling was successful, requeue to active queue directly and preserve timestamp",
			setup: func(tCtx ktesting.TContext, q *PriorityQueue, pgInfo *framework.QueuedPodGroupInfo) {
				pInfo1 := q.newQueuedPodInfo(tCtx, pod1, "fakePlugin")
				pInfo2 := q.newQueuedPodInfo(tCtx, pod2)
				q.pendingPodGroupPods.add(pInfo1)
				q.pendingPodGroupPods.add(pInfo2)
			},
			expectedInActiveQ:        true,
			expectedPodsInGroup:      2,
			expectPreservedTimestamp: true,
		},
		{
			name: "New pods only are present and pod group scheduling was successful, requeue to active queue directly with new timestamp",
			setup: func(tCtx ktesting.TContext, q *PriorityQueue, pgInfo *framework.QueuedPodGroupInfo) {
				pInfo2 := q.newQueuedPodInfo(tCtx, pod2)
				q.pendingPodGroupPods.add(pInfo2)
			},
			expectedInActiveQ:        true,
			expectedPodsInGroup:      1,
			expectPreservedTimestamp: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)

			c := testingclock.NewFakeClock(time.Now())
			m := makeEmptyQueueingHintMapPerProfile()
			preEnqueueMap := make(map[string]map[string]fwk.PreEnqueuePlugin)
			if test.blockOnPreEnqueue {
				preEnqueueMap[""] = map[string]fwk.PreEnqueuePlugin{
					"testPlugin": &preEnqueuePlugin{},
				}
			}

			opts := []Option{WithClock(c), WithQueueingHintMapPerProfile(m), WithPreEnqueuePluginMap(preEnqueueMap)}
			if test.disableBackoff {
				opts = append(opts, WithPodInitialBackoffDuration(0), WithPodMaxBackoffDuration(0))
			}
			q := NewTestQueue(tCtx, newDefaultQueueSort(), opts...)
			podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()
			if !test.skipAddPodGroup {
				q.AddPodGroup(tCtx.Logger(), podGroup)
			}

			pgInfo := q.newQueuedPodGroupInfo(q.newQueuedPodInfo(tCtx, pod1), podGroup)
			oldTimestamp := pgInfo.Timestamp

			test.setup(tCtx, q, pgInfo)
			c.Step(time.Second)

			err := q.AddAttemptedPodGroupIfNeeded(tCtx.Logger(), pgInfo, q.SchedulingCycle(), test.status)
			if err != nil {
				tCtx.Fatalf("Unexpected error from AddAttemptedPodGroupIfNeeded: %v", err)
			}

			if test.expectPreservedTimestamp && !pgInfo.Timestamp.Equal(oldTimestamp) {
				tCtx.Errorf("Expected timestamp to be preserved (%v), but got %v", oldTimestamp, pgInfo.Timestamp)
			}
			if !test.expectPreservedTimestamp && (test.expectedInActiveQ || test.expectedInBackoffQ || test.expectedInUnschedulableEntities) && pgInfo.Timestamp != c.Now() {
				tCtx.Errorf("Expected timestamp to be %v, but got %v", c.Now(), pgInfo.Timestamp)
			}
			if pgInfo.UnschedulableCount != test.expectedUnschedulableCount {
				tCtx.Errorf("Expected UnschedulableCount to be %v, got %v", test.expectedUnschedulableCount, pgInfo.UnschedulableCount)
			}
			if pgInfo.ConsecutiveErrorsCount != test.expectedConsecutiveErrorsCount {
				tCtx.Errorf("Expected ConsecutiveErrorsCount to be %v, got %v", test.expectedConsecutiveErrorsCount, pgInfo.ConsecutiveErrorsCount)
			}
			if isInActiveQ := q.activeQ.has(pgInfo); isInActiveQ != test.expectedInActiveQ {
				tCtx.Errorf("Expected pod group to be in activeQ: %v, got %v", test.expectedInActiveQ, isInActiveQ)
			}
			if isInBackoffQ := q.backoffQ.has(pgInfo); isInBackoffQ != test.expectedInBackoffQ {
				tCtx.Errorf("Expected pod group to be in backoffQ: %v, got %v", test.expectedInBackoffQ, isInBackoffQ)
			}
			if isInUnschedulable := q.unschedulableEntities.get(pgInfo) != nil; isInUnschedulable != test.expectedInUnschedulableEntities {
				tCtx.Errorf("Expected pod group to be in unschedulableEntities: %v, got %v", test.expectedInUnschedulableEntities, isInUnschedulable)
			}
			if q.pendingPodGroupPods.len() != 0 {
				tCtx.Errorf("Expected pendingPodGroupPods to be cleared")
			}
			for _, pod := range test.expectedInIncomplete {
				if !q.incompletePodGroupPods.has(pod) {
					tCtx.Errorf("Expected pod %v to be in incompletePodGroupPods", pod.Name)
				}
			}
			if len(pgInfo.QueuedPodInfos) != test.expectedPodsInGroup {
				tCtx.Errorf("Expected QueuedPodInfos to have %v elements, got %v", test.expectedPodsInGroup, len(pgInfo.QueuedPodInfos))
			}
		})
	}
}

func TestPriorityQueue_initPodMaxInUnschedulablePodsDuration(t *testing.T) {
	pod1 := st.MakePod().Name("test-pod-1").Namespace("ns1").UID("tp-1").NominatedNodeName("node1").Obj()
	pod2 := st.MakePod().Name("test-pod-2").Namespace("ns2").UID("tp-2").NominatedNodeName("node2").Obj()

	var timestamp = time.Now()
	pInfo1 := &framework.QueuedPodInfo{
		PodInfo: mustNewTestPodInfo(t, pod1),
		QueueingParams: framework.QueueingParams{
			Timestamp: timestamp.Add(-time.Second),
		},
	}
	pInfo2 := &framework.QueuedPodInfo{
		PodInfo: mustNewTestPodInfo(t, pod2),
		QueueingParams: framework.QueueingParams{
			Timestamp: timestamp.Add(-2 * time.Second),
		},
	}

	tests := []struct {
		name                              string
		podMaxInUnschedulablePodsDuration time.Duration
		operations                        []operation
		operands                          []*framework.QueuedPodInfo
		expected                          []*framework.QueuedPodInfo
	}{
		{
			name: "New priority queue by the default value of podMaxInUnschedulablePodsDuration",
			operations: []operation{
				addPodUnschedulablePods,
				addPodUnschedulablePods,
				flushUnscheduledQ,
			},
			operands: []*framework.QueuedPodInfo{pInfo1, pInfo2, nil},
			expected: []*framework.QueuedPodInfo{pInfo2, pInfo1},
		},
		{
			name:                              "New priority queue by user-defined value of podMaxInUnschedulablePodsDuration",
			podMaxInUnschedulablePodsDuration: 30 * time.Second,
			operations: []operation{
				addPodUnschedulablePods,
				addPodUnschedulablePods,
				flushUnscheduledQ,
			},
			operands: []*framework.QueuedPodInfo{pInfo1, pInfo2, nil},
			expected: []*framework.QueuedPodInfo{pInfo2, pInfo1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pInfo1.FlushTimestamp = time.Time{}
			pInfo2.FlushTimestamp = time.Time{}
			tCtx := ktesting.Init(t)
			logger := klog.FromContext(tCtx)
			var queue *PriorityQueue
			if test.podMaxInUnschedulablePodsDuration > 0 {
				queue = NewTestQueue(tCtx, newDefaultQueueSort(),
					WithClock(testingclock.NewFakeClock(timestamp)),
					WithPodMaxInUnschedulablePodsDuration(test.podMaxInUnschedulablePodsDuration))
			} else {
				queue = NewTestQueue(tCtx, newDefaultQueueSort(),
					WithClock(testingclock.NewFakeClock(timestamp)))
			}

			var podInfoList []*framework.QueuedPodInfo

			for i, op := range test.operations {
				op(tCtx, queue, test.operands[i])
			}

			expectedLen := len(test.expected)
			if queue.activeQ.len() != expectedLen {
				t.Fatalf("Expected %v items to be in activeQ, but got: %v", expectedLen, queue.activeQ.len())
			}

			for i := 0; i < expectedLen; i++ {
				entity, err := queue.activeQ.pop(logger)
				if err != nil {
					t.Errorf("Error while popping the head of the queue: %v", err)
				} else {
					pInfo := entity.(*framework.QueuedPodInfo)
					podInfoList = append(podInfoList, pInfo)
					// Cleanup attempts counter incremented in activeQ.pop()
					pInfo.Attempts = 0
				}
			}

			if diff := cmp.Diff(test.expected, podInfoList, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected QueuedPodInfo list (-want, +got):\n%s", diff)
			}
		})
	}
}

type operation func(tCtx ktesting.TContext, queue *PriorityQueue, pInfo *framework.QueuedPodInfo)

var (
	add = func(tCtx ktesting.TContext, queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		queue.Add(tCtx, pInfo.Pod)
	}
	pop = func(tCtx ktesting.TContext, queue *PriorityQueue, _ *framework.QueuedPodInfo) {
		_, err := queue.Pop(klog.FromContext(tCtx))
		if err != nil {
			tCtx.Fatalf("Unexpected error during Pop: %v", err)
		}
	}
	popAndRequeueAsUnschedulable = func(tCtx ktesting.TContext, queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		// To simulate the pod is failed in scheduling in the real world, Pop() the pod from activeQ before AddUnschedulablePodIfNotPresent() below.
		// UnschedulablePlugins and PendingPlugins will get cleared by Pop, so make a copy first.
		logger := klog.FromContext(tCtx)
		unschedulablePlugins := pInfo.UnschedulablePlugins.Clone()
		pendingPlugins := pInfo.PendingPlugins.Clone()
		queue.Add(tCtx, pInfo.Pod)
		entity, err := queue.Pop(logger)
		p := entity.(*framework.QueuedPodInfo)
		if err != nil {
			tCtx.Fatalf("Unexpected error during Pop: %v", err)
		} else if diff := cmp.Diff(pInfo.Pod, p.Pod); diff != "" {
			tCtx.Fatalf("Unexpected pod after Pop (-want, +got):\n%s", diff)
		}
		// Simulate plugins that are waiting for some events.
		p.UnschedulablePlugins = unschedulablePlugins
		p.PendingPlugins = pendingPlugins
		if err := queue.AddUnschedulablePodIfNotPresent(logger, p, 1); err != nil {
			tCtx.Fatalf("Unexpected error during AddUnschedulablePodIfNotPresent: %v", err)
		}
	}
	popAndRequeueAsBackoff = func(tCtx ktesting.TContext, queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		// To simulate the pod is failed in scheduling in the real world, Pop() the pod from activeQ before AddUnschedulablePodIfNotPresent() below.
		logger := klog.FromContext(tCtx)
		queue.Add(tCtx, pInfo.Pod)
		entity, err := queue.Pop(logger)
		p := entity.(*framework.QueuedPodInfo)
		if err != nil {
			tCtx.Fatalf("Unexpected error during Pop: %v", err)
		} else if diff := cmp.Diff(pInfo.Pod, p.Pod); diff != "" {
			tCtx.Fatalf("Unexpected pod after Pop (-want, +got):\n%s", diff)
		}
		// needs to increment it to make it backoff
		p.UnschedulableCount++
		// When there is no known unschedulable plugin, pods always go to the backoff queue.
		if err := queue.AddUnschedulablePodIfNotPresent(logger, p, 1); err != nil {
			tCtx.Fatalf("Unexpected error during AddUnschedulablePodIfNotPresent: %v", err)
		}
	}
	addPodActiveQ = func(tCtx ktesting.TContext, queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		queue.Add(tCtx, pInfo.Pod)
	}
	addPodActiveQDirectly = func(tCtx ktesting.TContext, queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		queue.activeQ.add(klog.FromContext(tCtx), pInfo, framework.EventUnscheduledPodAdd.Label())
	}
	addPodUnschedulablePods = func(tCtx ktesting.TContext, queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		if !pInfo.Gated() {
			// Update pod condition to unschedulable.
			podutil.UpdatePodCondition(&pInfo.Pod.Status, &v1.PodCondition{
				Type:    v1.PodScheduled,
				Status:  v1.ConditionFalse,
				Reason:  v1.PodReasonUnschedulable,
				Message: "fake scheduling failure",
			})
			// needs to increment it to make it backoff
			pInfo.UnschedulableCount++
		}
		queue.unschedulableEntities.addOrUpdate(pInfo, false, framework.EventUnscheduledPodAdd.Label())
	}
	deletePod = func(tCtx ktesting.TContext, queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		queue.Delete(tCtx.Logger(), pInfo.Pod)
	}
	updatePodQueueable = func(tCtx ktesting.TContext, queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		newPod := pInfo.Pod.DeepCopy()
		newPod.Labels = map[string]string{"queueable": ""}
		queue.Update(tCtx, pInfo.Pod, newPod)
	}
	addPodBackoffQ = func(tCtx ktesting.TContext, queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		queue.backoffQ.add(klog.FromContext(tCtx), pInfo, framework.EventUnscheduledPodAdd.Label())
	}
	moveAllToActiveOrBackoffQ = func(tCtx ktesting.TContext, queue *PriorityQueue, _ *framework.QueuedPodInfo) {
		queue.MoveAllToActiveOrBackoffQueue(klog.FromContext(tCtx), framework.EventUnschedulableTimeout, nil, nil, nil)
	}
	flushBackoffQ = func(tCtx ktesting.TContext, queue *PriorityQueue, _ *framework.QueuedPodInfo) {
		queue.clock.(*testingclock.FakeClock).Step(3 * time.Second)
		queue.flushBackoffQCompleted(klog.FromContext(tCtx))
	}
	moveClockForward = func(tCtx ktesting.TContext, queue *PriorityQueue, _ *framework.QueuedPodInfo) {
		queue.clock.(*testingclock.FakeClock).Step(3 * time.Second)
	}
	flushUnscheduledQ = func(tCtx ktesting.TContext, queue *PriorityQueue, _ *framework.QueuedPodInfo) {
		queue.clock.(*testingclock.FakeClock).Step(queue.podMaxInUnschedulablePodsDuration)
		queue.flushUnschedulableEntitiesLeftover(klog.FromContext(tCtx))
	}
	updatePluginToGateAllPods = func(tCtx ktesting.TContext, queue *PriorityQueue, _ *framework.QueuedPodInfo) {
		queue.preEnqueuePluginMap[""]["preEnqueuePlugin"] = &preEnqueuePlugin{allowlists: []string{""}}
	}
	updatePluginToUngateAllPods = func(tCtx ktesting.TContext, queue *PriorityQueue, _ *framework.QueuedPodInfo) {
		queue.preEnqueuePluginMap[""]["preEnqueuePlugin"] = &preEnqueuePlugin{allowlists: []string{"queueable"}}
	}
)

// TestPodTimestamp tests the operations related to QueuedPodInfo.
func TestPodTimestamp(t *testing.T) {
	pod1 := st.MakePod().Name("test-pod-1").Namespace("ns1").UID("tp-1").NominatedNodeName("node1").Obj()
	pod2 := st.MakePod().Name("test-pod-2").Namespace("ns2").UID("tp-2").NominatedNodeName("node2").Obj()

	var timestamp = time.Now()
	pInfo1 := &framework.QueuedPodInfo{
		PodInfo: mustNewTestPodInfo(t, pod1),
		QueueingParams: framework.QueueingParams{
			Timestamp: timestamp,
		},
	}
	pInfo2 := &framework.QueuedPodInfo{
		PodInfo: mustNewTestPodInfo(t, pod2),
		QueueingParams: framework.QueueingParams{
			Timestamp: timestamp.Add(time.Second),
		},
	}

	tests := []struct {
		name       string
		operations []operation
		operands   []*framework.QueuedPodInfo
		expected   []*framework.QueuedPodInfo
	}{
		{
			name: "add two pod to activeQ and sort them by the timestamp",
			operations: []operation{
				// Need to add the pods directly to the activeQ to override the timestamps.
				addPodActiveQDirectly,
				addPodActiveQDirectly,
			},
			operands: []*framework.QueuedPodInfo{pInfo2, pInfo1},
			expected: []*framework.QueuedPodInfo{pInfo1, pInfo2},
		},
		{
			name: "add two pod to unschedulableEntities then move them to activeQ and sort them by the timestamp",
			operations: []operation{
				addPodUnschedulablePods,
				addPodUnschedulablePods,
				moveClockForward,
				moveAllToActiveOrBackoffQ,
			},
			operands: []*framework.QueuedPodInfo{pInfo2, pInfo1, nil, nil},
			expected: []*framework.QueuedPodInfo{pInfo1, pInfo2},
		},
		{
			name: "add one pod to BackoffQ and move it to activeQ",
			operations: []operation{
				// Need to add the pods directly to activeQ to override the timestamps.
				addPodActiveQDirectly,
				addPodBackoffQ,
				flushBackoffQ,
				moveAllToActiveOrBackoffQ,
			},
			operands: []*framework.QueuedPodInfo{pInfo2, pInfo1, nil, nil},
			expected: []*framework.QueuedPodInfo{pInfo1, pInfo2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			logger := klog.FromContext(tCtx)
			queue := NewTestQueue(tCtx, newDefaultQueueSort(), WithClock(testingclock.NewFakeClock(timestamp)))
			var podInfoList []*framework.QueuedPodInfo

			for i, op := range test.operations {
				op(tCtx, queue, test.operands[i])
			}

			expectedLen := len(test.expected)
			if queue.activeQ.len() != expectedLen {
				t.Fatalf("Expected %v items to be in activeQ, but got: %v", expectedLen, queue.activeQ.len())
			}

			for i := 0; i < expectedLen; i++ {
				entity, err := queue.activeQ.pop(logger)
				if err != nil {
					t.Errorf("Error while popping the head of the queue: %v", err)
				} else {
					pInfo := entity.(*framework.QueuedPodInfo)
					podInfoList = append(podInfoList, pInfo)
					// Cleanup attempts counter incremented in activeQ.pop()
					pInfo.Attempts = 0
				}
			}

			if diff := cmp.Diff(test.expected, podInfoList, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
				t.Errorf("Unexpected QueuedPodInfo list (-want, +got):\n%s", diff)
			}
		})
	}
}

// TestSchedulerPodsMetric tests Prometheus metrics
func TestSchedulerPodsMetric(t *testing.T) {
	timestamp := time.Now()
	preenqueuePluginName := "preEnqueuePlugin"
	metrics.Register()
	total := 60
	queueableNum := 50
	queueable, failme := "queueable", "failme"
	// First 50 Pods are queueable.
	pInfos := makeQueuedPodInfos(queueableNum, "x", queueable, timestamp)
	// The last 10 Pods are not queueable.
	gated := makeQueuedPodInfos(total-queueableNum, "y", failme, timestamp)
	// Manually mark them as gated=true.
	for _, pInfo := range gated {
		setQueuedPodInfoGated(pInfo, preenqueuePluginName, []fwk.ClusterEvent{framework.EventUnscheduledPodUpdate})
	}
	pInfos = append(pInfos, gated...)
	totalWithDelay := 20
	pInfosWithDelay := makeQueuedPodInfos(totalWithDelay, "z", queueable, timestamp.Add(2*time.Second))

	resetPodInfos := func() {
		// reset PodInfo's Attempts because they influence the backoff time calculation.
		for i := range pInfos {
			pInfos[i].Attempts = 0
			pInfos[i].UnschedulableCount = 0
		}
		for i := range pInfosWithDelay {
			pInfosWithDelay[i].Attempts = 0
			pInfosWithDelay[i].UnschedulableCount = 0
		}
	}

	tests := []struct {
		name                       string
		operations                 []operation
		operands                   [][]*framework.QueuedPodInfo
		metricsName                string
		pluginMetricsSamplePercent int
		disablePopFromBackoffQ     bool
		wants                      string
	}{
		{
			name: "add pods to activeQ and unschedulableEntities",
			operations: []operation{
				addPodActiveQ,
				addPodUnschedulablePods,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:30],
				pInfos[30:],
			},
			metricsName: "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 30
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="gated"} 10
scheduler_pending_pods{queue="unschedulable"} 20
`,
		},
		{
			name: "add pods to all kinds of queues",
			operations: []operation{
				addPodActiveQ,
				addPodBackoffQ,
				addPodUnschedulablePods,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:15],
				pInfos[15:40],
				pInfos[40:],
			},
			metricsName: "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 15
scheduler_pending_pods{queue="backoff"} 25
scheduler_pending_pods{queue="gated"} 10
scheduler_pending_pods{queue="unschedulable"} 10
`,
		},
		{
			name: "add pods to unschedulableEntities and then move all to activeQ",
			operations: []operation{
				addPodUnschedulablePods,
				moveClockForward,
				moveAllToActiveOrBackoffQ,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:total],
				{nil},
				{nil},
			},
			metricsName: "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 50
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="gated"} 10
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
		{
			name: "make some pods subject to backoff, add pods to unschedulableEntities, and then move all to activeQ",
			operations: []operation{
				addPodUnschedulablePods,
				moveClockForward,
				addPodUnschedulablePods,
				moveAllToActiveOrBackoffQ,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[20:total],
				{nil},
				pInfosWithDelay[:20],
				{nil},
			},
			metricsName: "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 30
scheduler_pending_pods{queue="backoff"} 20
scheduler_pending_pods{queue="gated"} 10
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
		{
			name: "make some pods subject to backoff, add pods to unschedulableEntities/activeQ, move all to activeQ, and finally flush backoffQ",
			operations: []operation{
				addPodUnschedulablePods,
				addPodActiveQ,
				moveAllToActiveOrBackoffQ,
				flushBackoffQ,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:40],
				pInfos[40:50],
				{nil},
				{nil},
			},
			metricsName: "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 50
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="gated"} 0
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
		{
			name: "add pods to activeQ/unschedulableEntities and then delete some Pods",
			operations: []operation{
				addPodActiveQ,
				addPodUnschedulablePods,
				deletePod,
				deletePod,
				deletePod,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:30],
				pInfos[30:],
				pInfos[:2],
				pInfos[30:33],
				pInfos[50:54],
			},
			metricsName: "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 28
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="gated"} 6
scheduler_pending_pods{queue="unschedulable"} 17
`,
		},
		{
			name: "add pods to activeQ/unschedulableEntities and then update some Pods as queueable",
			operations: []operation{
				addPodActiveQ,
				addPodUnschedulablePods,
				updatePodQueueable,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:30],
				pInfos[30:],
				pInfos[50:55],
			},
			metricsName: "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 35
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="gated"} 5
scheduler_pending_pods{queue="unschedulable"} 20
`,
		},
		{
			name: "the metrics should not be recorded (pluginMetricsSamplePercent=0)",
			operations: []operation{
				add,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:1],
			},
			metricsName:                "scheduler_plugin_execution_duration_seconds",
			pluginMetricsSamplePercent: 0,
			wants: `
# HELP scheduler_plugin_execution_duration_seconds [ALPHA] Duration for running a plugin at a specific extension point.
# TYPE scheduler_plugin_execution_duration_seconds histogram
`, // the observed value will always be 0, because we don't proceed the fake clock.
		},
		{
			name: "the metrics should be recorded (pluginMetricsSamplePercent=100)",
			operations: []operation{
				add,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:1],
			},
			metricsName:                "scheduler_plugin_execution_duration_seconds",
			pluginMetricsSamplePercent: 100,
			wants: `
# HELP scheduler_plugin_execution_duration_seconds [ALPHA] Duration for running a plugin at a specific extension point.
# TYPE scheduler_plugin_execution_duration_seconds histogram
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="1e-05"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="1.5000000000000002e-05"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="2.2500000000000005e-05"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="3.375000000000001e-05"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="5.062500000000001e-05"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="7.593750000000002e-05"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.00011390625000000003"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.00017085937500000006"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.0002562890625000001"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.00038443359375000017"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.0005766503906250003"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.0008649755859375004"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.0012974633789062506"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.0019461950683593758"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.0029192926025390638"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.004378938903808595"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.006568408355712893"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.009852612533569338"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.014778918800354007"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="0.02216837820053101"} 1
scheduler_plugin_execution_duration_seconds_bucket{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success",le="+Inf"} 1
scheduler_plugin_execution_duration_seconds_sum{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success"} 0
scheduler_plugin_execution_duration_seconds_count{extension_point="PreEnqueue",plugin="preEnqueuePlugin",status="Success"} 1
`, // the observed value will always be 0, because we don't proceed the fake clock.
		},
		{
			name: "Gated metric should be 1 when Ungated to Gated transition into moveToActiveQ",
			operations: []operation{
				addPodUnschedulablePods,
				moveClockForward,
				updatePluginToGateAllPods,
				updatePodQueueable,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:1],
				{nil},
				{nil},
				pInfos[:1],
			},
			metricsName:                "scheduler_pending_pods",
			pluginMetricsSamplePercent: 100,
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 0
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="gated"} 1
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
		{
			name: "Gated metric should be 1 when Ungated to Gated transition into moveToBackoffQ",
			operations: []operation{
				addPodUnschedulablePods,
				updatePluginToGateAllPods,
				updatePodQueueable,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:1],
				{nil},
				pInfos[:1],
			},
			metricsName:                "scheduler_pending_pods",
			pluginMetricsSamplePercent: 100,
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 0
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="gated"} 1
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
		{
			name: "Gated metric should be 1 when Ungated to Gated transition when popFromBackoffQ is disabled",
			operations: []operation{
				addPodUnschedulablePods,
				moveClockForward,
				updatePluginToGateAllPods,
				updatePodQueueable,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:1],
				{nil},
				{nil},
				pInfos[:1],
			},
			metricsName:                "scheduler_pending_pods",
			pluginMetricsSamplePercent: 100,
			disablePopFromBackoffQ:     true,
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 0
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="gated"} 1
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
		{
			name: "Gated metric should be 0 when Ungated -> Gated -> Ungated (ActiveQ) transition",
			operations: []operation{
				addPodUnschedulablePods,
				moveClockForward,
				updatePluginToGateAllPods,
				updatePodQueueable,
				updatePluginToUngateAllPods,
				updatePodQueueable,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:1],
				{nil},
				{nil},
				pInfos[:1],
				{nil},
				pInfos[:1],
			},
			pluginMetricsSamplePercent: 100,
			metricsName:                "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 1
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="gated"} 0
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
		{
			name: "Gated metric should be 0 when Ungated -> Gated -> Ungated (BackoffQ) transition",
			operations: []operation{
				addPodUnschedulablePods,
				updatePluginToGateAllPods,
				updatePodQueueable,
				updatePluginToUngateAllPods,
				updatePodQueueable,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:1],
				{nil},
				pInfos[:1],
				{nil},
				pInfos[:1],
			},
			pluginMetricsSamplePercent: 100,
			metricsName:                "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 0
scheduler_pending_pods{queue="backoff"} 1
scheduler_pending_pods{queue="gated"} 0
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
		{
			name: "Gated metric should be 0 when Ungated -> Gated -> Ungated transition, when popFromBackoffQ is disabled",
			operations: []operation{
				addPodUnschedulablePods,
				moveClockForward,
				updatePluginToGateAllPods,
				updatePodQueueable,
				updatePluginToUngateAllPods,
				updatePodQueueable,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:1],
				{nil},
				{nil},
				pInfos[:1],
				{nil},
				pInfos[:1],
			},
			pluginMetricsSamplePercent: 100,
			disablePopFromBackoffQ:     true,
			wants: `
# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 1
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="gated"} 0
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
	}

	resetMetrics := func() {
		metrics.ActivePods().Set(0)
		metrics.BackoffPods().Set(0)
		metrics.UnschedulablePods().Set(0)
		metrics.GatedPods().Set(0)
		metrics.PluginExecutionDuration.Reset()
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			resetMetrics()
			resetPodInfos()

			m := makeEmptyQueueingHintMapPerProfile()
			m[""][framework.EventTargetPodUpdate] = []*QueueingHintFunction{
				{
					PluginName:     preenqueuePluginName,
					QueueingHintFn: queueHintReturnQueue,
				},
			}
			preenq := map[string]map[string]fwk.PreEnqueuePlugin{"": {(&preEnqueuePlugin{}).Name(): &preEnqueuePlugin{allowlists: []string{queueable}}}}
			recorder := metrics.NewMetricsAsyncRecorder(3, 20*time.Microsecond, tCtx.Done())
			queue := NewTestQueue(tCtx, newDefaultQueueSort(), WithClock(testingclock.NewFakeClock(timestamp)), WithPreEnqueuePluginMap(preenq), WithPluginMetricsSamplePercent(test.pluginMetricsSamplePercent), WithMetricsRecorder(recorder), WithQueueingHintMapPerProfile(m))
			queue.isPopFromBackoffQEnabled = !test.disablePopFromBackoffQ
			for i, op := range test.operations {
				for _, pInfo := range test.operands[i] {
					op(tCtx, queue, pInfo)
				}
			}

			recorder.FlushMetrics()

			if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(test.wants), test.metricsName); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// TestPerPodSchedulingMetrics makes sure pod schedule attempts is updated correctly while
// initialAttemptTimestamp stays the same during multiple add/pop operations.
func TestPerPodSchedulingMetrics(t *testing.T) {
	timestamp := time.Now()

	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	tests := []struct {
		name                            string
		perPodSchedulingMetricsScenario func(*testingclock.FakeClock, *PriorityQueue, *v1.Pod)
		wantAttempts                    int
		wantInitialAttemptTs            time.Time
	}{
		{
			// The queue operations are Add -> Pop.
			name: "pod is created and scheduled after 1 attempt",
			perPodSchedulingMetricsScenario: func(c *testingclock.FakeClock, queue *PriorityQueue, pod *v1.Pod) {
				queue.Add(ctx, pod)
			},
			wantAttempts:         1,
			wantInitialAttemptTs: timestamp,
		},
		{
			// The queue operations are Add -> Pop -> AddUnschedulablePodIfNotPresent -> flushUnschedulableEntitiesLeftover -> Pop.
			name: "pod is created and scheduled after 2 attempts",
			perPodSchedulingMetricsScenario: func(c *testingclock.FakeClock, queue *PriorityQueue, pod *v1.Pod) {
				queue.Add(ctx, pod)
				entity, err := queue.Pop(logger)
				pInfo := entity.(*framework.QueuedPodInfo)
				if err != nil {
					t.Fatalf("Failed to pop a pod %v", err)
				}

				pInfo.UnschedulablePlugins = sets.New("plugin")
				err = queue.AddUnschedulablePodIfNotPresent(logger, pInfo, 1)
				if err != nil {
					t.Fatalf("Failed to add unschedulable pod %v", err)
				}
				// Override clock to exceed the DefaultPodMaxInUnschedulablePodsDuration so that unschedulable pods
				// will be moved to activeQ
				c.SetTime(timestamp.Add(DefaultPodMaxInUnschedulablePodsDuration + 1))
				queue.flushUnschedulableEntitiesLeftover(logger)
			},
			wantAttempts:         2,
			wantInitialAttemptTs: timestamp,
		},
		{
			// The queue operations are Add -> Pop -> AddUnschedulablePodIfNotPresent -> flushUnschedulableEntitiesLeftover -> Update -> Pop.
			name: "pod is created and scheduled after 2 attempts but before the second pop, call update",
			perPodSchedulingMetricsScenario: func(c *testingclock.FakeClock, queue *PriorityQueue, pod *v1.Pod) {
				queue.Add(ctx, pod)
				entity, err := queue.Pop(logger)
				pInfo := entity.(*framework.QueuedPodInfo)
				if err != nil {
					t.Fatalf("Failed to pop a pod %v", err)
				}

				pInfo.UnschedulablePlugins = sets.New("plugin")
				err = queue.AddUnschedulablePodIfNotPresent(logger, pInfo, 1)
				if err != nil {
					t.Fatalf("Failed to add unschedulable pod %v", err)
				}
				// Override clock to exceed the DefaultPodMaxInUnschedulablePodsDuration so that unschedulable pods
				// will be moved to activeQ
				updatedTimestamp := timestamp
				c.SetTime(updatedTimestamp.Add(DefaultPodMaxInUnschedulablePodsDuration + 1))
				queue.flushUnschedulableEntitiesLeftover(logger)
				newPod := pod.DeepCopy()
				newPod.Generation = 1
				queue.Update(ctx, pod, newPod)
			},
			wantAttempts:         2,
			wantInitialAttemptTs: timestamp,
		},
		{
			// The queue operations are Add gated pod -> check unschedulableEntities -> lift gate & update pod -> Pop.
			name: "A gated pod is created and scheduled after lifting gate",
			perPodSchedulingMetricsScenario: func(c *testingclock.FakeClock, queue *PriorityQueue, pod *v1.Pod) {
				// Create a queue with PreEnqueuePlugin
				queue.preEnqueuePluginMap = map[string]map[string]fwk.PreEnqueuePlugin{"": {(&preEnqueuePlugin{}).Name(): &preEnqueuePlugin{allowlists: []string{"foo"}}}}
				queue.pluginMetricsSamplePercent = 0
				queue.Add(ctx, pod)
				// Check pod is added to the unschedulableEntities queue.
				if diff := cmp.Diff(pod, getUnschedulablePod(queue, pod)); diff != "" {
					t.Errorf("Unexpected pod in unschedulableEntities (-want, +got):\n%s", diff)
				}
				// Override clock to get different InitialAttemptTimestamp
				c.Step(1 * time.Minute)

				// Update pod with the required label to get it out of unschedulableEntities queue.
				updateGatedPod := pod.DeepCopy()
				updateGatedPod.Labels = map[string]string{"foo": ""}
				queue.Update(ctx, pod, updateGatedPod)
			},
			wantAttempts:         1,
			wantInitialAttemptTs: timestamp.Add(1 * time.Minute),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			c := testingclock.NewFakeClock(timestamp)
			pod := st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").Obj()
			queue := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(c))

			test.perPodSchedulingMetricsScenario(c, queue, pod)
			entity, err := queue.Pop(logger)
			if err != nil {
				t.Fatal(err)
			}
			if entity.GetAttempts() != test.wantAttempts {
				t.Errorf("Pod schedule attempt unexpected, got %v, want %v", entity.GetAttempts(), test.wantAttempts)
			}
			if *entity.GetInitialAttemptTimestamp() != test.wantInitialAttemptTs {
				t.Errorf("Pod initial schedule attempt timestamp unexpected, got %v, want %v", *entity.GetInitialAttemptTimestamp(), test.wantInitialAttemptTs)
			}
		})
	}
}

func TestIncomingPodsMetrics(t *testing.T) {
	timestamp := time.Now()
	unschedulablePlg := "unschedulable_plugin"
	var pInfos = make([]*framework.QueuedPodInfo, 0, 3)

	for i := 1; i <= 3; i++ {
		p := &framework.QueuedPodInfo{
			PodInfo: mustNewTestPodInfo(t,
				st.MakePod().Name(fmt.Sprintf("test-pod-%d", i)).Namespace(fmt.Sprintf("ns%d", i)).UID(fmt.Sprintf("tp-%d", i)).Obj()),
			QueueingParams: framework.QueueingParams{
				Timestamp:            timestamp,
				UnschedulablePlugins: sets.New(unschedulablePlg),
			},
		}
		pInfos = append(pInfos, p)
	}
	tests := []struct {
		name       string
		operations []operation
		want       string
	}{
		{
			name: "add pods to activeQ",
			operations: []operation{
				add,
			},
			want: `
            scheduler_queue_incoming_pods_total{event="UnscheduledPodAdd",queue="active"} 3
`,
		},
		{
			name: "add unscheduled pods then make them unschedulable",
			operations: []operation{
				popAndRequeueAsUnschedulable,
			},
			want: `scheduler_queue_incoming_pods_total{event="UnscheduledPodAdd",queue="active"} 3
             scheduler_queue_incoming_pods_total{event="ScheduleAttemptFailure",queue="unschedulable"} 3
`,
		},
		{
			name: "add unscheduled pods, make them unschedulable, and move them to backoffQ",
			operations: []operation{
				popAndRequeueAsUnschedulable,
				moveAllToActiveOrBackoffQ,
			},
			want: `scheduler_queue_incoming_pods_total{event="UnscheduledPodAdd",queue="active"} 3
			scheduler_queue_incoming_pods_total{event="ScheduleAttemptFailure",queue="unschedulable"} 3
            scheduler_queue_incoming_pods_total{event="UnschedulableTimeout",queue="backoff"} 3
`,
		},
		{
			name: "add unscheduled pods, make them unschedulable, and move them to activeQ",
			operations: []operation{
				popAndRequeueAsUnschedulable,
				moveClockForward,
				moveAllToActiveOrBackoffQ,
			},
			want: `scheduler_queue_incoming_pods_total{event="UnscheduledPodAdd",queue="active"} 3
			scheduler_queue_incoming_pods_total{event="ScheduleAttemptFailure",queue="unschedulable"} 3
            scheduler_queue_incoming_pods_total{event="UnschedulableTimeout",queue="active"} 3
`,
		},
		{
			name: "make some pods subject to backoff and add them to backoffQ, then flush backoffQ",
			operations: []operation{
				popAndRequeueAsBackoff,
				moveClockForward,
				flushBackoffQ,
			},
			want: `scheduler_queue_incoming_pods_total{event="UnscheduledPodAdd",queue="active"} 3
			scheduler_queue_incoming_pods_total{event="BackoffComplete",queue="active"} 3
            scheduler_queue_incoming_pods_total{event="ScheduleAttemptFailure",queue="backoff"} 3
`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			metrics.SchedulerQueueIncomingPods.Reset()
			queue := NewTestQueue(tCtx, newDefaultQueueSort(), WithClock(testingclock.NewFakeClock(timestamp)))
			for _, op := range test.operations {
				for _, pInfo := range pInfos {
					op(tCtx, queue, pInfo)
				}
			}
			metricName := metrics.SchedulerSubsystem + "_" + metrics.SchedulerQueueIncomingPods.Name
			if err := testutil.CollectAndCompare(metrics.SchedulerQueueIncomingPods, strings.NewReader(queueMetricMetadata+test.want), metricName); err != nil {
				t.Errorf("unexpected collecting result:\n%s", err)
			}

		})
	}
}

func TestBackOffFlow(t *testing.T) {
	cl := testingclock.NewFakeClock(time.Now())
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	steps := []struct {
		wantBackoff time.Duration
	}{
		{wantBackoff: time.Second},
		{wantBackoff: 2 * time.Second},
		{wantBackoff: 4 * time.Second},
		{wantBackoff: 8 * time.Second},
		{wantBackoff: 10 * time.Second},
		{wantBackoff: 10 * time.Second},
		{wantBackoff: 10 * time.Second},
	}
	for _, popFromBackoffQEnabled := range []bool{true, false} {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerPopFromBackoffQ, popFromBackoffQEnabled)

		q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(cl))

		pod := st.MakePod().Name("test-pod").Namespace("test-ns").UID("test-uid").Obj()
		podID := types.NamespacedName{
			Namespace: pod.Namespace,
			Name:      pod.Name,
		}
		q.Add(ctx, pod)

		for i, step := range steps {
			t.Run(fmt.Sprintf("step %d popFromBackoffQEnabled(%v)", i, popFromBackoffQEnabled), func(t *testing.T) {
				timestamp := cl.Now()
				// Simulate schedule attempt.
				entity, err := q.Pop(logger)
				if err != nil {
					t.Fatal(err)
				}
				podInfo := entity.(*framework.QueuedPodInfo)
				if podInfo.GetAttempts() != i+1 {
					t.Errorf("got attempts %d, want %d", podInfo.GetAttempts(), i+1)
				}
				podInfo.GetUnschedulablePlugins().Insert("unsched-plugin")
				err = q.AddUnschedulablePodIfNotPresent(logger, podInfo, int64(i))
				if err != nil {
					t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
				}

				// An event happens.
				q.MoveAllToActiveOrBackoffQueue(logger, framework.EventUnschedulableTimeout, nil, nil, nil)

				if !q.backoffQ.has(podInfo) {
					t.Errorf("pod %v is not in the backoff queue", podID)
				}

				// Check backoff duration.
				deadline := podInfo.BackoffExpiration
				backoff := deadline.Sub(timestamp)
				if popFromBackoffQEnabled {
					// If popFromBackoffQEnabled, the actual backoff can be calculated by rounding up to the ordering window duration.
					backoff = backoff.Truncate(backoffQOrderingWindowDuration) + backoffQOrderingWindowDuration
				}
				if backoff != step.wantBackoff {
					t.Errorf("got backoff %s, want %s", backoff, step.wantBackoff)
				}

				// Simulate routine that continuously flushes the backoff queue.
				cl.Step(backoffQOrderingWindowDuration)
				q.flushBackoffQCompleted(logger)
				// Still in backoff queue after an early flush.
				if !q.backoffQ.has(podInfo) {
					t.Errorf("pod %v is not in the backoff queue", podID)
				}
				// Moved out of the backoff queue after timeout.
				cl.Step(backoff)
				q.flushBackoffQCompleted(logger)
				if q.backoffQ.has(podInfo) {
					t.Errorf("pod %v is still in the backoff queue", podID)
				}
			})
		}
	}
}

func TestMoveAllToActiveOrBackoffQueue_PreEnqueueChecks(t *testing.T) {
	var podInfos []*framework.QueuedPodInfo
	for i := 0; i < 5; i++ {
		pInfo := newQueuedPodInfoForLookup(
			st.MakePod().Name(fmt.Sprintf("p%d", i)).Priority(int32(i)).Obj(),
		)
		podInfos = append(podInfos, pInfo)
	}

	tests := []struct {
		name            string
		preEnqueueCheck PreEnqueueCheck
		podInfos        []*framework.QueuedPodInfo
		event           fwk.ClusterEvent
		want            sets.Set[string]
	}{
		{
			name:     "nil PreEnqueueCheck",
			podInfos: podInfos,
			event:    framework.EventUnschedulableTimeout,
			want:     sets.New("p0", "p1", "p2", "p3", "p4"),
		},
		{
			name:            "move Pods with priority greater than 2",
			podInfos:        podInfos,
			event:           framework.EventUnschedulableTimeout,
			preEnqueueCheck: func(pod *v1.Pod) bool { return *pod.Spec.Priority >= 2 },
			want:            sets.New("p2", "p3", "p4"),
		},
		{
			name:     "move Pods with even priority and greater than 2",
			podInfos: podInfos,
			event:    framework.EventUnschedulableTimeout,
			preEnqueueCheck: func(pod *v1.Pod) bool {
				return *pod.Spec.Priority%2 == 0 && *pod.Spec.Priority >= 2
			},
			want: sets.New("p2", "p4"),
		},
		{
			name:     "move Pods with even and negative priority",
			podInfos: podInfos,
			event:    framework.EventUnschedulableTimeout,
			preEnqueueCheck: func(pod *v1.Pod) bool {
				return *pod.Spec.Priority%2 == 0 && *pod.Spec.Priority < 0
			},
		},
		{
			name:     "preCheck isn't called if the event is not interested by any plugins",
			podInfos: podInfos,
			event:    pvAdd, // No plugin is interested in this event.
			preEnqueueCheck: func(pod *v1.Pod) bool {
				panic("preCheck shouldn't be called")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := testingclock.NewFakeClock(time.Now().Truncate(backoffQOrderingWindowDuration))
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithClock(c))
			for _, podInfo := range tt.podInfos {
				// To simulate the pod is failed in scheduling in the real world, Pop() the pod from activeQ before AddUnschedulablePodIfNotPresent() below.
				q.Add(ctx, podInfo.Pod)
				if p, err := q.Pop(logger); err != nil {
					t.Errorf("Pop failed: %v", err)
				} else if diff := cmp.Diff(podInfo.Pod, p.(*framework.QueuedPodInfo).Pod); diff != "" {
					t.Errorf("Unexpected pod after Pop (-want, +got):\n%s", diff)
				}
				podInfo.UnschedulablePlugins = sets.New("plugin")
				err := q.AddUnschedulablePodIfNotPresent(logger, podInfo, q.activeQ.schedulingCycle())
				if err != nil {
					t.Fatalf("unexpected error from AddUnschedulablePodIfNotPresent: %v", err)
				}
			}
			q.MoveAllToActiveOrBackoffQueue(logger, tt.event, nil, nil, tt.preEnqueueCheck)
			got := sets.New[string]()
			c.Step(2 * q.backoffQ.podMaxBackoffDuration())
			gotPodInfos := q.backoffQ.popAllBackoffCompleted(logger)
			for _, pInfo := range gotPodInfos {
				got.Insert(pInfo.(*framework.QueuedPodInfo).Pod.Name)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Unexpected diff (-want, +got):\n%s", diff)
			}
		})
	}
}

func makeQueuedPodInfos(num int, namePrefix, label string, timestamp time.Time) []*framework.QueuedPodInfo {
	var pInfos = make([]*framework.QueuedPodInfo, 0, num)
	for i := 1; i <= num; i++ {
		p := &framework.QueuedPodInfo{
			PodInfo: mustNewPodInfo(
				st.MakePod().Name(fmt.Sprintf("%v-%d", namePrefix, i)).Namespace(fmt.Sprintf("ns%d", i)).Label(label, "").UID(fmt.Sprintf("tp-%d", i)).Obj()),
			QueueingParams: framework.QueueingParams{
				Timestamp:            timestamp,
				UnschedulablePlugins: sets.New[string](),
			},
		}
		pInfos = append(pInfos, p)
	}
	return pInfos
}

func mustNewTestPodInfo(t *testing.T, pod *v1.Pod) *framework.PodInfo {
	podInfo, err := framework.NewPodInfo(pod)
	if err != nil {
		t.Fatal(err)
	}
	return podInfo
}

func mustNewPodInfo(pod *v1.Pod) *framework.PodInfo {
	podInfo, err := framework.NewPodInfo(pod)
	if err != nil {
		panic(err)
	}
	return podInfo
}

// Test_isPodWorthRequeuing tests isPodWorthRequeuing function.
func Test_isPodWorthRequeuing(t *testing.T) {
	metrics.Register()
	count := 0
	queueHintReturnQueue := func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
		count++
		return fwk.Queue, nil
	}
	queueHintReturnSkip := func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
		count++
		return fwk.QueueSkip, nil
	}
	queueHintReturnErr := func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
		count++
		return fwk.QueueSkip, fmt.Errorf("unexpected error")
	}

	tests := []struct {
		name                   string
		podInfo                *framework.QueuedPodInfo
		event                  fwk.ClusterEvent
		oldObj                 interface{}
		newObj                 interface{}
		expected               queueingStrategy
		expectedExecutionCount int // expected total execution count of queueing hint function
		queueingHintMap        QueueingHintMapPerProfile
	}{
		{
			name: "return Queue when no queueing hint function is registered for the event",
			podInfo: &framework.QueuedPodInfo{
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("fooPlugin1"),
				},
				PodInfo: mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()),
			},
			event:                  nodeAdd,
			oldObj:                 nil,
			newObj:                 st.MakeNode().Obj(),
			expected:               queueSkip,
			expectedExecutionCount: 0,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					// no queueing hint function for NodeAdd.
					framework.EventAssignedPodAdd: {
						{
							// It will be ignored because the event is not NodeAdd.
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name: "Treat the event as Queue when QueueHintFn returns error",
			podInfo: &framework.QueuedPodInfo{
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("fooPlugin1"),
				},
				PodInfo: mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()),
			},
			event:                  nodeAdd,
			oldObj:                 nil,
			newObj:                 st.MakeNode().Obj(),
			expected:               queueAfterBackoff,
			expectedExecutionCount: 1,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					nodeAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnErr,
						},
					},
				},
			},
		},
		{
			name: "return Queue when the event is wildcard",
			podInfo: &framework.QueuedPodInfo{
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("fooPlugin1"),
				},
				PodInfo: mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()),
			},
			event:                  framework.EventUnschedulableTimeout,
			oldObj:                 nil,
			newObj:                 nil,
			expected:               queueAfterBackoff,
			expectedExecutionCount: 0,
			queueingHintMap:        QueueingHintMapPerProfile{},
		},
		{
			name: "return Queue when the event is wildcard and the wildcard targets the pod to be requeued right now",
			podInfo: &framework.QueuedPodInfo{
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("fooPlugin1"),
				},
				PodInfo: mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()),
			},
			event:                  framework.EventForceActivate,
			oldObj:                 nil,
			newObj:                 st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj(),
			expected:               queueAfterBackoff,
			expectedExecutionCount: 0,
			queueingHintMap:        QueueingHintMapPerProfile{},
		},
		{
			name: "return Skip when the event is wildcard, but the wildcard targets a different pod",
			podInfo: &framework.QueuedPodInfo{
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("fooPlugin1"),
				},
				PodInfo: mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()),
			},
			event:                  framework.EventForceActivate,
			oldObj:                 nil,
			newObj:                 st.MakePod().Name("pod-different").Namespace("ns2").UID("2").Obj(),
			expected:               queueSkip,
			expectedExecutionCount: 0,
			queueingHintMap:        QueueingHintMapPerProfile{},
		},
		{
			name: "interprets Queue from the Pending plugin as queueImmediately",
			podInfo: &framework.QueuedPodInfo{
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("fooPlugin1", "fooPlugin3"),
					PendingPlugins:       sets.New("fooPlugin2"),
				},
				PodInfo: mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()),
			},
			event:                  nodeAdd,
			oldObj:                 nil,
			newObj:                 st.MakeNode().Node,
			expected:               queueImmediately,
			expectedExecutionCount: 2,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					nodeAdd: {
						{
							PluginName: "fooPlugin1",
							// It returns Queue and it's interpreted as queueAfterBackoff.
							// But, the function continues to run other hints because the Pod has PendingPlugins, which can result in queueImmediately.
							QueueingHintFn: queueHintReturnQueue,
						},
						{
							PluginName: "fooPlugin2",
							// It's interpreted as queueImmediately.
							// The function doesn't run other hints because queueImmediately is the highest priority.
							QueueingHintFn: queueHintReturnQueue,
						},
						{
							PluginName:     "fooPlugin3",
							QueueingHintFn: queueHintReturnQueue,
						},
						{
							PluginName:     "fooPlugin4",
							QueueingHintFn: queueHintReturnErr,
						},
					},
				},
			},
		},
		{
			name: "interprets Queue from the Unschedulable plugin as queueAfterBackoff",
			podInfo: &framework.QueuedPodInfo{
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("fooPlugin1", "fooPlugin2"),
				},
				PodInfo: mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()),
			},
			event:                  nodeAdd,
			oldObj:                 nil,
			newObj:                 st.MakeNode().Obj(),
			expected:               queueAfterBackoff,
			expectedExecutionCount: 2,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					nodeAdd: {
						{
							// Skip will be ignored
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnSkip,
						},
						{
							// Skip will be ignored
							PluginName:     "fooPlugin2",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name: "Queueing hint function that isn't from the plugin in UnschedulablePlugins/PendingPlugins is ignored",
			podInfo: &framework.QueuedPodInfo{
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("fooPlugin1", "fooPlugin2"),
				},
				PodInfo: mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()),
			},
			event:                  nodeAdd,
			oldObj:                 nil,
			newObj:                 st.MakeNode().Node,
			expected:               queueSkip,
			expectedExecutionCount: 2,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					nodeAdd: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnSkip,
						},
						{
							PluginName:     "fooPlugin2",
							QueueingHintFn: queueHintReturnSkip,
						},
						{
							PluginName:     "fooPlugin3",
							QueueingHintFn: queueHintReturnQueue, // It'll be ignored.
						},
					},
				},
			},
		},
		{
			name: "If event is specific Node update event, queueing hint function for NodeUpdate/UpdateNodeLabel is also executed",
			podInfo: &framework.QueuedPodInfo{
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("fooPlugin1", "fooPlugin2"),
				},
				PodInfo: mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()),
			},
			event:                  fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.UpdateNodeLabel},
			oldObj:                 nil,
			newObj:                 st.MakeNode().Obj(),
			expected:               queueAfterBackoff,
			expectedExecutionCount: 1,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.UpdateNodeLabel}: {
						{
							PluginName: "fooPlugin1",
							// It's only executed and interpreted as queueAfterBackoff.
							// The function doesn't run other hints because this Pod doesn't have PendingPlugins.
							QueueingHintFn: queueHintReturnQueue,
						},
						{
							PluginName:     "fooPlugin2",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Update}: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
					nodeAdd: { // not executed because NodeAdd is unrelated.
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name: "If event with '*' Resource, queueing hint function for specified Resource is also executed",
			podInfo: &framework.QueuedPodInfo{
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("fooPlugin1"),
				},
				PodInfo: mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()),
			},
			event:                  fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add},
			oldObj:                 nil,
			newObj:                 st.MakeNode().Obj(),
			expected:               queueAfterBackoff,
			expectedExecutionCount: 1,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					fwk.ClusterEvent{Resource: fwk.WildCard, ActionType: fwk.Add}: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
		{
			name: "If event is a wildcard one, queueing hint function for all kinds of events is executed",
			podInfo: &framework.QueuedPodInfo{
				QueueingParams: framework.QueueingParams{
					UnschedulablePlugins: sets.New("fooPlugin1"),
				},
				PodInfo: mustNewPodInfo(st.MakePod().Name("pod1").Namespace("ns1").UID("1").Obj()),
			},
			event:                  fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.UpdateNodeLabel | fwk.UpdateNodeTaint},
			oldObj:                 nil,
			newObj:                 st.MakeNode().Obj(),
			expected:               queueAfterBackoff,
			expectedExecutionCount: 1,
			queueingHintMap: QueueingHintMapPerProfile{
				"": {
					fwk.ClusterEvent{Resource: fwk.WildCard, ActionType: fwk.All}: {
						{
							PluginName:     "fooPlugin1",
							QueueingHintFn: queueHintReturnQueue,
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			count = 0 // reset count every time
			logger, ctx := ktesting.NewTestContext(t)
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithQueueingHintMapPerProfile(test.queueingHintMap))
			actual := q.isPodWorthRequeuing(logger, test.podInfo, test.event, test.oldObj, test.newObj)
			if actual != test.expected {
				t.Errorf("isPodWorthRequeuing() = %v, want %v", actual, test.expected)
			}
			if count != test.expectedExecutionCount {
				t.Errorf("isPodWorthRequeuing() executed queueing hint functions %v times, expected: %v", count, test.expectedExecutionCount)
			}
		})
	}
}

func Test_queuedPodInfo_gatedSetUponCreationAndUnsetUponUpdate(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	plugin, _ := schedulinggates.New(ctx, nil, nil, plfeature.Features{})
	m := map[string]map[string]fwk.PreEnqueuePlugin{"": {names.SchedulingGates: plugin.(fwk.PreEnqueuePlugin)}}
	q := NewTestQueue(ctx, newDefaultQueueSort(), WithPreEnqueuePluginMap(m))

	gatedPod := st.MakePod().SchedulingGates([]string{"hello world"}).Obj()
	q.Add(ctx, gatedPod)

	if !q.unschedulableEntities.get(newQueuedPodInfoForLookup(gatedPod)).Gated() {
		t.Error("Expected pod to be gated")
	}

	ungatedPod := gatedPod.DeepCopy()
	ungatedPod.Spec.SchedulingGates = nil
	q.Update(ctx, gatedPod, ungatedPod)

	ungatedPodInfo, _ := q.Pop(logger)
	if ungatedPodInfo.Gated() {
		t.Error("Expected pod to be ungated")
	}
}

func TestPriorityQueue_GetPod(t *testing.T) {
	activeQPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			Namespace: "default",
		},
	}
	backoffQPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod2",
			Namespace: "default",
		},
	}
	unschedPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod3",
			Namespace: "default",
		},
	}

	logger, ctx := ktesting.NewTestContext(t)
	q := NewTestQueue(ctx, newDefaultQueueSort())
	q.activeQ.add(logger, newQueuedPodInfoForLookup(activeQPod), framework.EventUnscheduledPodAdd.Label())
	q.backoffQ.add(logger, newQueuedPodInfoForLookup(backoffQPod), framework.EventUnscheduledPodAdd.Label())
	q.unschedulableEntities.addOrUpdate(newQueuedPodInfoForLookup(unschedPod), false, framework.EventUnscheduledPodAdd.Label())

	tests := []struct {
		name        string
		podName     string
		namespace   string
		expectedPod *v1.Pod
		expectedOK  bool
	}{
		{
			name:        "pod is found in activeQ",
			podName:     "pod1",
			namespace:   "default",
			expectedPod: activeQPod,
			expectedOK:  true,
		},
		{
			name:        "pod is found in backoffQ",
			podName:     "pod2",
			namespace:   "default",
			expectedPod: backoffQPod,
			expectedOK:  true,
		},
		{
			name:        "pod is found in unschedulableEntities",
			podName:     "pod3",
			namespace:   "default",
			expectedPod: unschedPod,
			expectedOK:  true,
		},
		{
			name:        "pod is not found",
			podName:     "pod4",
			namespace:   "default",
			expectedPod: nil,
			expectedOK:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pInfo, ok := q.GetPod(tt.podName, tt.namespace, nil)
			if ok != tt.expectedOK {
				t.Errorf("Expected ok=%v, but got ok=%v", tt.expectedOK, ok)
			}

			if tt.expectedPod == nil {
				if pInfo == nil {
					return
				}
				t.Fatalf("Expected pod is empty, but got pod=%v", pInfo.Pod)
			}

			if !cmp.Equal(pInfo.Pod, tt.expectedPod) {
				t.Errorf("Expected pod=%v, but got pod=%v", tt.expectedPod, pInfo.Pod)
			}
		})
	}
}

func TestUnschedulablePodsMetric(t *testing.T) {
	type step func(tCtx ktesting.TContext, q *PriorityQueue)

	addPod := func(pInfo *framework.QueuedPodInfo) step {
		return func(tCtx ktesting.TContext, q *PriorityQueue) {
			add(tCtx, q, pInfo)
		}
	}
	deletePod := func(pInfo *framework.QueuedPodInfo) step {
		return func(tCtx ktesting.TContext, q *PriorityQueue) {
			deletePod(tCtx, q, pInfo)
		}
	}
	popPod := func() step {
		return func(tCtx ktesting.TContext, q *PriorityQueue) {
			pop(tCtx, q, nil)
		}
	}
	moveAllToActiveOrBackoffQ := func() step {
		return func(tCtx ktesting.TContext, q *PriorityQueue) {
			moveAllToActiveOrBackoffQ(tCtx, q, nil)
		}
	}
	updatePluginAllowList := func(pluginName string, list []string) step {
		return func(tCtx ktesting.TContext, q *PriorityQueue) {
			q.preEnqueuePluginMap[""][pluginName].(*preEnqueuePlugin).allowlists = list
		}
	}

	pluginName1 := "plugin1"
	pluginName2 := "plugin2"
	queueable := "queueable"
	timestamp := time.Now()
	pod := &framework.QueuedPodInfo{
		PodInfo: mustNewPodInfo(
			st.MakePod().Name("podA").Namespace("namespaceA").Label(queueable, "").UID("someUid").Obj()),
		QueueingParams: framework.QueueingParams{
			Timestamp:            timestamp,
			UnschedulablePlugins: sets.New[string](),
		},
	}

	resetMetrics := func() {
		metrics.UnschedulableReason(pluginName1, "").Set(0)
		metrics.UnschedulableReason(pluginName2, "").Set(0)
	}

	makeGated := func(pInfo *framework.QueuedPodInfo) *framework.QueuedPodInfo {
		return setQueuedPodInfoGated(pInfo.DeepCopy(), pluginName1, []fwk.ClusterEvent{framework.EventUnschedulableTimeout})
	}

	tests := []struct {
		name            string
		steps           []step
		expectedMetrics []int
	}{
		{
			name: "Unschedulable pods metric must be 0 after a pod is gated, ungated, re-queued, and eventually popped from the scheduling queue",
			steps: []step{
				updatePluginAllowList(pluginName1, []string{}),
				addPod(pod),
				moveAllToActiveOrBackoffQ(),
				updatePluginAllowList(pluginName1, []string{queueable}),
				moveAllToActiveOrBackoffQ(),
				popPod(),
			},
			expectedMetrics: []int{0, 0},
		},
		{
			name: "Unschedulable pods metric must be 0 after pod is gated and then deleted",
			steps: []step{
				updatePluginAllowList(pluginName1, []string{}),
				addPod(pod),
				moveAllToActiveOrBackoffQ(),
				deletePod(pod),
			},
			expectedMetrics: []int{0, 0},
		},
		{
			name: "Unschedulable pods metric must be 1 after pod is gated multiple time by the same plugin",
			steps: []step{
				updatePluginAllowList(pluginName1, []string{}),
				addPod(pod),
				moveAllToActiveOrBackoffQ(),
			},
			expectedMetrics: []int{1, 0},
		},
		{
			name: "Unschedulable pods metric must be 0 after non gated pods is added and then deleted",
			steps: []step{
				addPod(pod),
				deletePod(pod),
			},
			expectedMetrics: []int{0, 0},
		},
		{
			name: "Unschedulable pods metric should not be duplicate if gated pods added and then gated with the same plugin again",
			steps: []step{
				updatePluginAllowList(pluginName1, []string{}),
				addPod(makeGated(pod)),
				moveAllToActiveOrBackoffQ(),
			},
			expectedMetrics: []int{1, 0},
		},
		{
			name: "Unschedulable pods metric should be 0 if pod was gated by two plugins sequentially and then ungated and popped",
			steps: []step{
				updatePluginAllowList(pluginName1, []string{}),
				addPod(pod),
				updatePluginAllowList(pluginName1, []string{queueable}),
				updatePluginAllowList(pluginName2, []string{}),
				moveAllToActiveOrBackoffQ(),
				updatePluginAllowList(pluginName2, []string{queueable}),
				moveAllToActiveOrBackoffQ(),
				popPod(),
			},
			expectedMetrics: []int{0, 0},
		},
		{
			name: "Unschedulable pods metric should be 0 if pod was gated by two plugins sequentially and then deleted",
			steps: []step{
				updatePluginAllowList(pluginName1, []string{}),
				addPod(pod),
				updatePluginAllowList(pluginName1, []string{queueable}),
				updatePluginAllowList(pluginName2, []string{}),
				moveAllToActiveOrBackoffQ(),
				deletePod(pod),
			},
			expectedMetrics: []int{0, 0},
		},
		{
			name: "Unschedulable pods metric should be 1 for both plugins if pod was gated by two plugins sequentially",
			steps: []step{
				updatePluginAllowList(pluginName1, []string{}),
				addPod(pod),
				updatePluginAllowList(pluginName1, []string{queueable}),
				updatePluginAllowList(pluginName2, []string{}),
				moveAllToActiveOrBackoffQ(),
			},
			expectedMetrics: []int{1, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			resetMetrics()

			m := makeEmptyQueueingHintMapPerProfile()
			m[""][framework.EventUnschedulableTimeout] = []*QueueingHintFunction{
				{
					PluginName:     pluginName1,
					QueueingHintFn: queueHintReturnQueue,
				},
				{
					PluginName:     pluginName2,
					QueueingHintFn: queueHintReturnQueue,
				},
			}

			plugin1 := preEnqueuePlugin{name: pluginName1, allowlists: []string{queueable}}
			plugin2 := preEnqueuePlugin{name: pluginName2, allowlists: []string{queueable}}

			preenq := map[string]map[string]fwk.PreEnqueuePlugin{"": {pluginName1: &plugin1, pluginName2: &plugin2}}
			recorder := metrics.NewMetricsAsyncRecorder(3, 20*time.Microsecond, tCtx.Done())
			q := NewTestQueue(tCtx, newDefaultQueueSort(), WithClock(testingclock.NewFakeClock(timestamp)), WithPreEnqueuePluginMap(preenq), WithMetricsRecorder(recorder), WithQueueingHintMapPerProfile(m))

			for _, step := range tt.steps {
				step(tCtx, q)
			}

			for i, pluginName := range []string{pluginName1, pluginName2} {
				val, err := testutil.GetGaugeMetricValue(metrics.UnschedulableReason(pluginName, ""))

				if err != nil {
					t.Errorf("Error while collection metric value:\n%s", err)
				}
				if int(val) != tt.expectedMetrics[i] {
					t.Errorf("Unexpected metric for plugin %s result expected %d, actual %d", pluginName, tt.expectedMetrics[i], int(val))
				}
			}

		})
	}
}

func TestPriorityQueue_signPod(t *testing.T) {
	tests := []struct {
		name              string
		enableFeatureGate bool
		signers           map[string]PodSigner
		pod               *v1.Pod
		expectedSignature fwk.PodSignature
	}{
		{
			name:              "Feature gate disabled",
			enableFeatureGate: false,
			signers: map[string]PodSigner{
				"default-scheduler": func(ctx context.Context, pod *v1.Pod) fwk.PodSignature {
					return fwk.PodSignature("sig-1")
				},
			},
			pod: st.MakePod().Name("pod1").SchedulerName("default-scheduler").Obj(),
		},
		{
			name:              "No signers configured",
			enableFeatureGate: true,
			signers:           nil,
			pod:               st.MakePod().Name("pod1").SchedulerName("default-scheduler").Obj(),
		},
		{
			name:              "Signer not found for scheduler",
			enableFeatureGate: true,
			signers: map[string]PodSigner{
				"default-scheduler": func(ctx context.Context, pod *v1.Pod) fwk.PodSignature {
					return fwk.PodSignature("sig-1")
				},
			},
			pod: st.MakePod().Name("pod1").SchedulerName("custom-scheduler").Obj(),
		},
		{
			name:              "Successful signature computation",
			enableFeatureGate: true,
			signers: map[string]PodSigner{
				"default-scheduler": func(ctx context.Context, pod *v1.Pod) fwk.PodSignature {
					return fwk.PodSignature("sig-1")
				},
			},
			pod:               st.MakePod().Name("pod1").SchedulerName("default-scheduler").Obj(),
			expectedSignature: fwk.PodSignature("sig-1"),
		},
		{
			name:              "Signer returns nil (unsignable pod)",
			enableFeatureGate: true,
			signers: map[string]PodSigner{
				"default-scheduler": func(ctx context.Context, pod *v1.Pod) fwk.PodSignature {
					return nil
				},
			},
			pod: st.MakePod().Name("pod1").SchedulerName("default-scheduler").Obj(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.OpportunisticBatching, tt.enableFeatureGate)

			tCtx := ktesting.Init(t)
			q := NewTestQueue(tCtx, newDefaultQueueSort(), WithPodSigners(tt.signers))

			signature := q.signPod(tCtx, tt.pod)

			if !bytes.Equal(signature, tt.expectedSignature) {
				t.Errorf("Expected signature '%s', got '%s'", string(tt.expectedSignature), string(signature))
			}
		})
	}
}

func TestPriorityQueue_AddComputesSignature(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.OpportunisticBatching, true)

	signers := map[string]PodSigner{
		"default-scheduler": func(ctx context.Context, pod *v1.Pod) fwk.PodSignature {
			if val, ok := pod.Labels["key"]; ok {
				return fwk.PodSignature(fmt.Sprintf("sig-%s", val))
			}
			return fwk.PodSignature("sig-default")
		},
	}

	tCtx := ktesting.Init(t)
	q := NewTestQueue(tCtx, newDefaultQueueSort(), WithPodSigners(signers))

	pod := st.MakePod().Name("pod1").SchedulerName("default-scheduler").Label("key", "value1").Obj()
	q.Add(tCtx, pod)

	pInfo, exists := q.GetPod(pod.Name, pod.Namespace, nil)
	if !exists {
		t.Fatal("Pod not found in queue after Add")
	}
	if !bytes.Equal(pInfo.PodSignature, fwk.PodSignature("sig-value1")) {
		t.Errorf("Expected signature 'sig-value1', got '%s'", string(pInfo.PodSignature))
	}
}

func TestPriorityQueue_UpdateRecomputesSignature(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.OpportunisticBatching, true)

	signers := map[string]PodSigner{
		"default-scheduler": func(ctx context.Context, pod *v1.Pod) fwk.PodSignature {
			if val, ok := pod.Labels["key"]; ok {
				return fwk.PodSignature(fmt.Sprintf("sig-%s", val))
			}
			return fwk.PodSignature("sig-default")
		},
	}

	pod1 := st.MakePod().Name("pod1").SchedulerName("default-scheduler").Label("key", "value1").Obj()
	pod2 := pod1.DeepCopy()
	pod2.Labels["key"] = "value2"

	tests := []struct {
		name        string
		prepareFunc func(tCtx ktesting.TContext, q *PriorityQueue)
	}{
		{
			name: "pod in activeQ",
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) {
				q.Add(tCtx, pod1)
			},
		},
		{
			name: "pod in backoffQ",
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) {
				pInfo := q.newQueuedPodInfo(tCtx, pod1)
				q.backoffQ.add(klog.FromContext(tCtx), pInfo, framework.EventUnscheduledPodAdd.Label())
			},
		},
		{
			name: "pod in unschedulableEntities",
			prepareFunc: func(tCtx ktesting.TContext, q *PriorityQueue) {
				pInfo := q.newQueuedPodInfo(tCtx, pod1)
				q.unschedulableEntities.addOrUpdate(pInfo, false, framework.EventUnscheduledPodAdd.Label())
			},
		},
		{
			name:        "pod not in any queue",
			prepareFunc: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			q := NewTestQueue(tCtx, newDefaultQueueSort(), WithPodSigners(signers))

			if tt.prepareFunc != nil {
				tt.prepareFunc(tCtx, q)
			}

			// Update pod with different label
			q.Update(tCtx, pod1, pod2)

			// Check signature was recomputed
			pInfo, exists := q.GetPod(pod2.Name, pod2.Namespace, nil)
			if !exists {
				t.Fatal("Pod not found in queue after update")
			}
			if !bytes.Equal(pInfo.PodSignature, fwk.PodSignature("sig-value2")) {
				t.Errorf("Expected signature 'sig-value2', got '%s'", string(pInfo.PodSignature))
			}
		})
	}
}

func TestPriorityQueue_MultipleProfiles(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.OpportunisticBatching, true)

	_, ctx := ktesting.NewTestContext(t)

	signers := map[string]PodSigner{
		"scheduler-1": func(ctx context.Context, pod *v1.Pod) fwk.PodSignature {
			return fwk.PodSignature("sig-scheduler-1")
		},
		"scheduler-2": func(ctx context.Context, pod *v1.Pod) fwk.PodSignature {
			return fwk.PodSignature("sig-scheduler-2")
		},
	}

	q := NewTestQueue(ctx, newDefaultQueueSort(), WithPodSigners(signers))

	pod1 := st.MakePod().Name("pod1").SchedulerName("scheduler-1").Obj()
	pod2 := st.MakePod().Name("pod2").SchedulerName("scheduler-2").Obj()
	pod3 := st.MakePod().Name("pod3").SchedulerName("scheduler-3").Obj() // No signer

	q.Add(ctx, pod1)
	q.Add(ctx, pod2)
	q.Add(ctx, pod3)

	pInfo1, _ := q.GetPod(pod1.Name, pod1.Namespace, nil)
	pInfo2, _ := q.GetPod(pod2.Name, pod2.Namespace, nil)
	pInfo3, _ := q.GetPod(pod3.Name, pod3.Namespace, nil)

	if !bytes.Equal(pInfo1.PodSignature, fwk.PodSignature("sig-scheduler-1")) {
		t.Errorf("Pod1: expected 'sig-scheduler-1', got '%s'", string(pInfo1.PodSignature))
	}
	if !bytes.Equal(pInfo2.PodSignature, fwk.PodSignature("sig-scheduler-2")) {
		t.Errorf("Pod2: expected 'sig-scheduler-2', got '%s'", string(pInfo2.PodSignature))
	}
	if pInfo3.PodSignature != nil {
		t.Errorf("Pod3: expected nil signature (no signer), got '%s'", string(pInfo3.PodSignature))
	}

}

func TestConcurrentUpdateAndPop(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

	q := NewTestQueue(ctx, newDefaultQueueSort())

	podName := "test-pod"
	// Create a pod with high priority to ensure it's at the front
	pod := st.MakePod().Name(podName).Namespace("default").UID("uid-1").Priority(100).Obj()
	q.Add(ctx, pod)

	var wg sync.WaitGroup
	start := time.Now()
	testDuration := 3 * time.Second

	// Goroutine 1: Continuously Pop and re-Add
	wg.Go(func() {
		for time.Since(start) < testDuration {
			// Pop blocks if empty, but we verify we don't block forever or panic
			entity, err := q.Pop(logger)
			if err != nil {
				t.Errorf("Unexpected error during Pop: %v", err)
				return
			}
			if entity == nil {
				t.Errorf("Unexpected nil QueuedEntityInfo during Pop")
				return
			}
			pInfo := entity.(*framework.QueuedPodInfo)
			if pInfo.Pod.UID != pod.UID {
				t.Errorf("Expected pod UID %v, got %v", pod.UID, pInfo.Pod.UID)
			}
			// Simulate some work to widen the race window
			time.Sleep(100 * time.Microsecond)
			q.Done(pInfo.Pod.UID)
			// Re-add to queue to keep the cycle going
			q.Add(ctx, pInfo.Pod)
		}
	})

	// Goroutine 2: Continuously Update the pod
	wg.Go(func() {
		iter := 0
		currentPod := pod
		for time.Since(start) < testDuration {
			iter++
			newPod := currentPod.DeepCopy()
			newPod.Annotations = map[string]string{"ver": fmt.Sprintf("%d", iter)}
			// Update is atomic
			q.Update(ctx, currentPod, newPod)
			currentPod = newPod
			time.Sleep(50 * time.Microsecond)
		}
	})

	wg.Wait()
	q.Close()
}

type initialQueueState int

const (
	stateActive initialQueueState = iota
	statePopped
	stateBackoff
	stateUnschedulable
	stateGated
	stateIncomplete
)

func setupInitialPodGroupState(t *testing.T, ctx context.Context, q *PriorityQueue, initialPods []*v1.Pod, initialState initialQueueState, initialPodGroup *schedulingv1alpha3.PodGroup) {
	t.Helper()

	if initialState != stateIncomplete {
		logger := klog.FromContext(ctx)
		q.AddPodGroup(logger, initialPodGroup)
	}

	if len(initialPods) == 0 {
		return
	}
	logger := klog.FromContext(ctx)

	for _, pod := range initialPods {
		q.Add(ctx, pod)
	}

	pgLookup := newQueuedPodGroupInfoForLookup(initialPods[0])
	switch initialState {
	case statePopped:
		if _, err := q.Pop(logger); err != nil {
			t.Fatalf("Unexpected error from Pop: %v", err)
		}
	case stateBackoff:
		entity := q.activeQ.delete(pgLookup)
		if entity != nil {
			if pgInfo, ok := entity.(*framework.QueuedPodGroupInfo); ok {
				pgInfo.UnschedulableCount = 1
				pgInfo.Timestamp = q.clock.Now()
				pgInfo.UnschedulablePlugins = sets.New("fooPlugin")
			} else if pInfo, ok := entity.(*framework.QueuedPodInfo); ok {
				pInfo.UnschedulableCount = 1
				pInfo.Timestamp = q.clock.Now()
				pInfo.UnschedulablePlugins = sets.New("fooPlugin")
			}
			q.backoffQ.add(logger, entity, framework.EventUnscheduledPodAdd.Label())
		}
	case stateUnschedulable:
		entity := q.activeQ.delete(pgLookup)
		if entity != nil {
			q.unschedulableEntities.addOrUpdate(entity, false, framework.EventUnscheduledPodAdd.Label())
		}
	case stateGated:
		entity := q.activeQ.delete(pgLookup)
		if entity != nil {
			entity.SetGatingPlugin("preEnqueuePlugin", []fwk.ClusterEvent{pvAdd})
			q.unschedulableEntities.addOrUpdate(entity, false, framework.EventUnscheduledPodAdd.Label())
		}
	}
}

func TestAddPodGroupMember(t *testing.T) {
	pgName := "pg-test"
	p1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	p2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").PodGroupName(pgName).Obj()
	gatedP1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").PodGroupName(pgName).Obj()
	gatedP2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").PodGroupName(pgName).Obj()

	tests := []struct {
		name                          string
		initialPod                    *v1.Pod
		initialState                  initialQueueState
		incomingPod                   *v1.Pod
		expectedInActiveQ             bool
		expectedInUnschedulable       bool
		expectedInBackoffQ            bool
		expectedInPendingPodGroupPods bool
		expectedInIncomplete          []*v1.Pod
		expectedGated                 bool
		expectedGroupSize             int
	}{
		{
			name:              "first member added (ungated), creates new pod group in activeQ",
			incomingPod:       p1,
			expectedInActiveQ: true,
			expectedGroupSize: 1,
		},
		{
			name:                    "first member added (gated), creates new pod group directly in unschedulableEntities",
			incomingPod:             gatedP1,
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       1,
		},
		{
			name:                          "matching pod group is currently in flight (last popped), pod (ungated) goes to pendingPodGroupPods",
			initialPod:                    p1,
			initialState:                  statePopped,
			incomingPod:                   p2,
			expectedInPendingPodGroupPods: true,
		},
		{
			name:                          "matching pod group is currently in flight (last popped), pod (gated) goes to pendingPodGroupPods",
			initialPod:                    p1,
			initialState:                  statePopped,
			incomingPod:                   gatedP2,
			expectedInPendingPodGroupPods: true,
		},
		{
			name:              "existing pod group in activeQ (ungated addition), stays in activeQ",
			initialPod:        p1,
			initialState:      stateActive,
			incomingPod:       p2,
			expectedInActiveQ: true,
			expectedGroupSize: 2,
		},
		{
			name:                    "existing pod group in activeQ (gated addition), becomes gated and moves to unschedulableEntities",
			initialPod:              p1,
			initialState:            stateActive,
			incomingPod:             gatedP2,
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       2,
		},
		{
			name:               "existing pod group in backoffQ (ungated addition), stays in backoffQ",
			initialPod:         p1,
			initialState:       stateBackoff,
			incomingPod:        p2,
			expectedInBackoffQ: true,
			expectedGroupSize:  2,
		},
		{
			name:                    "existing pod group in backoffQ (gated addition), becomes gated and moves to unschedulableEntities",
			initialPod:              p1,
			initialState:            stateBackoff,
			incomingPod:             gatedP2,
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       2,
		},
		{
			name:                    "existing pod group in unschedulableEntities (ungated addition), stays in unschedulableEntities as ungated",
			initialPod:              p1,
			initialState:            stateUnschedulable,
			incomingPod:             p2,
			expectedInUnschedulable: true,
			expectedGated:           false,
			expectedGroupSize:       2,
		},
		{
			name:                    "existing pod group in unschedulableEntities (gated addition), stays in unschedulableEntities as ungated",
			initialPod:              p1,
			initialState:            stateUnschedulable,
			incomingPod:             gatedP2,
			expectedInUnschedulable: true,
			expectedGated:           false,
			expectedGroupSize:       2,
		},
		{
			name:                    "existing gated pod group in unschedulableEntities (ungated addition), stays in unschedulableEntities as gated",
			initialPod:              gatedP1,
			initialState:            stateGated,
			incomingPod:             p2,
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       2,
		},
		{
			name:                 "first member added before pod group exists, goes to incompletePodGroupPods",
			initialState:         stateIncomplete,
			incomingPod:          p1,
			expectedInIncomplete: []*v1.Pod{p1},
		},
		{
			name:                 "existing pod in incompletePodGroupPods, stays in incompletePodGroupPods",
			initialPod:           p1,
			initialState:         stateIncomplete,
			incomingPod:          p2,
			expectedInIncomplete: []*v1.Pod{p1, p2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			// Configure a PreEnqueue plugin that allows pods with label "allow", but gates others.
			preEnqueueMap := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"preEnqueuePlugin": &preEnqueuePlugin{allowlists: []string{"allow"}},
				},
			}

			q := NewTestQueue(ctx, newDefaultQueueSort(), WithPreEnqueuePluginMap(preEnqueueMap))
			podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()

			var initialPods []*v1.Pod
			if tt.initialPod != nil {
				initialPods = []*v1.Pod{tt.initialPod}
			}
			setupInitialPodGroupState(t, ctx, q, initialPods, tt.initialState, podGroup)

			// Add incoming pod
			q.Add(ctx, tt.incomingPod)

			// Verify conditions
			pgLookup := newQueuedPodGroupInfoForLookup(tt.incomingPod)

			if inActive := q.activeQ.has(pgLookup); inActive != tt.expectedInActiveQ {
				t.Errorf("Expected incoming pod in activeQ: %v, got %v", tt.expectedInActiveQ, inActive)
			}
			var entity framework.QueuedEntityInfo
			if tt.expectedInActiveQ {
				entity, _ = q.activeQ.get(pgLookup)
			}

			if inBackoff := q.backoffQ.has(pgLookup); inBackoff != tt.expectedInBackoffQ {
				t.Errorf("Expected incoming pod in backoffQ: %v, got %v", tt.expectedInBackoffQ, inBackoff)
			}
			if tt.expectedInBackoffQ {
				entity, _ = q.backoffQ.get(pgLookup)
			}

			unschedulableEntity := q.unschedulableEntities.get(pgLookup)
			inUnschedulable := unschedulableEntity != nil
			if inUnschedulable != tt.expectedInUnschedulable {
				t.Errorf("Expected incoming pod in unschedulableEntities: %v, got %v", tt.expectedInUnschedulable, inUnschedulable)
			}
			if tt.expectedInUnschedulable {
				entity = unschedulableEntity
			}

			if entity != nil {
				if isGated := entity.Gated(); isGated != tt.expectedGated {
					t.Errorf("Expected pod group to be gated: %v, got %v", tt.expectedGated, isGated)
				}
				if size := entity.Size(); size != tt.expectedGroupSize {
					t.Errorf("Expected pod group to be of size: %d, got %d", tt.expectedGroupSize, size)
				}

				// Verify effective addition of the incoming pod
				foundMember := false
				for _, pInfo := range entity.(*framework.QueuedPodGroupInfo).QueuedPodInfos {
					if pInfo.Pod.Name == tt.incomingPod.Name {
						foundMember = true
						break
					}
				}
				if !foundMember {
					t.Errorf("Incoming pod %s was not found in the pod group members", tt.incomingPod.Name)
				}
			}

			inPending := q.pendingPodGroupPods.has(tt.incomingPod)
			if inPending != tt.expectedInPendingPodGroupPods {
				t.Errorf("Expected incoming pod in pendingPodGroupPods: %v, got %v", tt.expectedInPendingPodGroupPods, inPending)
			}

			for _, pod := range tt.expectedInIncomplete {
				if !q.incompletePodGroupPods.has(pod) {
					t.Errorf("Expected pod %v in incompletePodGroupPods", pod.Name)
				}
			}
			if q.incompletePodGroupPods.len() != len(tt.expectedInIncomplete) {
				t.Errorf("Expected incompletePodGroupPods size to be %v, got %v", len(tt.expectedInIncomplete), q.incompletePodGroupPods.len())
			}
		})
	}
}

func TestDeletePodGroupMember(t *testing.T) {
	pgName := "pg-test"
	p1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	p2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").PodGroupName(pgName).Obj()
	p3 := st.MakePod().Name("pod3").Namespace("ns1").UID("pod3").Label("allow", "").PodGroupName(pgName).Obj()
	gatedP2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").PodGroupName(pgName).Obj()
	gatedP3 := st.MakePod().Name("pod3").Namespace("ns1").UID("pod3").PodGroupName(pgName).Obj()
	notFoundPod := st.MakePod().Name("pod-not-found").Namespace("ns1").UID("pod-not-found").Label("allow", "").PodGroupName(pgName).Obj()

	tests := []struct {
		name                    string
		initialPods             []*v1.Pod
		initialState            initialQueueState
		pendingPods             []*v1.Pod
		podToDelete             *v1.Pod
		expectedInActiveQ       bool
		expectedInUnschedulable bool
		expectedInBackoffQ      bool
		expectedPodsInPending   int
		expectedInIncomplete    []*v1.Pod
		expectedGated           bool
		expectedGroupSize       int
	}{
		{
			name:                  "delete from pendingPodGroupPods (group in flight)",
			initialPods:           []*v1.Pod{p1},
			initialState:          statePopped,
			pendingPods:           []*v1.Pod{p2},
			podToDelete:           p2,
			expectedPodsInPending: 0,
		},
		{
			name:                  "delete one pending pod where there are more pending pods for a group in flight",
			initialPods:           []*v1.Pod{p1, p2, p3},
			initialState:          statePopped,
			pendingPods:           []*v1.Pod{p2, p3},
			podToDelete:           p2,
			expectedPodsInPending: 1,
		},
		{
			name:              "delete the only pod of a group in activeQ, completely removes group",
			initialPods:       []*v1.Pod{p1},
			initialState:      stateActive,
			podToDelete:       p1,
			expectedInActiveQ: false,
		},
		{
			name:              "delete one pod of a size-2 group in activeQ, decreases size and keeps in activeQ",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateActive,
			podToDelete:       p2,
			expectedInActiveQ: true,
			expectedGroupSize: 1,
		},
		{
			name:               "delete the only pod of a group in backoffQ, completely removes group",
			initialPods:        []*v1.Pod{p1},
			initialState:       stateBackoff,
			podToDelete:        p1,
			expectedInBackoffQ: false,
		},
		{
			name:               "delete one pod of a size-2 group in backoffQ, keeps group in backoffQ",
			initialPods:        []*v1.Pod{p1, p2},
			initialState:       stateBackoff,
			podToDelete:        p2,
			expectedInBackoffQ: true,
			expectedGroupSize:  1,
		},
		{
			name:                    "delete the only pod of a group in unschedulableEntities, completely removes group",
			initialPods:             []*v1.Pod{p1},
			initialState:            stateUnschedulable,
			podToDelete:             p1,
			expectedInUnschedulable: false,
		},
		{
			name:                    "delete one pod of a size-2 group in unschedulableEntities, decreases size and keeps in unschedulableEntities",
			initialPods:             []*v1.Pod{p1, p2},
			initialState:            stateUnschedulable,
			podToDelete:             p2,
			expectedInUnschedulable: true,
			expectedGroupSize:       1,
		},
		{
			name:              "delete a pod that doesn't exist in any queue or pending map",
			initialPods:       []*v1.Pod{p1},
			initialState:      stateActive,
			podToDelete:       notFoundPod,
			expectedInActiveQ: true,
			expectedGroupSize: 1,
		},
		{
			name:                    "delete the only gated pod from a group, ungates the pod group",
			initialPods:             []*v1.Pod{p1, gatedP2},
			initialState:            stateGated,
			podToDelete:             gatedP2,
			expectedInUnschedulable: true,
			expectedGated:           false,
			expectedGroupSize:       1,
		},
		{
			name:                    "delete one gated pod from a group where another gated pod remains, group stays gated",
			initialPods:             []*v1.Pod{p1, gatedP2, gatedP3},
			initialState:            stateGated,
			podToDelete:             gatedP2,
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       2,
		},
		{
			name:                 "delete from incompletePodGroupPods",
			initialPods:          []*v1.Pod{p1},
			initialState:         stateIncomplete,
			podToDelete:          p1,
			expectedInIncomplete: nil,
		},
		{
			name:                 "delete one pod from incompletePodGroupPods with multiple pods",
			initialPods:          []*v1.Pod{p1, p2},
			initialState:         stateIncomplete,
			podToDelete:          p1,
			expectedInIncomplete: []*v1.Pod{p2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			preEnqueueMap := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"preEnqueuePlugin": &preEnqueuePlugin{allowlists: []string{"allow"}},
				},
			}
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithPreEnqueuePluginMap(preEnqueueMap))
			podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()

			setupInitialPodGroupState(t, ctx, q, tt.initialPods, tt.initialState, podGroup)
			// Add pending pods
			for _, pod := range tt.pendingPods {
				q.Add(ctx, pod)
			}

			// Delete the target pod
			q.Delete(logger, tt.podToDelete)

			// Verify conditions
			pgLookup := newQueuedPodGroupInfoForLookup(tt.podToDelete)

			if inActive := q.activeQ.has(pgLookup); inActive != tt.expectedInActiveQ {
				t.Errorf("Expected target pod group in activeQ: %v, got %v", tt.expectedInActiveQ, inActive)
			}
			var entity framework.QueuedEntityInfo
			if tt.expectedInActiveQ {
				entity, _ = q.activeQ.get(pgLookup)
			}

			if inBackoff := q.backoffQ.has(pgLookup); inBackoff != tt.expectedInBackoffQ {
				t.Errorf("Expected target pod group in backoffQ: %v, got %v", tt.expectedInBackoffQ, inBackoff)
			}
			if tt.expectedInBackoffQ {
				entity, _ = q.backoffQ.get(pgLookup)
			}

			unschedulableEntity := q.unschedulableEntities.get(pgLookup)
			inUnschedulable := unschedulableEntity != nil
			if inUnschedulable != tt.expectedInUnschedulable {
				t.Errorf("Expected target pod group in unschedulableEntities: %v, got %v", tt.expectedInUnschedulable, inUnschedulable)
			}
			if tt.expectedInUnschedulable {
				entity = unschedulableEntity
			}

			if entity != nil {
				if size := entity.Size(); size != tt.expectedGroupSize {
					t.Errorf("Expected pod group to be of size: %d, got %d", tt.expectedGroupSize, size)
				}

				// Verify effective removal of the deleted pod
				for _, pInfo := range entity.(*framework.QueuedPodGroupInfo).QueuedPodInfos {
					if pInfo.Pod.Name == tt.podToDelete.Name {
						t.Errorf("Deleted pod %s is still present in the pod group members", tt.podToDelete.Name)
					}
				}
			}

			if pendingLen := q.pendingPodGroupPods.len(); pendingLen != tt.expectedPodsInPending {
				t.Errorf("Expected pod group to have %d pods in pendingPodGroupPods, got %d", tt.expectedPodsInPending, pendingLen)
			}
			if q.pendingPodGroupPods.has(tt.podToDelete) {
				t.Errorf("Deleted pod %s is still present in pendingPodGroupPods map", tt.podToDelete.Name)
			}

			for _, pod := range tt.expectedInIncomplete {
				if !q.incompletePodGroupPods.has(pod) {
					t.Errorf("Expected pod %s in incompletePodGroupPods", pod.Name)
				}
			}
			if q.incompletePodGroupPods.len() != len(tt.expectedInIncomplete) {
				t.Errorf("Expected incompletePodGroupPods size to be %d, got %d", len(tt.expectedInIncomplete), q.incompletePodGroupPods.len())
			}
		})
	}
}

func TestUpdatePodGroupMember(t *testing.T) {
	pgName := "pg-test"
	p1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	p2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").PodGroupName(pgName).Obj()
	updatedP2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").Label("update", "true").PodGroupName(pgName).Obj()
	nominatedP2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").Label("update", "true").NominatedNodeName("node1").PodGroupName(pgName).Obj()
	gatedP1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").PodGroupName(pgName).Obj()
	ungatedP1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	updatedGatedP1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("updated", "true").PodGroupName(pgName).Obj()
	gatedP2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").PodGroupName(pgName).Obj()
	notFoundPod := st.MakePod().Name("pod-not-found").Namespace("ns1").UID("pod-not-found").Label("allow", "").PodGroupName(pgName).Obj()
	updatedNotFoundPod := st.MakePod().Name("pod-not-found").Namespace("ns1").UID("pod-not-found").Label("allow", "").Label("updated", "true").PodGroupName(pgName).Obj()

	tests := []struct {
		name                    string
		initialPods             []*v1.Pod
		initialState            initialQueueState
		pendingPods             []*v1.Pod
		oldPod                  *v1.Pod
		newPod                  *v1.Pod
		expectedInActiveQ       bool
		expectedInUnschedulable bool
		expectedInBackoffQ      bool
		expectedPodsInPending   int
		expectedInIncomplete    *v1.Pod
		expectedGated           bool
		expectedGroupSize       int
		expectedNominatedPods   map[types.UID]string
	}{
		{
			name:              "update pod in activeQ, keeps pod group in activeQ",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateActive,
			oldPod:            p2,
			newPod:            updatedP2,
			expectedInActiveQ: true,
			expectedGroupSize: 2,
		},
		{
			name:               "update pod in backoffQ, keeps pod group in backoffQ",
			initialPods:        []*v1.Pod{p1, p2},
			initialState:       stateBackoff,
			oldPod:             p2,
			newPod:             updatedP2,
			expectedInBackoffQ: true,
			expectedGroupSize:  2,
		},
		{
			name:              "update pod in unschedulableEntities moves it to activeQ (ungated pod group)",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateUnschedulable,
			oldPod:            p2,
			newPod:            updatedP2,
			expectedInActiveQ: true,
			expectedGroupSize: 2,
		},
		{
			name:                  "update pod in pendingPodGroupPods (group in flight), updates pod in pending buffer",
			initialPods:           []*v1.Pod{p1},
			initialState:          statePopped,
			pendingPods:           []*v1.Pod{p2},
			oldPod:                p2,
			newPod:                updatedP2,
			expectedPodsInPending: 1,
		},
		{
			name:              "gated pod update (removal of gate), ungates and moves pod group to activeQ",
			initialPods:       []*v1.Pod{gatedP1},
			initialState:      stateGated,
			oldPod:            gatedP1,
			newPod:            ungatedP1,
			expectedInActiveQ: true,
			expectedGated:     false,
			expectedGroupSize: 1,
		},
		{
			name:                    "gated pod update without removal of gate, keeps pod group in unschedulableEntities as gated",
			initialPods:             []*v1.Pod{gatedP1},
			initialState:            stateGated,
			oldPod:                  gatedP1,
			newPod:                  updatedGatedP1,
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       1,
		},
		{
			name:                    "ungate one pod but another remains gated, keeps pod group in unschedulableEntities as gated",
			initialPods:             []*v1.Pod{gatedP1, gatedP2},
			initialState:            stateGated,
			oldPod:                  gatedP1,
			newPod:                  ungatedP1,
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       2,
		},
		{
			name:              "updating pod that doesn't exist in any queue, creates new pod group in activeQ",
			initialPods:       nil,
			initialState:      stateActive,
			oldPod:            notFoundPod,
			newPod:            updatedNotFoundPod,
			expectedInActiveQ: true,
			expectedGated:     false,
			expectedGroupSize: 1,
		},
		{
			name:              "update pod to nominated in activeQ, keeps pod group in activeQ",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateActive,
			oldPod:            p2,
			newPod:            nominatedP2,
			expectedInActiveQ: true,
			expectedGroupSize: 2,
			expectedNominatedPods: map[types.UID]string{
				"pod2": "node1",
			},
		},
		{
			name:               "update pod to nominated in backoffQ, keeps pod group in backoffQ",
			initialPods:        []*v1.Pod{p1, p2},
			initialState:       stateBackoff,
			oldPod:             p2,
			newPod:             nominatedP2,
			expectedInBackoffQ: true,
			expectedGroupSize:  2,
			expectedNominatedPods: map[types.UID]string{
				"pod2": "node1",
			},
		},
		{
			name:              "update pod to nominated in unschedulableEntities moves it to activeQ (ungated pod group)",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateUnschedulable,
			oldPod:            p2,
			newPod:            nominatedP2,
			expectedInActiveQ: true,
			expectedGroupSize: 2,
			expectedNominatedPods: map[types.UID]string{
				"pod2": "node1",
			},
		},
		{
			name:                  "update pod to nominated in pendingPodGroupPods (group in flight), updates pod in pending buffer",
			initialPods:           []*v1.Pod{p1},
			initialState:          statePopped,
			pendingPods:           []*v1.Pod{p2},
			oldPod:                p2,
			newPod:                nominatedP2,
			expectedPodsInPending: 1,
			expectedNominatedPods: map[types.UID]string{
				"pod2": "node1",
			},
		},
		{
			name:                 "update pod in incompletePodGroupPods",
			initialPods:          []*v1.Pod{p2},
			initialState:         stateIncomplete,
			oldPod:               p2,
			newPod:               updatedP2,
			expectedInIncomplete: updatedP2,
		},
		{
			name:                 "updating incomplete pod that doesn't exist in any queue, adds the pod to incompletePodGroupPods",
			initialPods:          nil,
			initialState:         stateIncomplete,
			oldPod:               notFoundPod,
			newPod:               updatedNotFoundPod,
			expectedInIncomplete: updatedNotFoundPod,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			preEnqueueMap := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"preEnqueuePlugin": &preEnqueuePlugin{allowlists: []string{"allow"}},
				},
			}
			q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{tt.newPod}, WithPreEnqueuePluginMap(preEnqueueMap))
			podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()

			setupInitialPodGroupState(t, ctx, q, tt.initialPods, tt.initialState, podGroup)
			// Add pending pods
			for _, pod := range tt.pendingPods {
				q.Add(ctx, pod)
			}

			// Perform the update
			q.Update(ctx, tt.oldPod, tt.newPod)

			// Verify conditions
			pgLookup := newQueuedPodGroupInfoForLookup(tt.newPod)

			if inActive := q.activeQ.has(pgLookup); inActive != tt.expectedInActiveQ {
				t.Errorf("Expected target pod group in activeQ: %v, got %v", tt.expectedInActiveQ, inActive)
			}
			var entity framework.QueuedEntityInfo
			if tt.expectedInActiveQ {
				entity, _ = q.activeQ.get(pgLookup)
			}

			if inBackoff := q.backoffQ.has(pgLookup); inBackoff != tt.expectedInBackoffQ {
				t.Errorf("Expected target pod group in backoffQ: %v, got %v", tt.expectedInBackoffQ, inBackoff)
			}
			if tt.expectedInBackoffQ {
				entity, _ = q.backoffQ.get(pgLookup)
			}

			unschedulableEntity := q.unschedulableEntities.get(pgLookup)
			inUnschedulable := unschedulableEntity != nil
			if inUnschedulable != tt.expectedInUnschedulable {
				t.Errorf("Expected target pod group in unschedulableEntities: %v, got %v", tt.expectedInUnschedulable, inUnschedulable)
			}
			if tt.expectedInUnschedulable {
				entity = unschedulableEntity
			}

			if entity != nil {
				if isGated := entity.Gated(); isGated != tt.expectedGated {
					t.Errorf("Expected pod group to be gated: %v, got %v", tt.expectedGated, isGated)
				}
				if size := entity.Size(); size != tt.expectedGroupSize {
					t.Errorf("Expected pod group to be of size: %d, got %d", tt.expectedGroupSize, size)
				}

				// Verify effective update of the updated pod
				foundUpdated := false
				for _, pInfo := range entity.(*framework.QueuedPodGroupInfo).QueuedPodInfos {
					if pInfo.Pod.Name == tt.newPod.Name {
						foundUpdated = true
						if diff := cmp.Diff(tt.newPod, pInfo.Pod); diff != "" {
							t.Errorf("Queued member pod differs from newPod (-want +got):\n%s", diff)
						}
						break
					}
				}
				if !foundUpdated {
					t.Errorf("Updated pod %s was not found in the pod group members", tt.newPod.Name)
				}
			}

			if pendingLen := q.pendingPodGroupPods.len(); pendingLen != tt.expectedPodsInPending {
				t.Errorf("Expected pod group to have %d pods in pendingPodGroupPods, got %d", tt.expectedPodsInPending, pendingLen)
			}
			if tt.expectedPodsInPending > 0 {
				pInfo := q.pendingPodGroupPods.get(tt.newPod)
				if diff := cmp.Diff(tt.newPod, pInfo.Pod); diff != "" {
					t.Errorf("Pending member pod differs from newPod (-want +got):\n%s", diff)
				}
			}

			if tt.expectedInIncomplete != nil {
				pInfo := q.incompletePodGroupPods.get(tt.expectedInIncomplete)
				if diff := cmp.Diff(tt.expectedInIncomplete, pInfo.Pod); diff != "" {
					t.Errorf("Unexpected pod in incompletePodGroupPods (-want +got):\n%s", diff)
				}
			} else if incompleteLen := q.incompletePodGroupPods.len(); incompleteLen != 0 {
				t.Errorf("Expected no pods in incompletePodGroupPods, got %v", incompleteLen)
			}

			if diff := cmp.Diff(tt.expectedNominatedPods, q.nominatedPodToNode, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Unexpected nominated pods (-want +got):\n%s", diff)
			}
		})
	}
}

func TestActivatePodGroupMember(t *testing.T) {
	pgName := "pg-test"
	p1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	p2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").PodGroupName(pgName).Obj()
	gatedP1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").PodGroupName(pgName).Obj()
	notFoundPod := st.MakePod().Name("pod-not-found").Namespace("ns1").UID("pod-not-found").Label("allow", "").PodGroupName(pgName).Obj()

	tests := []struct {
		name                       string
		initialPods                []*v1.Pod
		initialState               initialQueueState
		pendingPods                []*v1.Pod
		podToActivate              *v1.Pod
		expectedInActiveQ          bool
		expectedInUnschedulable    bool
		expectedPodsInPending      int
		expectedInIncomplete       []*v1.Pod
		expectedGated              bool
		expectedForceActivateEvent bool
	}{
		{
			name:              "activate pod group in unschedulableEntities (ungated), moves pod group to activeQ",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateUnschedulable,
			podToActivate:     p1,
			expectedInActiveQ: true,
		},
		{
			name:                    "activate pod group in unschedulableEntities (gated), keeps pod group in unschedulableEntities as gated",
			initialPods:             []*v1.Pod{gatedP1},
			initialState:            stateGated,
			podToActivate:           gatedP1,
			expectedInUnschedulable: true,
			expectedGated:           true,
		},
		{
			name:              "activate pod group in backoffQ, moves pod group to activeQ",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateBackoff,
			podToActivate:     p1,
			expectedInActiveQ: true,
		},
		{
			name:                       "activate pod group in flight (popped), pod group remains in flight but tracks event",
			initialPods:                []*v1.Pod{p1, p2},
			initialState:               statePopped,
			podToActivate:              p1,
			expectedForceActivateEvent: true,
		},
		{
			name:              "activating pod that does not exist in any queue or in flight is ignored",
			podToActivate:     notFoundPod,
			expectedInActiveQ: false,
		},
		// {
		// 	name:                           "activate pending pod when pod group is in flight, pods remain pending in pendingPodGroupPods map, but track event",
		// 	initialPods:                    []*v1.Pod{p1},
		// 	initialState:                   statePopped,
		// 	pendingPods:                    []*v1.Pod{p2},
		// 	podToActivate:                  p2,
		// 	expectedForceActivateEvent:     true,
		// 	expectedPodsInPending:          1,
		// },
		{
			name:                 "activate pod in incompletePodGroupPods is a no-op",
			initialPods:          []*v1.Pod{p1, p2},
			initialState:         stateIncomplete,
			podToActivate:        p1,
			expectedInIncomplete: []*v1.Pod{p1, p2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			preEnqueueMap := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"preEnqueuePlugin": &preEnqueuePlugin{allowlists: []string{"allow"}},
				},
			}
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithPreEnqueuePluginMap(preEnqueueMap))
			podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()

			setupInitialPodGroupState(t, ctx, q, tt.initialPods, tt.initialState, podGroup)
			// Add pending pods
			for _, pod := range tt.pendingPods {
				q.Add(ctx, pod)
			}

			// Activate the pod
			q.Activate(logger, map[string]*v1.Pod{string(tt.podToActivate.UID): tt.podToActivate})

			// Verify conditions
			pgLookup := newQueuedPodGroupInfoForLookup(tt.podToActivate)

			if inActive := q.activeQ.has(pgLookup); inActive != tt.expectedInActiveQ {
				t.Errorf("Expected target pod group in activeQ: %v, got %v", tt.expectedInActiveQ, inActive)
			}
			var entity framework.QueuedEntityInfo
			if tt.expectedInActiveQ {
				entity, _ = q.activeQ.get(pgLookup)
			}

			if q.backoffQ.has(pgLookup) {
				t.Errorf("Expected target pod group not to be present in backoffQ")
			}

			unschedulableEntity := q.unschedulableEntities.get(pgLookup)
			inUnschedulable := unschedulableEntity != nil
			if inUnschedulable != tt.expectedInUnschedulable {
				t.Errorf("Expected target pod group in unschedulableEntities: %v, got %v", tt.expectedInUnschedulable, inUnschedulable)
			}
			if tt.expectedInUnschedulable {
				entity = unschedulableEntity
			}

			if entity != nil {
				if isGated := entity.Gated(); isGated != tt.expectedGated {
					t.Errorf("Expected pod group to be gated: %v, got %v", tt.expectedGated, isGated)
				}
			}

			if pendingLen := q.pendingPodGroupPods.len(); pendingLen != tt.expectedPodsInPending {
				t.Errorf("Expected pod group to have %d pods in pendingPodGroupPods, got %d", tt.expectedPodsInPending, pendingLen)
			}
			if tt.expectedPodsInPending > 0 && !q.pendingPodGroupPods.has(tt.podToActivate) {
				t.Errorf("Pod %s was not found in pendingPodGroupPods map", tt.podToActivate.Name)
			}

			for _, pod := range tt.expectedInIncomplete {
				if !q.incompletePodGroupPods.has(pod) {
					t.Errorf("Expected pod %s in incompletePodGroupPods", pod.Name)
				}
			}
			if q.incompletePodGroupPods.len() != len(tt.expectedInIncomplete) {
				t.Errorf("Expected incompletePodGroupPods size to be %d, got %d", len(tt.expectedInIncomplete), q.incompletePodGroupPods.len())
			}

			if tt.expectedForceActivateEvent {
				foundEvent := false
				for _, ev := range q.activeQ.listInFlightEvents() {
					clusterEvent, ok := ev.(*clusterEvent)
					if !ok {
						continue
					}
					if clusterEvent.event.Label() == framework.EventForceActivate.Label() {
						foundEvent = true
						break
					}
				}
				if !foundEvent {
					t.Errorf("Expected ForceActivate in-flight event to be tracked, but it wasn't")
				}
			}
		})
	}
}

func TestMoveAllToActiveOrBackoffQueuePodGroupMember(t *testing.T) {
	pgName := "pg-test"
	p1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	p2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").PodGroupName(pgName).Obj()
	gatedP1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").PodGroupName(pgName).Obj()

	tests := []struct {
		name                    string
		initialPods             []*v1.Pod
		initialState            initialQueueState
		event                   fwk.ClusterEvent
		preCheck                PreEnqueueCheck
		expectedInActiveQ       bool
		expectedInUnschedulable bool
		expectedGated           bool
		expectedGroupSize       int
		expectedInFlightEvent   bool
	}{
		{
			name:              "event of interest moves ungated pod group from unschedulableEntities to activeQ",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateUnschedulable,
			event:             fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add},
			expectedInActiveQ: true,
			expectedGroupSize: 2,
		},
		{
			name:                    "event not of interest keeps pod group in unschedulableEntities",
			initialPods:             []*v1.Pod{p1, p2},
			initialState:            stateUnschedulable,
			event:                   fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Delete},
			expectedInUnschedulable: true,
			expectedGroupSize:       2,
		},
		// {
		// 	name:              "gated pod group matching gating event moves the pod group to activeQ",
		// 	initialPods:       []*v1.Pod{gatedP1, p2},
		// 	initialState:      stateGated,
		// 	event:             fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Add},
		// 	expectedInActiveQ: true,
		// 	expectedGroupSize: 2,
		// },
		{
			name:                    "gated pod group not matching gating event stays in unschedulableEntities as gated",
			initialPods:             []*v1.Pod{gatedP1, p2},
			initialState:            stateGated,
			event:                   fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Delete},
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       2,
		},
		{
			name:                    "gated pod group matching event for ungated pod stays in unschedulableEntities as gated",
			initialPods:             []*v1.Pod{gatedP1, p2},
			initialState:            stateGated,
			event:                   fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add},
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       2,
		},
		{
			name:                  "event received while pod group is in flight is tracked in inFlightEvents",
			initialPods:           []*v1.Pod{p1, p2},
			initialState:          statePopped,
			event:                 fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add},
			expectedInFlightEvent: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			preEnqueueMap := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"preEnqueuePlugin": &preEnqueuePlugin{allowlists: []string{"allow"}},
				},
			}

			m := makeEmptyQueueingHintMapPerProfile()
			m[""] = map[fwk.ClusterEvent][]*QueueingHintFunction{
				{Resource: fwk.Pod, ActionType: fwk.Add}: {
					{
						PluginName:     "preEnqueuePlugin",
						QueueingHintFn: queueHintReturnQueue,
					},
				},
				{Resource: fwk.Node, ActionType: fwk.Add}: {
					{
						PluginName: "otherPlugin",
						QueueingHintFn: func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
							if pod.Name == "pod1" {
								return fwk.Queue, nil
							}
							return fwk.QueueSkip, nil
						},
					},
				},
			}

			q := NewTestQueue(ctx, newDefaultQueueSort(), WithPreEnqueuePluginMap(preEnqueueMap), WithQueueingHintMapPerProfile(m))
			podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()

			setupInitialPodGroupState(t, ctx, q, tt.initialPods, tt.initialState, podGroup)

			// MoveAllToActiveOrBackoffQueue with the given event
			q.MoveAllToActiveOrBackoffQueue(logger, tt.event, nil, nil, tt.preCheck)

			// Verify conditions
			pgLookup := newQueuedPodGroupInfoForLookup(tt.initialPods[0])

			if inActive := q.activeQ.has(pgLookup); inActive != tt.expectedInActiveQ {
				t.Errorf("Expected target pod group in activeQ: %v, got %v", tt.expectedInActiveQ, inActive)
			}
			var entity framework.QueuedEntityInfo
			if tt.expectedInActiveQ {
				entity, _ = q.activeQ.get(pgLookup)
			}

			if q.backoffQ.has(pgLookup) {
				t.Errorf("Expected target pod group not to be present in backoffQ")
			}

			unschedulableEntity := q.unschedulableEntities.get(pgLookup)
			inUnschedulable := unschedulableEntity != nil
			if inUnschedulable != tt.expectedInUnschedulable {
				t.Errorf("Expected target pod group in unschedulableEntities: %v, got %v", tt.expectedInUnschedulable, inUnschedulable)
			}
			if tt.expectedInUnschedulable {
				entity = unschedulableEntity
			}

			if entity != nil {
				if isGated := entity.Gated(); isGated != tt.expectedGated {
					t.Errorf("Expected pod group to be gated: %v, got %v", tt.expectedGated, isGated)
				}
				if size := entity.Size(); size != tt.expectedGroupSize {
					t.Errorf("Expected pod group to be of size: %d, got %d", tt.expectedGroupSize, size)
				}

				if tt.expectedGroupSize > 0 {
					foundPod := false
					for _, pInfo := range entity.(*framework.QueuedPodGroupInfo).QueuedPodInfos {
						if pInfo.Pod.Name == tt.initialPods[0].Name {
							foundPod = true
							break
						}
					}
					if !foundPod {
						t.Errorf("Pod %s was not found in the pod group members", tt.initialPods[0].Name)
					}
				}
			}

			if tt.expectedInFlightEvent {
				foundEvent := false
				for _, ev := range q.activeQ.listInFlightEvents() {
					clusterEvent, ok := ev.(*clusterEvent)
					if !ok {
						continue
					}
					if clusterEvent.event.Label() == tt.event.Label() {
						foundEvent = true
						break
					}
				}
				if !foundEvent {
					t.Errorf("Expected in-flight event %s to be tracked, but it wasn't", tt.event.Label())
				}
				return
			}
		})
	}
}

func TestFlushBackoffQCompletedPodGroupMember(t *testing.T) {
	pgName := "pg-test"
	p1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	p2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").PodGroupName(pgName).Obj()

	tests := []struct {
		name               string
		initialPods        []*v1.Pod
		initialState       initialQueueState
		advanceClock       time.Duration
		expectedInActiveQ  bool
		expectedInBackoffQ bool
		expectedGroupSize  int
	}{
		{
			name:              "flushing backoffQ after backoff completes moves pod group to activeQ",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateBackoff,
			advanceClock:      10 * time.Second,
			expectedInActiveQ: true,
			expectedGroupSize: 2,
		},
		{
			name:               "flushing backoffQ before backoff completes keeps pod group in backoffQ",
			initialPods:        []*v1.Pod{p1, p2},
			initialState:       stateBackoff,
			advanceClock:       0,
			expectedInBackoffQ: true,
			expectedGroupSize:  2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			preEnqueueMap := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"preEnqueuePlugin": &preEnqueuePlugin{allowlists: []string{"allow"}},
				},
			}

			fakeClock := testingclock.NewFakeClock(time.Now())
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithPreEnqueuePluginMap(preEnqueueMap), WithClock(fakeClock))
			podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()

			setupInitialPodGroupState(t, ctx, q, tt.initialPods, tt.initialState, podGroup)

			if tt.advanceClock > 0 {
				fakeClock.Step(tt.advanceClock)
			}

			// Flush the backoff queue
			q.flushBackoffQCompleted(logger)

			// Verify conditions
			pgLookup := newQueuedPodGroupInfoForLookup(tt.initialPods[0])

			if inActive := q.activeQ.has(pgLookup); inActive != tt.expectedInActiveQ {
				t.Errorf("Expected target pod group in activeQ: %v, got %v", tt.expectedInActiveQ, inActive)
			}
			var entity framework.QueuedEntityInfo
			if tt.expectedInActiveQ {
				entity, _ = q.activeQ.get(pgLookup)
			}

			if inBackoff := q.backoffQ.has(pgLookup); inBackoff != tt.expectedInBackoffQ {
				t.Errorf("Expected target pod group in backoffQ: %v, got %v", tt.expectedInBackoffQ, inBackoff)
			}
			if tt.expectedInBackoffQ {
				entity, _ = q.backoffQ.get(pgLookup)
			}

			if q.unschedulableEntities.get(pgLookup) != nil {
				t.Errorf("Expected target pod group not to be present in unschedulableEntities")
			}

			if entity != nil {
				if size := entity.Size(); size != tt.expectedGroupSize {
					t.Errorf("Expected pod group to be of size: %d, got %d", tt.expectedGroupSize, size)
				}
			}
		})
	}
}

func TestFlushUnschedulableEntitiesLeftoverPodGroupMember(t *testing.T) {
	pgName := "pg-test"
	p1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	p2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").PodGroupName(pgName).Obj()
	gatedP1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").PodGroupName(pgName).Obj()

	tests := []struct {
		name                    string
		initialPods             []*v1.Pod
		initialState            initialQueueState
		advanceClock            time.Duration
		expectedInActiveQ       bool
		expectedInUnschedulable bool
		expectedGated           bool
		expectedGroupSize       int
	}{
		{
			name:              "flushing unschedulable leftover after max duration moves ungated pod group to activeQ",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateUnschedulable,
			advanceClock:      DefaultPodMaxInUnschedulablePodsDuration + time.Second,
			expectedInActiveQ: true,
			expectedGroupSize: 2,
		},
		{
			name:                    "flushing unschedulable leftover before max duration keeps pod group in unschedulableEntities",
			initialPods:             []*v1.Pod{p1, p2},
			initialState:            stateUnschedulable,
			advanceClock:            0,
			expectedInUnschedulable: true,
			expectedGroupSize:       2,
		},
		{
			name:                    "flushing unschedulable leftover for gated pod group after max duration keeps it in unschedulableEntities as gated",
			initialPods:             []*v1.Pod{gatedP1, p2},
			initialState:            stateGated,
			advanceClock:            DefaultPodMaxInUnschedulablePodsDuration + time.Second,
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			preEnqueueMap := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"preEnqueuePlugin": &preEnqueuePlugin{allowlists: []string{"allow"}},
				},
			}

			fakeClock := testingclock.NewFakeClock(time.Now())
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithPreEnqueuePluginMap(preEnqueueMap), WithClock(fakeClock))
			podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()

			setupInitialPodGroupState(t, ctx, q, tt.initialPods, tt.initialState, podGroup)

			if tt.advanceClock > 0 {
				fakeClock.Step(tt.advanceClock)
			}

			// Flush unschedulable entities
			q.flushUnschedulableEntitiesLeftover(logger)

			// Verify conditions
			pgLookup := newQueuedPodGroupInfoForLookup(tt.initialPods[0])

			if inActive := q.activeQ.has(pgLookup); inActive != tt.expectedInActiveQ {
				t.Errorf("Expected target pod group in activeQ: %v, got %v", tt.expectedInActiveQ, inActive)
			}
			var entity framework.QueuedEntityInfo
			if tt.expectedInActiveQ {
				entity, _ = q.activeQ.get(pgLookup)
			}

			if q.backoffQ.has(pgLookup) {
				t.Errorf("Expected target pod group not to be present in backoffQ")
			}

			unschedulableEntity := q.unschedulableEntities.get(pgLookup)
			inUnschedulable := unschedulableEntity != nil
			if inUnschedulable != tt.expectedInUnschedulable {
				t.Errorf("Expected target pod group in unschedulableEntities: %v, got %v", tt.expectedInUnschedulable, inUnschedulable)
			}
			if tt.expectedInUnschedulable {
				entity = unschedulableEntity
			}

			if entity != nil {
				if isGated := entity.Gated(); isGated != tt.expectedGated {
					t.Errorf("Expected pod group to be gated: %v, got %v", tt.expectedGated, isGated)
				}
				if size := entity.Size(); size != tt.expectedGroupSize {
					t.Errorf("Expected pod group to be of size: %d, got %d", tt.expectedGroupSize, size)
				}
			}
		})
	}
}

func TestAddUnschedulablePodIfNotPresentPodGroupMember(t *testing.T) {
	pgName := "pg-test"
	p1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	p2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").PodGroupName(pgName).Obj()
	p3 := st.MakePod().Name("pod3").Namespace("ns1").UID("pod3").Label("allow", "").PodGroupName(pgName).Obj()
	gatedP3 := st.MakePod().Name("pod3").Namespace("ns1").UID("pod3").PodGroupName(pgName).Obj()

	pInfo1 := &framework.QueuedPodInfo{
		PodInfo: mustNewPodInfo(p1),
		QueueingParams: framework.QueueingParams{
			UnschedulablePlugins: sets.New("fooPlugin"),
		},
	}
	pInfo1NoPlugins := &framework.QueuedPodInfo{
		PodInfo: mustNewPodInfo(p1),
	}
	pInfo2 := &framework.QueuedPodInfo{
		PodInfo: mustNewPodInfo(p2),
		QueueingParams: framework.QueueingParams{
			UnschedulablePlugins: sets.New("barPlugin"),
		},
	}
	gatedPInfo3 := &framework.QueuedPodInfo{
		PodInfo: mustNewPodInfo(gatedP3),
		QueueingParams: framework.QueueingParams{
			UnschedulablePlugins: sets.New("bazPlugin"),
		},
	}

	tests := []struct {
		name                    string
		initialPods             []*v1.Pod
		initialState            initialQueueState
		clearLastPopped         bool
		deletePodGroup          bool
		podsToAdd               []*framework.QueuedPodInfo
		expectedPodsInPending   int
		expectedInActiveQ       bool
		expectedInUnschedulable bool
		expectedInBackoffQ      bool
		expectedInIncomplete    []*v1.Pod
		expectedGated           bool
		expectedGroupSize       int
	}{
		{
			name:                  "single pod group member added with plugin to pending when group is in flight",
			initialPods:           []*v1.Pod{p1, p2},
			initialState:          statePopped,
			podsToAdd:             []*framework.QueuedPodInfo{pInfo1},
			expectedPodsInPending: 1,
		},
		{
			name:                  "single pod group member added with no plugins (consecutive error) to pending",
			initialPods:           []*v1.Pod{p1, p2},
			initialState:          statePopped,
			podsToAdd:             []*framework.QueuedPodInfo{pInfo1NoPlugins},
			expectedPodsInPending: 1,
		},
		{
			name:                  "multiple pod group members, including gated added with plugins sequentially to pending when group is in flight",
			initialPods:           []*v1.Pod{p1, p2, p3},
			initialState:          statePopped,
			podsToAdd:             []*framework.QueuedPodInfo{pInfo1, pInfo2, gatedPInfo3},
			expectedPodsInPending: 3,
		},
		{
			name:              "pod group is not last popped entity and no group exists, a new group is created and added to activeQ",
			initialPods:       nil,
			podsToAdd:         []*framework.QueuedPodInfo{pInfo1},
			expectedInActiveQ: true,
			expectedGroupSize: 1,
		},
		{
			name:                    "pod group exists in unschedulableEntities, pod is added and group remains in unschedulableEntities",
			initialPods:             []*v1.Pod{p2},
			initialState:            stateUnschedulable,
			clearLastPopped:         true,
			podsToAdd:               []*framework.QueuedPodInfo{pInfo1},
			expectedInUnschedulable: true,
			expectedGroupSize:       2,
		},
		{
			name:              "pod group exists in activeQ, pod is added and group remains in activeQ",
			initialPods:       []*v1.Pod{p2},
			initialState:      stateActive,
			clearLastPopped:   true,
			podsToAdd:         []*framework.QueuedPodInfo{pInfo1},
			expectedInActiveQ: true,
			expectedGroupSize: 2,
		},
		{
			name:               "pod group exists in backoffQ, pod is added and group remains in backoffQ",
			initialPods:        []*v1.Pod{p2},
			initialState:       stateBackoff,
			clearLastPopped:    true,
			podsToAdd:          []*framework.QueuedPodInfo{pInfo1},
			expectedInBackoffQ: true,
			expectedGroupSize:  2,
		},
		{
			name:                    "pod group exists in unschedulableEntities (gated), ungated pod is added and group remains gated in unschedulableEntities",
			initialPods:             []*v1.Pod{gatedP3},
			initialState:            stateGated,
			clearLastPopped:         true,
			podsToAdd:               []*framework.QueuedPodInfo{pInfo1},
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       2,
		},
		{
			name:                 "requeuing a pod group members after PodGroup was deleted moves them to incompletePodGroupPods",
			initialPods:          []*v1.Pod{p1, p2},
			initialState:         statePopped,
			deletePodGroup:       true,
			podsToAdd:            []*framework.QueuedPodInfo{pInfo1, pInfo2},
			expectedInIncomplete: []*v1.Pod{p1, p2},
		},
		{
			name:                 "requeuing a pod group members after PodGroup was deleted and is not last popped entity moves them to incompletePodGroupPods",
			initialPods:          []*v1.Pod{p1, p2},
			initialState:         statePopped,
			clearLastPopped:      true,
			deletePodGroup:       true,
			podsToAdd:            []*framework.QueuedPodInfo{pInfo1, pInfo2},
			expectedInIncomplete: []*v1.Pod{p1, p2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			preEnqueueMap := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"preEnqueuePlugin": &preEnqueuePlugin{allowlists: []string{"allow"}},
				},
			}

			q := NewTestQueue(ctx, newDefaultQueueSort(), WithPreEnqueuePluginMap(preEnqueueMap))
			podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()

			setupInitialPodGroupState(t, ctx, q, tt.initialPods, tt.initialState, podGroup)

			if tt.clearLastPopped {
				q.activeQ.clearPoppedEntity()
			}

			if tt.deletePodGroup {
				q.DeletePodGroup(logger, podGroup)
			}

			// Add unschedulable pods
			for _, pInfo := range tt.podsToAdd {
				pInfoCloned := pInfo.DeepCopy()
				if err := q.AddUnschedulablePodIfNotPresent(logger, pInfoCloned, q.SchedulingCycle()); err != nil {
					t.Errorf("Failed to add unschedulable pods %s: %v", pInfoCloned.Pod.Name, err)
				}
			}

			// Verify conditions
			pgLookup := newQueuedPodGroupInfoForLookup(tt.podsToAdd[0].Pod)

			if inActive := q.activeQ.has(pgLookup); inActive != tt.expectedInActiveQ {
				t.Errorf("Expected pod group in activeQ: %v, got %v", tt.expectedInActiveQ, inActive)
			}
			var entity framework.QueuedEntityInfo
			if tt.expectedInActiveQ {
				entity, _ = q.activeQ.get(pgLookup)
			}

			if inBackoff := q.backoffQ.has(pgLookup); inBackoff != tt.expectedInBackoffQ {
				t.Errorf("Expected pod group in backoffQ: %v, got %v", tt.expectedInBackoffQ, inBackoff)
			}
			if tt.expectedInBackoffQ {
				entity, _ = q.backoffQ.get(pgLookup)
			}

			unschedulableEntity := q.unschedulableEntities.get(pgLookup)
			inUnschedulable := unschedulableEntity != nil
			if inUnschedulable != tt.expectedInUnschedulable {
				t.Errorf("Expected pod group in unschedulableEntities: %v, got %v", tt.expectedInUnschedulable, inUnschedulable)
			}
			if tt.expectedInUnschedulable {
				entity = unschedulableEntity
			}

			if entity != nil {
				if isGated := entity.Gated(); isGated != tt.expectedGated {
					t.Errorf("Expected pod group to be gated: %v, got %v", tt.expectedGated, isGated)
				}
				if size := entity.Size(); size != tt.expectedGroupSize {
					t.Errorf("Expected pod group to be of size: %d, got %d", tt.expectedGroupSize, size)
				}
			}

			if pendingLen := q.pendingPodGroupPods.len(); pendingLen != tt.expectedPodsInPending {
				t.Errorf("Expected pod group to have %d pods in pendingPodGroupPods, got %d", tt.expectedPodsInPending, pendingLen)
			}

			for _, pod := range tt.expectedInIncomplete {
				if !q.incompletePodGroupPods.has(pod) {
					t.Errorf("Expected pod %s in incompletePodGroupPods", pod.Name)
				}
			}
			if q.incompletePodGroupPods.len() != len(tt.expectedInIncomplete) {
				t.Errorf("Expected incompletePodGroupPods size to be %d, got %d", len(tt.expectedInIncomplete), q.incompletePodGroupPods.len())
			}
		})
	}
}

func TestAddPodGroup(t *testing.T) {
	pgName := "pg-test"
	p1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	p2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").PodGroupName(pgName).Obj()
	gatedP1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").PodGroupName(pgName).Obj()
	podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()

	tests := []struct {
		name                    string
		initialPods             []*v1.Pod
		initialState            initialQueueState
		expectedInActiveQ       bool
		expectedInUnschedulable bool
		expectedGated           bool
		expectedGroupSize       int
	}{
		{
			name:              "add pod group without waiting pods",
			initialState:      stateIncomplete,
			expectedInActiveQ: false,
		},
		{
			name:              "add pod group moves pods from incompletePodGroupPods to activeQ",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateIncomplete,
			expectedInActiveQ: true,
			expectedGroupSize: 2,
		},
		{
			name:                    "add pod group moves gated pods from incompletePodGroupPods to unschedulableEntities",
			initialPods:             []*v1.Pod{gatedP1, p2},
			initialState:            stateIncomplete,
			expectedInUnschedulable: true,
			expectedGated:           true,
			expectedGroupSize:       2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			// Configure a PreEnqueue plugin that allows pods with label "allow", but gates others.
			preEnqueueMap := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"preEnqueuePlugin": &preEnqueuePlugin{allowlists: []string{"allow"}},
				},
			}
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithPreEnqueuePluginMap(preEnqueueMap))

			setupInitialPodGroupState(t, ctx, q, tt.initialPods, tt.initialState, podGroup)

			q.AddPodGroup(logger, podGroup)

			pgLookup := &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: podGroup.Namespace,
					Name:      podGroup.Name,
				},
			}
			gotPodGroup, ok := q.workloadForest.getPodGroup(podGroup)
			if !ok {
				t.Fatalf("Expected pod group to be in workloadForest")
			}
			if diff := cmp.Diff(podGroup, gotPodGroup); diff != "" {
				t.Errorf("Unexpected pod group object (-want +got):\n%s", diff)
			}

			if inActive := q.activeQ.has(pgLookup); inActive != tt.expectedInActiveQ {
				t.Errorf("Expected incoming pod in activeQ: %v, got %v", tt.expectedInActiveQ, inActive)
			}
			var entity framework.QueuedEntityInfo
			if tt.expectedInActiveQ {
				entity, _ = q.activeQ.get(pgLookup)
			}

			unschedulableEntity := q.unschedulableEntities.get(pgLookup)
			inUnschedulable := unschedulableEntity != nil
			if inUnschedulable != tt.expectedInUnschedulable {
				t.Errorf("Expected incoming pod in unschedulableEntities: %v, got %v", tt.expectedInUnschedulable, inUnschedulable)
			}
			if tt.expectedInUnschedulable {
				entity = unschedulableEntity
			}

			if entity != nil {
				if isGated := entity.Gated(); isGated != tt.expectedGated {
					t.Errorf("Expected pod group to be gated: %v, got %v", tt.expectedGated, isGated)
				}
				if size := entity.Size(); size != tt.expectedGroupSize {
					t.Errorf("Expected pod group to be of size: %d, got %d", tt.expectedGroupSize, size)
				}
				pgInfo := entity.(*framework.QueuedPodGroupInfo)
				if diff := cmp.Diff(podGroup, pgInfo.PodGroup); diff != "" {
					t.Errorf("Unexpected pod group object in QueuedPodGroupInfo (-want +got):\n%s", diff)
				}
			}

			for _, pod := range tt.initialPods {
				if q.incompletePodGroupPods.has(pod) {
					t.Errorf("Expected pod %s not to be present in incompletePodGroupPods", pod.Name)
				}
			}
		})
	}
}

func TestUpdatePodGroup(t *testing.T) {
	pgName := "pg-test"
	p1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	p2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").PodGroupName(pgName).Obj()
	gatedP1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").PodGroupName(pgName).Obj()
	podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").MinCount(1).Obj()
	updatedPodGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").MinCount(2).Obj()

	tests := []struct {
		name                          string
		initialPods                   []*v1.Pod
		initialState                  initialQueueState
		expectedInActiveQ             bool
		expectedInBackoffQ            bool
		expectedInUnschedulable       bool
		expectedInPendingPodGroupPods bool
		expectedGated                 bool
	}{
		{
			name:         "update pod group without pods",
			initialState: stateActive, // will just add the PodGroup
		},
		{
			name:              "update active pod group",
			initialPods:       []*v1.Pod{p1, p2},
			initialState:      stateActive,
			expectedInActiveQ: true,
		},
		{
			name:               "update backoff pod group",
			initialPods:        []*v1.Pod{p1, p2},
			initialState:       stateBackoff,
			expectedInBackoffQ: true,
		},
		{
			name:                    "update unschedulable pod group",
			initialPods:             []*v1.Pod{p1, p2},
			initialState:            stateUnschedulable,
			expectedInUnschedulable: true,
		},
		{
			name:                    "update gated pod group",
			initialPods:             []*v1.Pod{gatedP1, p2},
			initialState:            stateGated,
			expectedInUnschedulable: true,
			expectedGated:           true,
		},
		{
			name:                          "update in-flight pod group",
			initialPods:                   []*v1.Pod{p1, p2},
			initialState:                  statePopped,
			expectedInPendingPodGroupPods: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			preEnqueueMap := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"preEnqueuePlugin": &preEnqueuePlugin{allowlists: []string{"allow"}},
				},
			}
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithPreEnqueuePluginMap(preEnqueueMap))

			setupInitialPodGroupState(t, ctx, q, tt.initialPods, tt.initialState, podGroup)

			q.UpdatePodGroup(logger, updatedPodGroup)

			gotPodGroup, ok := q.workloadForest.getPodGroup(podGroup)
			if !ok {
				t.Fatalf("Expected pod group to be in workloadForest")
			}
			if diff := cmp.Diff(updatedPodGroup, gotPodGroup); diff != "" {
				t.Errorf("Unexpected pod group object (-want +got):\n%s", diff)
			}

			pgLookup := &framework.QueuedPodGroupInfo{
				PodGroupInfo: &framework.PodGroupInfo{
					Namespace: podGroup.Namespace,
					Name:      podGroup.Name,
				},
			}
			if inActive := q.activeQ.has(pgLookup); inActive != tt.expectedInActiveQ {
				t.Errorf("Expected target pod group in activeQ: %v, got %v", tt.expectedInActiveQ, inActive)
			}
			var entity framework.QueuedEntityInfo
			if tt.expectedInActiveQ {
				entity, _ = q.activeQ.get(pgLookup)
			}

			if inBackoff := q.backoffQ.has(pgLookup); inBackoff != tt.expectedInBackoffQ {
				t.Errorf("Expected target pod group in backoffQ: %v, got %v", tt.expectedInBackoffQ, inBackoff)
			}
			if tt.expectedInBackoffQ {
				entity, _ = q.backoffQ.get(pgLookup)
			}

			unschedulableEntity := q.unschedulableEntities.get(pgLookup)
			inUnschedulable := unschedulableEntity != nil
			if inUnschedulable != tt.expectedInUnschedulable {
				t.Errorf("Expected target pod group in unschedulableEntities: %v, got %v", tt.expectedInUnschedulable, inUnschedulable)
			}
			if tt.expectedInUnschedulable {
				entity = unschedulableEntity
			}

			if entity != nil {
				if isGated := entity.Gated(); isGated != tt.expectedGated {
					t.Errorf("Expected pod group to be gated: %v, got %v", tt.expectedGated, isGated)
				}
				pgInfo := entity.(*framework.QueuedPodGroupInfo)
				if diff := cmp.Diff(updatedPodGroup, pgInfo.PodGroup); diff != "" {
					t.Errorf("Unexpected pod group object in QueuedPodGroupInfo (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestDeletePodGroup(t *testing.T) {
	pgName := "pg-test"
	p1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Label("allow", "").PodGroupName(pgName).Obj()
	p2 := st.MakePod().Name("pod2").Namespace("ns1").UID("pod2").Label("allow", "").PodGroupName(pgName).Obj()
	gatedP1 := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").PodGroupName(pgName).Obj()
	podGroup := st.MakePodGroup().Name(pgName).Namespace("ns1").Obj()

	tests := []struct {
		name                 string
		initialPods          []*v1.Pod
		initialState         initialQueueState
		pendingPods          []*v1.Pod
		expectedInIncomplete []*v1.Pod
	}{
		{
			name:                 "delete active pod group moves pods to incompletePodGroupPods",
			initialPods:          []*v1.Pod{p1, p2},
			initialState:         stateActive,
			expectedInIncomplete: []*v1.Pod{p1, p2},
		},
		{
			name:                 "delete backoff pod group moves pods to incompletePodGroupPods",
			initialPods:          []*v1.Pod{p1, p2},
			initialState:         stateBackoff,
			expectedInIncomplete: []*v1.Pod{p1, p2},
		},
		{
			name:                 "delete unschedulable pod group moves pods to incompletePodGroupPods",
			initialPods:          []*v1.Pod{p1, p2},
			initialState:         stateUnschedulable,
			expectedInIncomplete: []*v1.Pod{p1, p2},
		},
		{
			name:                 "delete gated pod group moves pods to incompletePodGroupPods",
			initialPods:          []*v1.Pod{gatedP1, p2},
			initialState:         stateGated,
			expectedInIncomplete: []*v1.Pod{gatedP1, p2},
		},
		{
			name:                 "delete in-flight pod group moves pending pods to incompletePodGroupPods",
			initialPods:          []*v1.Pod{p1, p2},
			initialState:         statePopped,
			pendingPods:          []*v1.Pod{p1, p2},
			expectedInIncomplete: []*v1.Pod{p1, p2},
		},
		{
			name:                 "delete in-flight pod group with no pending pods",
			initialPods:          []*v1.Pod{p1, p2},
			initialState:         statePopped,
			expectedInIncomplete: []*v1.Pod{},
		},
		{
			name:                 "delete pod group with no pods",
			expectedInIncomplete: []*v1.Pod{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: true,
			})

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			preEnqueueMap := map[string]map[string]fwk.PreEnqueuePlugin{
				"": {
					"preEnqueuePlugin": &preEnqueuePlugin{allowlists: []string{"allow"}},
				},
			}
			q := NewTestQueue(ctx, newDefaultQueueSort(), WithPreEnqueuePluginMap(preEnqueueMap))

			setupInitialPodGroupState(t, ctx, q, tt.initialPods, tt.initialState, podGroup)

			for _, pod := range tt.pendingPods {
				q.Add(ctx, pod)
			}

			q.DeletePodGroup(logger, podGroup)

			_, ok := q.workloadForest.getPodGroup(podGroup)
			if ok {
				t.Errorf("Expected pod group not to be present in workloadForest")
			}

			pgLookup := newQueuedPodGroupInfoForLookup(p1)
			if q.activeQ.has(pgLookup) {
				t.Errorf("Expected pod group not to be in activeQ")
			}
			if q.backoffQ.has(pgLookup) {
				t.Errorf("Expected pod group not to be in backoffQ")
			}
			if q.unschedulableEntities.get(pgLookup) != nil {
				t.Errorf("Expected pod group not to be in unschedulableEntities")
			}

			for _, pod := range tt.pendingPods {
				if q.pendingPodGroupPods.has(pod) {
					t.Errorf("Expected pod %s not to be in pendingPodGroupPods", pod.Name)
				}
			}

			for _, pod := range tt.expectedInIncomplete {
				if !q.incompletePodGroupPods.has(pod) {
					t.Errorf("Expected pod %s in incompletePodGroupPods", pod.Name)
				}
			}
			if q.incompletePodGroupPods.len() != len(tt.expectedInIncomplete) {
				t.Errorf("Expected incompletePodGroupPods size to be %d, got %d", len(tt.expectedInIncomplete), q.incompletePodGroupPods.len())
			}
		})
	}
}
