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
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/component-base/metrics/testutil"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

const queueMetricMetadata = `
		# HELP scheduler_queue_incoming_pods_total [ALPHA] Number of pods added to scheduling queues by event and queue type.
		# TYPE scheduler_queue_incoming_pods_total counter
	`

var lowPriority, midPriority, highPriority = int32(0), int32(100), int32(1000)
var mediumPriority = (lowPriority + highPriority) / 2
var highPriorityPod, highPriNominatedPod, medPriorityPod, unschedulablePod = v1.Pod{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "hpp",
		Namespace: "ns1",
		UID:       "hppns1",
	},
	Spec: v1.PodSpec{
		Priority: &highPriority,
	},
},
	v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "hpp",
			Namespace: "ns1",
			UID:       "hppns1",
		},
		Spec: v1.PodSpec{
			Priority: &highPriority,
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node1",
		},
	},
	v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mpp",
			Namespace: "ns2",
			UID:       "mppns2",
			Annotations: map[string]string{
				"annot2": "val2",
			},
		},
		Spec: v1.PodSpec{
			Priority: &mediumPriority,
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node1",
		},
	},
	v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "up",
			Namespace: "ns1",
			UID:       "upns1",
			Annotations: map[string]string{
				"annot2": "val2",
			},
		},
		Spec: v1.PodSpec{
			Priority: &lowPriority,
		},
		Status: v1.PodStatus{
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodScheduled,
					Status: v1.ConditionFalse,
					Reason: v1.PodReasonUnschedulable,
				},
			},
			NominatedNodeName: "node1",
		},
	}

func getUnschedulablePod(p *PriorityQueue, pod *v1.Pod) *v1.Pod {
	pInfo := p.unschedulableQ.get(pod)
	if pInfo != nil {
		return pInfo.Pod
	}
	return nil
}

func TestPriorityQueue_Add(t *testing.T) {
	q := NewPriorityQueue(newDefaultQueueSort())
	if err := q.Add(&medPriorityPod); err != nil {
		t.Errorf("add failed: %v", err)
	}
	if err := q.Add(&unschedulablePod); err != nil {
		t.Errorf("add failed: %v", err)
	}
	if err := q.Add(&highPriorityPod); err != nil {
		t.Errorf("add failed: %v", err)
	}
	expectedNominatedPods := &nominatedPodMap{
		nominatedPodToNode: map[types.UID]string{
			medPriorityPod.UID:   "node1",
			unschedulablePod.UID: "node1",
		},
		nominatedPods: map[string][]*v1.Pod{
			"node1": {&medPriorityPod, &unschedulablePod},
		},
	}
	if !reflect.DeepEqual(q.PodNominator, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after adding pods. Expected: %v, got: %v", expectedNominatedPods, q.PodNominator)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &highPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Pod.Name)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &medPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Pod.Name)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &unschedulablePod {
		t.Errorf("Expected: %v after Pop, but got: %v", unschedulablePod.Name, p.Pod.Name)
	}
	if len(q.PodNominator.(*nominatedPodMap).nominatedPods["node1"]) != 2 {
		t.Errorf("Expected medPriorityPod and unschedulablePod to be still present in nomindatePods: %v", q.PodNominator.(*nominatedPodMap).nominatedPods["node1"])
	}
}

func newDefaultQueueSort() framework.LessFunc {
	sort := &queuesort.PrioritySort{}
	return sort.Less
}

func TestPriorityQueue_AddWithReversePriorityLessFunc(t *testing.T) {
	q := NewPriorityQueue(newDefaultQueueSort())
	if err := q.Add(&medPriorityPod); err != nil {
		t.Errorf("add failed: %v", err)
	}
	if err := q.Add(&highPriorityPod); err != nil {
		t.Errorf("add failed: %v", err)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &highPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Pod.Name)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &medPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Pod.Name)
	}
}

func TestPriorityQueue_AddUnschedulableIfNotPresent(t *testing.T) {
	q := NewPriorityQueue(newDefaultQueueSort())
	q.Add(&highPriNominatedPod)
	q.AddUnschedulableIfNotPresent(newQueuedPodInfoNoTimestamp(&highPriNominatedPod), q.SchedulingCycle()) // Must not add anything.
	q.AddUnschedulableIfNotPresent(newQueuedPodInfoNoTimestamp(&unschedulablePod), q.SchedulingCycle())
	expectedNominatedPods := &nominatedPodMap{
		nominatedPodToNode: map[types.UID]string{
			unschedulablePod.UID:    "node1",
			highPriNominatedPod.UID: "node1",
		},
		nominatedPods: map[string][]*v1.Pod{
			"node1": {&highPriNominatedPod, &unschedulablePod},
		},
	}
	if !reflect.DeepEqual(q.PodNominator, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after adding pods. Expected: %v, got: %v", expectedNominatedPods, q.PodNominator)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &highPriNominatedPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriNominatedPod.Name, p.Pod.Name)
	}
	if len(q.PodNominator.(*nominatedPodMap).nominatedPods) != 1 {
		t.Errorf("Expected nomindatePods to have one element: %v", q.PodNominator)
	}
	if getUnschedulablePod(q, &unschedulablePod) != &unschedulablePod {
		t.Errorf("Pod %v was not found in the unschedulableQ.", unschedulablePod.Name)
	}
}

// TestPriorityQueue_AddUnschedulableIfNotPresent_Backoff tests the scenarios when
// AddUnschedulableIfNotPresent is called asynchronously.
// Pods in and before current scheduling cycle will be put back to activeQueue
// if we were trying to schedule them when we received move request.
func TestPriorityQueue_AddUnschedulableIfNotPresent_Backoff(t *testing.T) {
	q := NewPriorityQueue(newDefaultQueueSort(), WithClock(clock.NewFakeClock(time.Now())))
	totalNum := 10
	expectedPods := make([]v1.Pod, 0, totalNum)
	for i := 0; i < totalNum; i++ {
		priority := int32(i)
		p := v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("pod%d", i),
				Namespace: fmt.Sprintf("ns%d", i),
				UID:       types.UID(fmt.Sprintf("upns%d", i)),
			},
			Spec: v1.PodSpec{
				Priority: &priority,
			},
		}
		expectedPods = append(expectedPods, p)
		// priority is to make pods ordered in the PriorityQueue
		q.Add(&p)
	}

	// Pop all pods except for the first one
	for i := totalNum - 1; i > 0; i-- {
		p, _ := q.Pop()
		if !reflect.DeepEqual(&expectedPods[i], p.Pod) {
			t.Errorf("Unexpected pod. Expected: %v, got: %v", &expectedPods[i], p)
		}
	}

	// move all pods to active queue when we were trying to schedule them
	q.MoveAllToActiveOrBackoffQueue("test")
	oldCycle := q.SchedulingCycle()

	firstPod, _ := q.Pop()
	if !reflect.DeepEqual(&expectedPods[0], firstPod.Pod) {
		t.Errorf("Unexpected pod. Expected: %v, got: %v", &expectedPods[0], firstPod)
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

		if err := q.AddUnschedulableIfNotPresent(newQueuedPodInfoNoTimestamp(unschedulablePod), oldCycle); err != nil {
			t.Errorf("Failed to call AddUnschedulableIfNotPresent(%v): %v", unschedulablePod.Name, err)
		}
	}

	// Since there was a move request at the same cycle as "oldCycle", these pods
	// should be in the backoff queue.
	for i := 1; i < totalNum; i++ {
		if _, exists, _ := q.podBackoffQ.Get(newQueuedPodInfoNoTimestamp(&expectedPods[i])); !exists {
			t.Errorf("Expected %v to be added to podBackoffQ.", expectedPods[i].Name)
		}
	}
}

func TestPriorityQueue_Pop(t *testing.T) {
	q := NewPriorityQueue(newDefaultQueueSort())
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		if p, err := q.Pop(); err != nil || p.Pod != &medPriorityPod {
			t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Pod.Name)
		}
		if len(q.PodNominator.(*nominatedPodMap).nominatedPods["node1"]) != 1 {
			t.Errorf("Expected medPriorityPod to be present in nomindatePods: %v", q.PodNominator.(*nominatedPodMap).nominatedPods["node1"])
		}
	}()
	q.Add(&medPriorityPod)
	wg.Wait()
}

func TestPriorityQueue_Update(t *testing.T) {
	q := NewPriorityQueue(newDefaultQueueSort())
	q.Update(nil, &highPriorityPod)
	if _, exists, _ := q.activeQ.Get(newQueuedPodInfoNoTimestamp(&highPriorityPod)); !exists {
		t.Errorf("Expected %v to be added to activeQ.", highPriorityPod.Name)
	}
	if len(q.PodNominator.(*nominatedPodMap).nominatedPods) != 0 {
		t.Errorf("Expected nomindatePods to be empty: %v", q.PodNominator)
	}
	// Update highPriorityPod and add a nominatedNodeName to it.
	q.Update(&highPriorityPod, &highPriNominatedPod)
	if q.activeQ.Len() != 1 {
		t.Error("Expected only one item in activeQ.")
	}
	if len(q.PodNominator.(*nominatedPodMap).nominatedPods) != 1 {
		t.Errorf("Expected one item in nomindatePods map: %v", q.PodNominator)
	}
	// Updating an unschedulable pod which is not in any of the two queues, should
	// add the pod to activeQ.
	q.Update(&unschedulablePod, &unschedulablePod)
	if _, exists, _ := q.activeQ.Get(newQueuedPodInfoNoTimestamp(&unschedulablePod)); !exists {
		t.Errorf("Expected %v to be added to activeQ.", unschedulablePod.Name)
	}
	// Updating a pod that is already in activeQ, should not change it.
	q.Update(&unschedulablePod, &unschedulablePod)
	if len(q.unschedulableQ.podInfoMap) != 0 {
		t.Error("Expected unschedulableQ to be empty.")
	}
	if _, exists, _ := q.activeQ.Get(newQueuedPodInfoNoTimestamp(&unschedulablePod)); !exists {
		t.Errorf("Expected: %v to be added to activeQ.", unschedulablePod.Name)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &highPriNominatedPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Pod.Name)
	}
	// Updating a pod that is in unschedulableQ in a way that it may
	// become schedulable should add the pod to the activeQ.
	q.AddUnschedulableIfNotPresent(q.newQueuedPodInfo(&medPriorityPod), q.SchedulingCycle())
	if len(q.unschedulableQ.podInfoMap) != 1 {
		t.Error("Expected unschedulableQ to be 1.")
	}
	updatedPod := medPriorityPod.DeepCopy()
	updatedPod.ClusterName = "test"
	q.Update(&medPriorityPod, updatedPod)
	if p, err := q.Pop(); err != nil || p.Pod != updatedPod {
		t.Errorf("Expected: %v after Pop, but got: %v", updatedPod.Name, p.Pod.Name)
	}
}

func TestPriorityQueue_Delete(t *testing.T) {
	q := NewPriorityQueue(newDefaultQueueSort())
	q.Update(&highPriorityPod, &highPriNominatedPod)
	q.Add(&unschedulablePod)
	if err := q.Delete(&highPriNominatedPod); err != nil {
		t.Errorf("delete failed: %v", err)
	}
	if _, exists, _ := q.activeQ.Get(newQueuedPodInfoNoTimestamp(&unschedulablePod)); !exists {
		t.Errorf("Expected %v to be in activeQ.", unschedulablePod.Name)
	}
	if _, exists, _ := q.activeQ.Get(newQueuedPodInfoNoTimestamp(&highPriNominatedPod)); exists {
		t.Errorf("Didn't expect %v to be in activeQ.", highPriorityPod.Name)
	}
	if len(q.PodNominator.(*nominatedPodMap).nominatedPods) != 1 {
		t.Errorf("Expected nomindatePods to have only 'unschedulablePod': %v", q.PodNominator.(*nominatedPodMap).nominatedPods)
	}
	if err := q.Delete(&unschedulablePod); err != nil {
		t.Errorf("delete failed: %v", err)
	}
	if len(q.PodNominator.(*nominatedPodMap).nominatedPods) != 0 {
		t.Errorf("Expected nomindatePods to be empty: %v", q.PodNominator)
	}
}

func TestPriorityQueue_MoveAllToActiveOrBackoffQueue(t *testing.T) {
	c := clock.NewFakeClock(time.Now())
	q := NewPriorityQueue(newDefaultQueueSort(), WithClock(c))
	q.Add(&medPriorityPod)
	q.AddUnschedulableIfNotPresent(q.newQueuedPodInfo(&unschedulablePod), q.SchedulingCycle())
	q.AddUnschedulableIfNotPresent(q.newQueuedPodInfo(&highPriorityPod), q.SchedulingCycle())
	// Pods is still backing off, move the pod into backoffQ.
	q.MoveAllToActiveOrBackoffQueue("test")
	if q.activeQ.Len() != 1 {
		t.Errorf("Expected 1 item to be in activeQ, but got: %v", q.activeQ.Len())
	}
	if q.podBackoffQ.Len() != 2 {
		t.Errorf("Expected 2 items to be in podBackoffQ, but got: %v", q.podBackoffQ.Len())
	}

	// pop out the pods in the backoffQ.
	q.podBackoffQ.Pop()
	q.podBackoffQ.Pop()

	q.schedulingCycle++
	q.AddUnschedulableIfNotPresent(q.newQueuedPodInfo(&unschedulablePod), q.SchedulingCycle())
	q.AddUnschedulableIfNotPresent(q.newQueuedPodInfo(&highPriorityPod), q.SchedulingCycle())
	if q.unschedulableQ.get(&unschedulablePod) == nil || q.unschedulableQ.get(&highPriorityPod) == nil {
		t.Errorf("Expected %v and %v in the unschedulableQ", unschedulablePod.Name, highPriorityPod.Name)
	}
	// Move clock by podInitialBackoffDuration, so that pods in the unschedulableQ would pass the backing off,
	// and the pods will be moved into activeQ.
	c.Step(q.podInitialBackoffDuration)
	q.MoveAllToActiveOrBackoffQueue("test")
	if q.activeQ.Len() != 3 {
		t.Errorf("Expected 3 items to be in activeQ, but got: %v", q.activeQ.Len())
	}
	if q.podBackoffQ.Len() != 0 {
		t.Errorf("Expected 0 item to be in podBackoffQ, but got: %v", q.podBackoffQ.Len())
	}

}

// TestPriorityQueue_AssignedPodAdded tests AssignedPodAdded. It checks that
// when a pod with pod affinity is in unschedulableQ and another pod with a
// matching label is added, the unschedulable pod is moved to activeQ.
func TestPriorityQueue_AssignedPodAdded(t *testing.T) {
	affinityPod := unschedulablePod.DeepCopy()
	affinityPod.Name = "afp"
	affinityPod.Spec = v1.PodSpec{
		Affinity: &v1.Affinity{
			PodAffinity: &v1.PodAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
					{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "service",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"securityscan", "value2"},
								},
							},
						},
						TopologyKey: "region",
					},
				},
			},
		},
		Priority: &mediumPriority,
	}
	labelPod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "lbp",
			Namespace: affinityPod.Namespace,
			Labels:    map[string]string{"service": "securityscan"},
		},
		Spec: v1.PodSpec{NodeName: "machine1"},
	}

	c := clock.NewFakeClock(time.Now())
	q := NewPriorityQueue(newDefaultQueueSort(), WithClock(c))
	q.Add(&medPriorityPod)
	// Add a couple of pods to the unschedulableQ.
	q.AddUnschedulableIfNotPresent(q.newQueuedPodInfo(&unschedulablePod), q.SchedulingCycle())
	q.AddUnschedulableIfNotPresent(q.newQueuedPodInfo(affinityPod), q.SchedulingCycle())

	// Move clock to make the unschedulable pods complete backoff.
	c.Step(DefaultPodInitialBackoffDuration + time.Second)
	// Simulate addition of an assigned pod. The pod has matching labels for
	// affinityPod. So, affinityPod should go to activeQ.
	q.AssignedPodAdded(&labelPod)
	if getUnschedulablePod(q, affinityPod) != nil {
		t.Error("affinityPod is still in the unschedulableQ.")
	}
	if _, exists, _ := q.activeQ.Get(newQueuedPodInfoNoTimestamp(affinityPod)); !exists {
		t.Error("affinityPod is not moved to activeQ.")
	}
	// Check that the other pod is still in the unschedulableQ.
	if getUnschedulablePod(q, &unschedulablePod) == nil {
		t.Error("unschedulablePod is not in the unschedulableQ.")
	}
}

func TestPriorityQueue_NominatedPodsForNode(t *testing.T) {
	q := NewPriorityQueue(newDefaultQueueSort())
	q.Add(&medPriorityPod)
	q.Add(&unschedulablePod)
	q.Add(&highPriorityPod)
	if p, err := q.Pop(); err != nil || p.Pod != &highPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Pod.Name)
	}
	expectedList := []*v1.Pod{&medPriorityPod, &unschedulablePod}
	if !reflect.DeepEqual(expectedList, q.NominatedPodsForNode("node1")) {
		t.Error("Unexpected list of nominated Pods for node.")
	}
	if q.NominatedPodsForNode("node2") != nil {
		t.Error("Expected list of nominated Pods for node2 to be empty.")
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

	q := NewPriorityQueue(newDefaultQueueSort())
	q.Add(&medPriorityPod)
	q.AddUnschedulableIfNotPresent(q.newQueuedPodInfo(&unschedulablePod), q.SchedulingCycle())
	q.AddUnschedulableIfNotPresent(q.newQueuedPodInfo(&highPriorityPod), q.SchedulingCycle())

	expectedSet := makeSet([]*v1.Pod{&medPriorityPod, &unschedulablePod, &highPriorityPod})
	if !reflect.DeepEqual(expectedSet, makeSet(q.PendingPods())) {
		t.Error("Unexpected list of pending Pods.")
	}
	// Move all to active queue. We should still see the same set of pods.
	q.MoveAllToActiveOrBackoffQueue("test")
	if !reflect.DeepEqual(expectedSet, makeSet(q.PendingPods())) {
		t.Error("Unexpected list of pending Pods...")
	}
}

func TestPriorityQueue_UpdateNominatedPodForNode(t *testing.T) {
	q := NewPriorityQueue(newDefaultQueueSort())
	if err := q.Add(&medPriorityPod); err != nil {
		t.Errorf("add failed: %v", err)
	}
	// Update unschedulablePod on a different node than specified in the pod.
	q.AddNominatedPod(&unschedulablePod, "node5")

	// Update nominated node name of a pod on a node that is not specified in the pod object.
	q.AddNominatedPod(&highPriorityPod, "node2")
	expectedNominatedPods := &nominatedPodMap{
		nominatedPodToNode: map[types.UID]string{
			medPriorityPod.UID:   "node1",
			highPriorityPod.UID:  "node2",
			unschedulablePod.UID: "node5",
		},
		nominatedPods: map[string][]*v1.Pod{
			"node1": {&medPriorityPod},
			"node2": {&highPriorityPod},
			"node5": {&unschedulablePod},
		},
	}
	if !reflect.DeepEqual(q.PodNominator, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after adding pods. Expected: %v, got: %v", expectedNominatedPods, q.PodNominator)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &medPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Pod.Name)
	}
	// List of nominated pods shouldn't change after popping them from the queue.
	if !reflect.DeepEqual(q.PodNominator, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after popping pods. Expected: %v, got: %v", expectedNominatedPods, q.PodNominator)
	}
	// Update one of the nominated pods that doesn't have nominatedNodeName in the
	// pod object. It should be updated correctly.
	q.AddNominatedPod(&highPriorityPod, "node4")
	expectedNominatedPods = &nominatedPodMap{
		nominatedPodToNode: map[types.UID]string{
			medPriorityPod.UID:   "node1",
			highPriorityPod.UID:  "node4",
			unschedulablePod.UID: "node5",
		},
		nominatedPods: map[string][]*v1.Pod{
			"node1": {&medPriorityPod},
			"node4": {&highPriorityPod},
			"node5": {&unschedulablePod},
		},
	}
	if !reflect.DeepEqual(q.PodNominator, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after updating pods. Expected: %v, got: %v", expectedNominatedPods, q.PodNominator)
	}

	// Delete a nominated pod that doesn't have nominatedNodeName in the pod
	// object. It should be deleted.
	q.DeleteNominatedPodIfExists(&highPriorityPod)
	expectedNominatedPods = &nominatedPodMap{
		nominatedPodToNode: map[types.UID]string{
			medPriorityPod.UID:   "node1",
			unschedulablePod.UID: "node5",
		},
		nominatedPods: map[string][]*v1.Pod{
			"node1": {&medPriorityPod},
			"node5": {&unschedulablePod},
		},
	}
	if !reflect.DeepEqual(q.PodNominator, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after deleting pods. Expected: %v, got: %v", expectedNominatedPods, q.PodNominator)
	}
}

func TestPriorityQueue_NewWithOptions(t *testing.T) {
	q := NewPriorityQueue(
		newDefaultQueueSort(),
		WithPodInitialBackoffDuration(2*time.Second),
		WithPodMaxBackoffDuration(20*time.Second),
	)

	if q.podInitialBackoffDuration != 2*time.Second {
		t.Errorf("Unexpected pod backoff initial duration. Expected: %v, got: %v", 2*time.Second, q.podInitialBackoffDuration)
	}

	if q.podMaxBackoffDuration != 20*time.Second {
		t.Errorf("Unexpected pod backoff max duration. Expected: %v, got: %v", 2*time.Second, q.podMaxBackoffDuration)
	}
}

func TestUnschedulablePodsMap(t *testing.T) {
	var pods = []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "p0",
				Namespace: "ns1",
				Annotations: map[string]string{
					"annot1": "val1",
				},
			},
			Status: v1.PodStatus{
				NominatedNodeName: "node1",
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "p1",
				Namespace: "ns1",
				Annotations: map[string]string{
					"annot": "val",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "p2",
				Namespace: "ns2",
				Annotations: map[string]string{
					"annot2": "val2", "annot3": "val3",
				},
			},
			Status: v1.PodStatus{
				NominatedNodeName: "node3",
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "p3",
				Namespace: "ns4",
			},
			Status: v1.PodStatus{
				NominatedNodeName: "node1",
			},
		},
	}
	var updatedPods = make([]*v1.Pod, len(pods))
	updatedPods[0] = pods[0].DeepCopy()
	updatedPods[1] = pods[1].DeepCopy()
	updatedPods[3] = pods[3].DeepCopy()

	tests := []struct {
		name                   string
		podsToAdd              []*v1.Pod
		expectedMapAfterAdd    map[string]*framework.QueuedPodInfo
		podsToUpdate           []*v1.Pod
		expectedMapAfterUpdate map[string]*framework.QueuedPodInfo
		podsToDelete           []*v1.Pod
		expectedMapAfterDelete map[string]*framework.QueuedPodInfo
	}{
		{
			name:      "create, update, delete subset of pods",
			podsToAdd: []*v1.Pod{pods[0], pods[1], pods[2], pods[3]},
			expectedMapAfterAdd: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[0]): {Pod: pods[0]},
				util.GetPodFullName(pods[1]): {Pod: pods[1]},
				util.GetPodFullName(pods[2]): {Pod: pods[2]},
				util.GetPodFullName(pods[3]): {Pod: pods[3]},
			},
			podsToUpdate: []*v1.Pod{updatedPods[0]},
			expectedMapAfterUpdate: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[0]): {Pod: updatedPods[0]},
				util.GetPodFullName(pods[1]): {Pod: pods[1]},
				util.GetPodFullName(pods[2]): {Pod: pods[2]},
				util.GetPodFullName(pods[3]): {Pod: pods[3]},
			},
			podsToDelete: []*v1.Pod{pods[0], pods[1]},
			expectedMapAfterDelete: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[2]): {Pod: pods[2]},
				util.GetPodFullName(pods[3]): {Pod: pods[3]},
			},
		},
		{
			name:      "create, update, delete all",
			podsToAdd: []*v1.Pod{pods[0], pods[3]},
			expectedMapAfterAdd: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[0]): {Pod: pods[0]},
				util.GetPodFullName(pods[3]): {Pod: pods[3]},
			},
			podsToUpdate: []*v1.Pod{updatedPods[3]},
			expectedMapAfterUpdate: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[0]): {Pod: pods[0]},
				util.GetPodFullName(pods[3]): {Pod: updatedPods[3]},
			},
			podsToDelete:           []*v1.Pod{pods[0], pods[3]},
			expectedMapAfterDelete: map[string]*framework.QueuedPodInfo{},
		},
		{
			name:      "delete non-existing and existing pods",
			podsToAdd: []*v1.Pod{pods[1], pods[2]},
			expectedMapAfterAdd: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[1]): {Pod: pods[1]},
				util.GetPodFullName(pods[2]): {Pod: pods[2]},
			},
			podsToUpdate: []*v1.Pod{updatedPods[1]},
			expectedMapAfterUpdate: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[1]): {Pod: updatedPods[1]},
				util.GetPodFullName(pods[2]): {Pod: pods[2]},
			},
			podsToDelete: []*v1.Pod{pods[2], pods[3]},
			expectedMapAfterDelete: map[string]*framework.QueuedPodInfo{
				util.GetPodFullName(pods[1]): {Pod: updatedPods[1]},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upm := newUnschedulablePodsMap(nil)
			for _, p := range test.podsToAdd {
				upm.addOrUpdate(newQueuedPodInfoNoTimestamp(p))
			}
			if !reflect.DeepEqual(upm.podInfoMap, test.expectedMapAfterAdd) {
				t.Errorf("Unexpected map after adding pods. Expected: %v, got: %v",
					test.expectedMapAfterAdd, upm.podInfoMap)
			}

			if len(test.podsToUpdate) > 0 {
				for _, p := range test.podsToUpdate {
					upm.addOrUpdate(newQueuedPodInfoNoTimestamp(p))
				}
				if !reflect.DeepEqual(upm.podInfoMap, test.expectedMapAfterUpdate) {
					t.Errorf("Unexpected map after updating pods. Expected: %v, got: %v",
						test.expectedMapAfterUpdate, upm.podInfoMap)
				}
			}
			for _, p := range test.podsToDelete {
				upm.delete(p)
			}
			if !reflect.DeepEqual(upm.podInfoMap, test.expectedMapAfterDelete) {
				t.Errorf("Unexpected map after deleting pods. Expected: %v, got: %v",
					test.expectedMapAfterDelete, upm.podInfoMap)
			}
			upm.clear()
			if len(upm.podInfoMap) != 0 {
				t.Errorf("Expected the map to be empty, but has %v elements.", len(upm.podInfoMap))
			}
		})
	}
}

func TestSchedulingQueue_Close(t *testing.T) {
	q := NewPriorityQueue(newDefaultQueueSort())
	wantErr := fmt.Errorf(queueClosed)
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		pod, err := q.Pop()
		if err.Error() != wantErr.Error() {
			t.Errorf("Expected err %q from Pop() if queue is closed, but got %q", wantErr.Error(), err.Error())
		}
		if pod != nil {
			t.Errorf("Expected pod nil from Pop() if queue is closed, but got: %v", pod)
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
	c := clock.NewFakeClock(time.Now())
	q := NewPriorityQueue(newDefaultQueueSort(), WithClock(c))
	// Add a few pods to priority queue.
	for i := 0; i < 5; i++ {
		p := v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("test-pod-%v", i),
				Namespace: "ns1",
				UID:       types.UID(fmt.Sprintf("tp00%v", i)),
			},
			Spec: v1.PodSpec{
				Priority: &highPriority,
			},
			Status: v1.PodStatus{
				NominatedNodeName: "node1",
			},
		}
		q.Add(&p)
	}
	c.Step(time.Microsecond)
	// Simulate a pod being popped by the scheduler, determined unschedulable, and
	// then moved back to the active queue.
	p1, err := q.Pop()
	if err != nil {
		t.Errorf("Error while popping the head of the queue: %v", err)
	}
	// Update pod condition to unschedulable.
	podutil.UpdatePodCondition(&p1.Pod.Status, &v1.PodCondition{
		Type:          v1.PodScheduled,
		Status:        v1.ConditionFalse,
		Reason:        v1.PodReasonUnschedulable,
		Message:       "fake scheduling failure",
		LastProbeTime: metav1.Now(),
	})
	// Put in the unschedulable queue.
	q.AddUnschedulableIfNotPresent(p1, q.SchedulingCycle())
	c.Step(DefaultPodInitialBackoffDuration)
	// Move all unschedulable pods to the active queue.
	q.MoveAllToActiveOrBackoffQueue("test")
	// Simulation is over. Now let's pop all pods. The pod popped first should be
	// the last one we pop here.
	for i := 0; i < 5; i++ {
		p, err := q.Pop()
		if err != nil {
			t.Errorf("Error while popping pods from the queue: %v", err)
		}
		if (i == 4) != (p1 == p) {
			t.Errorf("A pod tried before is not the last pod popped: i: %v, pod name: %v", i, p.Pod.Name)
		}
	}
}

// TestPodFailedSchedulingMultipleTimesDoesNotBlockNewerPod tests
// that a pod determined as unschedulable multiple times doesn't block any newer pod.
// This behavior ensures that an unschedulable pod does not block head of the queue when there
// are frequent events that move pods to the active queue.
func TestPodFailedSchedulingMultipleTimesDoesNotBlockNewerPod(t *testing.T) {
	c := clock.NewFakeClock(time.Now())
	q := NewPriorityQueue(newDefaultQueueSort(), WithClock(c))

	// Add an unschedulable pod to a priority queue.
	// This makes a situation that the pod was tried to schedule
	// and had been determined unschedulable so far
	unschedulablePod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod-unscheduled",
			Namespace: "ns1",
			UID:       "tp001",
		},
		Spec: v1.PodSpec{
			Priority: &highPriority,
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node1",
		},
	}

	// Update pod condition to unschedulable.
	podutil.UpdatePodCondition(&unschedulablePod.Status, &v1.PodCondition{
		Type:    v1.PodScheduled,
		Status:  v1.ConditionFalse,
		Reason:  v1.PodReasonUnschedulable,
		Message: "fake scheduling failure",
	})

	// Put in the unschedulable queue
	q.AddUnschedulableIfNotPresent(newQueuedPodInfoNoTimestamp(&unschedulablePod), q.SchedulingCycle())
	// Move clock to make the unschedulable pods complete backoff.
	c.Step(DefaultPodInitialBackoffDuration + time.Second)
	// Move all unschedulable pods to the active queue.
	q.MoveAllToActiveOrBackoffQueue("test")

	// Simulate a pod being popped by the scheduler,
	// At this time, unschedulable pod should be popped.
	p1, err := q.Pop()
	if err != nil {
		t.Errorf("Error while popping the head of the queue: %v", err)
	}
	if p1.Pod != &unschedulablePod {
		t.Errorf("Expected that test-pod-unscheduled was popped, got %v", p1.Pod.Name)
	}

	// Assume newer pod was added just after unschedulable pod
	// being popped and before being pushed back to the queue.
	newerPod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test-newer-pod",
			Namespace:         "ns1",
			UID:               "tp002",
			CreationTimestamp: metav1.Now(),
		},
		Spec: v1.PodSpec{
			Priority: &highPriority,
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node1",
		},
	}
	q.Add(&newerPod)

	// And then unschedulablePod was determined as unschedulable AGAIN.
	podutil.UpdatePodCondition(&unschedulablePod.Status, &v1.PodCondition{
		Type:    v1.PodScheduled,
		Status:  v1.ConditionFalse,
		Reason:  v1.PodReasonUnschedulable,
		Message: "fake scheduling failure",
	})

	// And then, put unschedulable pod to the unschedulable queue
	q.AddUnschedulableIfNotPresent(newQueuedPodInfoNoTimestamp(&unschedulablePod), q.SchedulingCycle())
	// Move clock to make the unschedulable pods complete backoff.
	c.Step(DefaultPodInitialBackoffDuration + time.Second)
	// Move all unschedulable pods to the active queue.
	q.MoveAllToActiveOrBackoffQueue("test")

	// At this time, newerPod should be popped
	// because it is the oldest tried pod.
	p2, err2 := q.Pop()
	if err2 != nil {
		t.Errorf("Error while popping the head of the queue: %v", err2)
	}
	if p2.Pod != &newerPod {
		t.Errorf("Expected that test-newer-pod was popped, got %v", p2.Pod.Name)
	}
}

// TestHighPriorityBackoff tests that a high priority pod does not block
// other pods if it is unschedulable
func TestHighPriorityBackoff(t *testing.T) {
	q := NewPriorityQueue(newDefaultQueueSort())

	midPod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-midpod",
			Namespace: "ns1",
			UID:       types.UID("tp-mid"),
		},
		Spec: v1.PodSpec{
			Priority: &midPriority,
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node1",
		},
	}
	highPod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-highpod",
			Namespace: "ns1",
			UID:       types.UID("tp-high"),
		},
		Spec: v1.PodSpec{
			Priority: &highPriority,
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node1",
		},
	}
	q.Add(&midPod)
	q.Add(&highPod)
	// Simulate a pod being popped by the scheduler, determined unschedulable, and
	// then moved back to the active queue.
	p, err := q.Pop()
	if err != nil {
		t.Errorf("Error while popping the head of the queue: %v", err)
	}
	if p.Pod != &highPod {
		t.Errorf("Expected to get high priority pod, got: %v", p)
	}
	// Update pod condition to unschedulable.
	podutil.UpdatePodCondition(&p.Pod.Status, &v1.PodCondition{
		Type:    v1.PodScheduled,
		Status:  v1.ConditionFalse,
		Reason:  v1.PodReasonUnschedulable,
		Message: "fake scheduling failure",
	})
	// Put in the unschedulable queue.
	q.AddUnschedulableIfNotPresent(p, q.SchedulingCycle())
	// Move all unschedulable pods to the active queue.
	q.MoveAllToActiveOrBackoffQueue("test")

	p, err = q.Pop()
	if err != nil {
		t.Errorf("Error while popping the head of the queue: %v", err)
	}
	if p.Pod != &midPod {
		t.Errorf("Expected to get mid priority pod, got: %v", p)
	}
}

// TestHighPriorityFlushUnschedulableQLeftover tests that pods will be moved to
// activeQ after one minutes if it is in unschedulableQ
func TestHighPriorityFlushUnschedulableQLeftover(t *testing.T) {
	c := clock.NewFakeClock(time.Now())
	q := NewPriorityQueue(newDefaultQueueSort(), WithClock(c))
	midPod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-midpod",
			Namespace: "ns1",
			UID:       types.UID("tp-mid"),
		},
		Spec: v1.PodSpec{
			Priority: &midPriority,
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node1",
		},
	}
	highPod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-highpod",
			Namespace: "ns1",
			UID:       types.UID("tp-high"),
		},
		Spec: v1.PodSpec{
			Priority: &highPriority,
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node1",
		},
	}

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

	q.AddUnschedulableIfNotPresent(q.newQueuedPodInfo(&highPod), q.SchedulingCycle())
	q.AddUnschedulableIfNotPresent(q.newQueuedPodInfo(&midPod), q.SchedulingCycle())
	c.Step(unschedulableQTimeInterval + time.Second)
	q.flushUnschedulableQLeftover()

	if p, err := q.Pop(); err != nil || p.Pod != &highPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Pod.Name)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &midPod {
		t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Pod.Name)
	}
}

type operation func(queue *PriorityQueue, pInfo *framework.QueuedPodInfo)

var (
	add = func(queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		queue.Add(pInfo.Pod)
	}
	addUnschedulablePodBackToUnschedulableQ = func(queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		queue.AddUnschedulableIfNotPresent(pInfo, 0)
	}
	addUnschedulablePodBackToBackoffQ = func(queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		queue.AddUnschedulableIfNotPresent(pInfo, -1)
	}
	addPodActiveQ = func(queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		queue.activeQ.Add(pInfo)
	}
	updatePodActiveQ = func(queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		queue.activeQ.Update(pInfo)
	}
	addPodUnschedulableQ = func(queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		// Update pod condition to unschedulable.
		podutil.UpdatePodCondition(&pInfo.Pod.Status, &v1.PodCondition{
			Type:    v1.PodScheduled,
			Status:  v1.ConditionFalse,
			Reason:  v1.PodReasonUnschedulable,
			Message: "fake scheduling failure",
		})
		queue.unschedulableQ.addOrUpdate(pInfo)
	}
	addPodBackoffQ = func(queue *PriorityQueue, pInfo *framework.QueuedPodInfo) {
		queue.podBackoffQ.Add(pInfo)
	}
	moveAllToActiveOrBackoffQ = func(queue *PriorityQueue, _ *framework.QueuedPodInfo) {
		queue.MoveAllToActiveOrBackoffQueue("test")
	}
	flushBackoffQ = func(queue *PriorityQueue, _ *framework.QueuedPodInfo) {
		queue.clock.(*clock.FakeClock).Step(2 * time.Second)
		queue.flushBackoffQCompleted()
	}
	moveClockForward = func(queue *PriorityQueue, _ *framework.QueuedPodInfo) {
		queue.clock.(*clock.FakeClock).Step(2 * time.Second)
	}
)

// TestPodTimestamp tests the operations related to QueuedPodInfo.
func TestPodTimestamp(t *testing.T) {
	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod-1",
			Namespace: "ns1",
			UID:       types.UID("tp-1"),
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node1",
		},
	}

	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod-2",
			Namespace: "ns2",
			UID:       types.UID("tp-2"),
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node2",
		},
	}

	var timestamp = time.Now()
	pInfo1 := &framework.QueuedPodInfo{
		Pod:       pod1,
		Timestamp: timestamp,
	}
	pInfo2 := &framework.QueuedPodInfo{
		Pod:       pod2,
		Timestamp: timestamp.Add(time.Second),
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
				addPodActiveQ,
				addPodActiveQ,
			},
			operands: []*framework.QueuedPodInfo{pInfo2, pInfo1},
			expected: []*framework.QueuedPodInfo{pInfo1, pInfo2},
		},
		{
			name: "update two pod to activeQ and sort them by the timestamp",
			operations: []operation{
				updatePodActiveQ,
				updatePodActiveQ,
			},
			operands: []*framework.QueuedPodInfo{pInfo2, pInfo1},
			expected: []*framework.QueuedPodInfo{pInfo1, pInfo2},
		},
		{
			name: "add two pod to unschedulableQ then move them to activeQ and sort them by the timestamp",
			operations: []operation{
				addPodUnschedulableQ,
				addPodUnschedulableQ,
				moveClockForward,
				moveAllToActiveOrBackoffQ,
			},
			operands: []*framework.QueuedPodInfo{pInfo2, pInfo1, nil, nil},
			expected: []*framework.QueuedPodInfo{pInfo1, pInfo2},
		},
		{
			name: "add one pod to BackoffQ and move it to activeQ",
			operations: []operation{
				addPodActiveQ,
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
			queue := NewPriorityQueue(newDefaultQueueSort(), WithClock(clock.NewFakeClock(timestamp)))
			var podInfoList []*framework.QueuedPodInfo

			for i, op := range test.operations {
				op(queue, test.operands[i])
			}

			for i := 0; i < len(test.expected); i++ {
				if pInfo, err := queue.activeQ.Pop(); err != nil {
					t.Errorf("Error while popping the head of the queue: %v", err)
				} else {
					podInfoList = append(podInfoList, pInfo.(*framework.QueuedPodInfo))
				}
			}

			if !reflect.DeepEqual(test.expected, podInfoList) {
				t.Errorf("Unexpected QueuedPodInfo list. Expected: %v, got: %v",
					test.expected, podInfoList)
			}
		})
	}
}

// TestPendingPodsMetric tests Prometheus metrics related with pending pods
func TestPendingPodsMetric(t *testing.T) {
	timestamp := time.Now()
	metrics.Register()
	total := 50
	pInfos := makeQueuedPodInfos(total, timestamp)
	totalWithDelay := 20
	pInfosWithDelay := makeQueuedPodInfos(totalWithDelay, timestamp.Add(2*time.Second))

	tests := []struct {
		name        string
		operations  []operation
		operands    [][]*framework.QueuedPodInfo
		metricsName string
		wants       string
	}{
		{
			name: "add pods to activeQ and unschedulableQ",
			operations: []operation{
				addPodActiveQ,
				addPodUnschedulableQ,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:30],
				pInfos[30:],
			},
			metricsName: "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [ALPHA] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableQ.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 30
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="unschedulable"} 20
`,
		},
		{
			name: "add pods to all kinds of queues",
			operations: []operation{
				addPodActiveQ,
				addPodBackoffQ,
				addPodUnschedulableQ,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:15],
				pInfos[15:40],
				pInfos[40:],
			},
			metricsName: "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [ALPHA] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableQ.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 15
scheduler_pending_pods{queue="backoff"} 25
scheduler_pending_pods{queue="unschedulable"} 10
`,
		},
		{
			name: "add pods to unschedulableQ and then move all to activeQ",
			operations: []operation{
				addPodUnschedulableQ,
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
# HELP scheduler_pending_pods [ALPHA] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableQ.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 50
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
		{
			name: "make some pods subject to backoff, add pods to unschedulableQ, and then move all to activeQ",
			operations: []operation{
				addPodUnschedulableQ,
				moveClockForward,
				addPodUnschedulableQ,
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
# HELP scheduler_pending_pods [ALPHA] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableQ.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 30
scheduler_pending_pods{queue="backoff"} 20
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
		{
			name: "make some pods subject to backoff, add pods to unschedulableQ/activeQ, move all to activeQ, and finally flush backoffQ",
			operations: []operation{
				addPodUnschedulableQ,
				addPodActiveQ,
				moveAllToActiveOrBackoffQ,
				flushBackoffQ,
			},
			operands: [][]*framework.QueuedPodInfo{
				pInfos[:40],
				pInfos[40:],
				{nil},
				{nil},
			},
			metricsName: "scheduler_pending_pods",
			wants: `
# HELP scheduler_pending_pods [ALPHA] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableQ.
# TYPE scheduler_pending_pods gauge
scheduler_pending_pods{queue="active"} 50
scheduler_pending_pods{queue="backoff"} 0
scheduler_pending_pods{queue="unschedulable"} 0
`,
		},
	}

	resetMetrics := func() {
		metrics.ActivePods().Set(0)
		metrics.BackoffPods().Set(0)
		metrics.UnschedulablePods().Set(0)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			queue := NewPriorityQueue(newDefaultQueueSort(), WithClock(clock.NewFakeClock(timestamp)))
			for i, op := range test.operations {
				for _, pInfo := range test.operands[i] {
					op(queue, pInfo)
				}
			}

			if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(test.wants), test.metricsName); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// TestPerPodSchedulingMetrics makes sure pod schedule attempts is updated correctly while
// initialAttemptTimestamp stays the same during multiple add/pop operations.
func TestPerPodSchedulingMetrics(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
			UID:       types.UID("test-uid"),
		},
	}
	timestamp := time.Now()

	// Case 1: A pod is created and scheduled after 1 attempt. The queue operations are
	// Add -> Pop.
	c := clock.NewFakeClock(timestamp)
	queue := NewPriorityQueue(newDefaultQueueSort(), WithClock(c))
	queue.Add(pod)
	pInfo, err := queue.Pop()
	if err != nil {
		t.Fatalf("Failed to pop a pod %v", err)
	}
	checkPerPodSchedulingMetrics("Attempt once", t, pInfo, 1, timestamp)

	// Case 2: A pod is created and scheduled after 2 attempts. The queue operations are
	// Add -> Pop -> AddUnschedulableIfNotPresent -> flushUnschedulableQLeftover -> Pop.
	c = clock.NewFakeClock(timestamp)
	queue = NewPriorityQueue(newDefaultQueueSort(), WithClock(c))
	queue.Add(pod)
	pInfo, err = queue.Pop()
	if err != nil {
		t.Fatalf("Failed to pop a pod %v", err)
	}
	queue.AddUnschedulableIfNotPresent(pInfo, 1)
	// Override clock to exceed the unschedulableQTimeInterval so that unschedulable pods
	// will be moved to activeQ
	c.SetTime(timestamp.Add(unschedulableQTimeInterval + 1))
	queue.flushUnschedulableQLeftover()
	pInfo, err = queue.Pop()
	if err != nil {
		t.Fatalf("Failed to pop a pod %v", err)
	}
	checkPerPodSchedulingMetrics("Attempt twice", t, pInfo, 2, timestamp)

	// Case 3: Similar to case 2, but before the second pop, call update, the queue operations are
	// Add -> Pop -> AddUnschedulableIfNotPresent -> flushUnschedulableQLeftover -> Update -> Pop.
	c = clock.NewFakeClock(timestamp)
	queue = NewPriorityQueue(newDefaultQueueSort(), WithClock(c))
	queue.Add(pod)
	pInfo, err = queue.Pop()
	if err != nil {
		t.Fatalf("Failed to pop a pod %v", err)
	}
	queue.AddUnschedulableIfNotPresent(pInfo, 1)
	// Override clock to exceed the unschedulableQTimeInterval so that unschedulable pods
	// will be moved to activeQ
	c.SetTime(timestamp.Add(unschedulableQTimeInterval + 1))
	queue.flushUnschedulableQLeftover()
	newPod := pod.DeepCopy()
	newPod.Generation = 1
	queue.Update(pod, newPod)
	pInfo, err = queue.Pop()
	if err != nil {
		t.Fatalf("Failed to pop a pod %v", err)
	}
	checkPerPodSchedulingMetrics("Attempt twice with update", t, pInfo, 2, timestamp)
}

func TestIncomingPodsMetrics(t *testing.T) {
	timestamp := time.Now()
	metrics.Register()
	var pInfos = make([]*framework.QueuedPodInfo, 0, 3)
	for i := 1; i <= 3; i++ {
		p := &framework.QueuedPodInfo{
			Pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("test-pod-%d", i),
					Namespace: fmt.Sprintf("ns%d", i),
					UID:       types.UID(fmt.Sprintf("tp-%d", i)),
				},
			},
			Timestamp: timestamp,
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
            scheduler_queue_incoming_pods_total{event="PodAdd",queue="active"} 3
`,
		},
		{
			name: "add pods to unschedulableQ",
			operations: []operation{
				addUnschedulablePodBackToUnschedulableQ,
			},
			want: `
             scheduler_queue_incoming_pods_total{event="ScheduleAttemptFailure",queue="unschedulable"} 3
`,
		},
		{
			name: "add pods to unschedulableQ and then move all to backoffQ",
			operations: []operation{
				addUnschedulablePodBackToUnschedulableQ,
				moveAllToActiveOrBackoffQ,
			},
			want: ` scheduler_queue_incoming_pods_total{event="ScheduleAttemptFailure",queue="unschedulable"} 3
            scheduler_queue_incoming_pods_total{event="test",queue="backoff"} 3
`,
		},
		{
			name: "add pods to unschedulableQ and then move all to activeQ",
			operations: []operation{
				addUnschedulablePodBackToUnschedulableQ,
				moveClockForward,
				moveAllToActiveOrBackoffQ,
			},
			want: ` scheduler_queue_incoming_pods_total{event="ScheduleAttemptFailure",queue="unschedulable"} 3
            scheduler_queue_incoming_pods_total{event="test",queue="active"} 3
`,
		},
		{
			name: "make some pods subject to backoff and add them to backoffQ, then flush backoffQ",
			operations: []operation{
				addUnschedulablePodBackToBackoffQ,
				moveClockForward,
				flushBackoffQ,
			},
			want: ` scheduler_queue_incoming_pods_total{event="BackoffComplete",queue="active"} 3
            scheduler_queue_incoming_pods_total{event="ScheduleAttemptFailure",queue="backoff"} 3
`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			metrics.SchedulerQueueIncomingPods.Reset()
			queue := NewPriorityQueue(newDefaultQueueSort(), WithClock(clock.NewFakeClock(timestamp)))
			for _, op := range test.operations {
				for _, pInfo := range pInfos {
					op(queue, pInfo)
				}
			}
			metricName := metrics.SchedulerSubsystem + "_" + metrics.SchedulerQueueIncomingPods.Name
			if err := testutil.CollectAndCompare(metrics.SchedulerQueueIncomingPods, strings.NewReader(queueMetricMetadata+test.want), metricName); err != nil {
				t.Errorf("unexpected collecting result:\n%s", err)
			}

		})
	}
}

func checkPerPodSchedulingMetrics(name string, t *testing.T, pInfo *framework.QueuedPodInfo, wantAttemtps int, wantInitialAttemptTs time.Time) {
	if pInfo.Attempts != wantAttemtps {
		t.Errorf("[%s] Pod schedule attempt unexpected, got %v, want %v", name, pInfo.Attempts, wantAttemtps)
	}
	if pInfo.InitialAttemptTimestamp != wantInitialAttemptTs {
		t.Errorf("[%s] Pod initial schedule attempt timestamp unexpected, got %v, want %v", name, pInfo.InitialAttemptTimestamp, wantInitialAttemptTs)
	}
}

func TestBackOffFlow(t *testing.T) {
	cl := clock.NewFakeClock(time.Now())
	q := NewPriorityQueue(newDefaultQueueSort(), WithClock(cl))
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
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
			UID:       "test-uid",
		},
	}
	podID := nsNameForPod(pod)
	if err := q.Add(pod); err != nil {
		t.Fatal(err)
	}

	for i, step := range steps {
		t.Run(fmt.Sprintf("step %d", i), func(t *testing.T) {
			timestamp := cl.Now()
			// Simulate schedule attempt.
			podInfo, err := q.Pop()
			if err != nil {
				t.Fatal(err)
			}
			if podInfo.Attempts != i+1 {
				t.Errorf("got attempts %d, want %d", podInfo.Attempts, i+1)
			}
			if err := q.AddUnschedulableIfNotPresent(podInfo, int64(i)); err != nil {
				t.Fatal(err)
			}

			// An event happens.
			q.MoveAllToActiveOrBackoffQueue("deleted pod")

			if _, ok, _ := q.podBackoffQ.Get(podInfo); !ok {
				t.Errorf("pod %v is not in the backoff queue", podID)
			}

			// Check backoff duration.
			deadline := q.getBackoffTime(podInfo)
			backoff := deadline.Sub(timestamp)
			if backoff != step.wantBackoff {
				t.Errorf("got backoff %s, want %s", backoff, step.wantBackoff)
			}

			// Simulate routine that continuously flushes the backoff queue.
			cl.Step(time.Millisecond)
			q.flushBackoffQCompleted()
			// Still in backoff queue after an early flush.
			if _, ok, _ := q.podBackoffQ.Get(podInfo); !ok {
				t.Errorf("pod %v is not in the backoff queue", podID)
			}
			// Moved out of the backoff queue after timeout.
			cl.Step(backoff)
			q.flushBackoffQCompleted()
			if _, ok, _ := q.podBackoffQ.Get(podInfo); ok {
				t.Errorf("pod %v is still in the backoff queue", podID)
			}
		})
	}
}

func makeQueuedPodInfos(num int, timestamp time.Time) []*framework.QueuedPodInfo {
	var pInfos = make([]*framework.QueuedPodInfo, 0, num)
	for i := 1; i <= num; i++ {
		p := &framework.QueuedPodInfo{
			Pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("test-pod-%d", i),
					Namespace: fmt.Sprintf("ns%d", i),
					UID:       types.UID(fmt.Sprintf("tp-%d", i)),
				},
			},
			Timestamp: timestamp,
		}
		pInfos = append(pInfos, p)
	}
	return pInfos
}
