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
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/component-base/metrics/testutil"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	frameworkplugins "k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
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

func addOrUpdateUnschedulablePod(p *PriorityQueue, podInfo *framework.PodInfo) {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.unschedulableQ.addOrUpdate(podInfo)
}

func getUnschedulablePod(p *PriorityQueue, pod *v1.Pod) *v1.Pod {
	p.lock.Lock()
	defer p.lock.Unlock()
	pInfo := p.unschedulableQ.get(pod)
	if pInfo != nil {
		return pInfo.Pod
	}
	return nil
}

func TestPriorityQueue_Add(t *testing.T) {
	q := createAndRunPriorityQueue(newDefaultFramework())
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
	if !reflect.DeepEqual(q.nominatedPods, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after adding pods. Expected: %v, got: %v", expectedNominatedPods, q.nominatedPods)
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
	if len(q.nominatedPods.nominatedPods["node1"]) != 2 {
		t.Errorf("Expected medPriorityPod and unschedulablePod to be still present in nomindatePods: %v", q.nominatedPods.nominatedPods["node1"])
	}
}

func newDefaultFramework() framework.Framework {
	defaultCfg := algorithmprovider.NewRegistry(1)[schedulerapi.SchedulerDefaultProviderName]
	pl, pls := defaultCfg.FrameworkPlugins, defaultCfg.FrameworkPluginConfig
	fakeClient := fake.NewSimpleClientset()
	fwk, err := framework.NewFramework(
		frameworkplugins.NewInTreeRegistry(),
		pl,
		pls,
		framework.WithClientSet(fakeClient),
		framework.WithInformerFactory(informers.NewSharedInformerFactory(fakeClient, 0)),
		framework.WithSnapshotSharedLister(cache.NewEmptySnapshot()),
	)
	if err != nil {
		panic(err)
	}
	return fwk
}

func TestPriorityQueue_AddWithReversePriorityLessFunc(t *testing.T) {
	q := createAndRunPriorityQueue(newDefaultFramework())
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
	q := createAndRunPriorityQueue(newDefaultFramework())
	q.Add(&highPriNominatedPod)
	q.AddUnschedulableIfNotPresent(newPodInfoNoTimestamp(&highPriNominatedPod), q.SchedulingCycle()) // Must not add anything.
	q.AddUnschedulableIfNotPresent(newPodInfoNoTimestamp(&unschedulablePod), q.SchedulingCycle())
	expectedNominatedPods := &nominatedPodMap{
		nominatedPodToNode: map[types.UID]string{
			unschedulablePod.UID:    "node1",
			highPriNominatedPod.UID: "node1",
		},
		nominatedPods: map[string][]*v1.Pod{
			"node1": {&highPriNominatedPod, &unschedulablePod},
		},
	}
	if !reflect.DeepEqual(q.nominatedPods, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after adding pods. Expected: %v, got: %v", expectedNominatedPods, q.nominatedPods)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &highPriNominatedPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriNominatedPod.Name, p.Pod.Name)
	}
	if len(q.nominatedPods.nominatedPods) != 1 {
		t.Errorf("Expected nomindatePods to have one element: %v", q.nominatedPods)
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
	q := createAndRunPriorityQueue(newDefaultFramework(), WithClock(clock.NewFakeClock(time.Now())))
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

		if err := q.AddUnschedulableIfNotPresent(newPodInfoNoTimestamp(unschedulablePod), oldCycle); err != nil {
			t.Errorf("Failed to call AddUnschedulableIfNotPresent(%v): %v", unschedulablePod.Name, err)
		}
	}

	q.lock.RLock()
	// Since there was a move request at the same cycle as "oldCycle", these pods
	// should be in the backoff queue.
	for i := 1; i < totalNum; i++ {
		if _, exists, _ := q.podBackoffQ.Get(newPodInfoNoTimestamp(&expectedPods[i])); !exists {
			t.Errorf("Expected %v to be added to podBackoffQ.", expectedPods[i].Name)
		}
	}
	q.lock.RUnlock()
}

func TestPriorityQueue_Pop(t *testing.T) {
	q := createAndRunPriorityQueue(newDefaultFramework())
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		if p, err := q.Pop(); err != nil || p.Pod != &medPriorityPod {
			t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Pod.Name)
		}
		if len(q.nominatedPods.nominatedPods["node1"]) != 1 {
			t.Errorf("Expected medPriorityPod to be present in nomindatePods: %v", q.nominatedPods.nominatedPods["node1"])
		}
	}()
	q.Add(&medPriorityPod)
	wg.Wait()
}

func TestPriorityQueue_Update(t *testing.T) {
	q := createAndRunPriorityQueue(newDefaultFramework())
	q.Update(nil, &highPriorityPod)
	if _, exists, _ := q.activeQ.Get(newPodInfoNoTimestamp(&highPriorityPod)); !exists {
		t.Errorf("Expected %v to be added to activeQ.", highPriorityPod.Name)
	}
	if len(q.nominatedPods.nominatedPods) != 0 {
		t.Errorf("Expected nomindatePods to be empty: %v", q.nominatedPods)
	}
	// Update highPriorityPod and add a nominatedNodeName to it.
	q.Update(&highPriorityPod, &highPriNominatedPod)
	if q.activeQ.Len() != 1 {
		t.Error("Expected only one item in activeQ.")
	}
	if len(q.nominatedPods.nominatedPods) != 1 {
		t.Errorf("Expected one item in nomindatePods map: %v", q.nominatedPods)
	}
	// Updating an unschedulable pod which is not in any of the two queues, should
	// add the pod to activeQ.
	q.Update(&unschedulablePod, &unschedulablePod)
	if _, exists, _ := q.activeQ.Get(newPodInfoNoTimestamp(&unschedulablePod)); !exists {
		t.Errorf("Expected %v to be added to activeQ.", unschedulablePod.Name)
	}
	// Updating a pod that is already in activeQ, should not change it.
	q.Update(&unschedulablePod, &unschedulablePod)
	if len(q.unschedulableQ.podInfoMap) != 0 {
		t.Error("Expected unschedulableQ to be empty.")
	}
	if _, exists, _ := q.activeQ.Get(newPodInfoNoTimestamp(&unschedulablePod)); !exists {
		t.Errorf("Expected: %v to be added to activeQ.", unschedulablePod.Name)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &highPriNominatedPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Pod.Name)
	}
}

func TestPriorityQueue_Delete(t *testing.T) {
	q := createAndRunPriorityQueue(newDefaultFramework())
	q.Update(&highPriorityPod, &highPriNominatedPod)
	q.Add(&unschedulablePod)
	if err := q.Delete(&highPriNominatedPod); err != nil {
		t.Errorf("delete failed: %v", err)
	}
	if _, exists, _ := q.activeQ.Get(newPodInfoNoTimestamp(&unschedulablePod)); !exists {
		t.Errorf("Expected %v to be in activeQ.", unschedulablePod.Name)
	}
	if _, exists, _ := q.activeQ.Get(newPodInfoNoTimestamp(&highPriNominatedPod)); exists {
		t.Errorf("Didn't expect %v to be in activeQ.", highPriorityPod.Name)
	}
	if len(q.nominatedPods.nominatedPods) != 1 {
		t.Errorf("Expected nomindatePods to have only 'unschedulablePod': %v", q.nominatedPods.nominatedPods)
	}
	if err := q.Delete(&unschedulablePod); err != nil {
		t.Errorf("delete failed: %v", err)
	}
	if len(q.nominatedPods.nominatedPods) != 0 {
		t.Errorf("Expected nomindatePods to be empty: %v", q.nominatedPods)
	}
}

func TestPriorityQueue_MoveAllToActiveQueue(t *testing.T) {
	q := createAndRunPriorityQueue(newDefaultFramework())
	q.Add(&medPriorityPod)
	addOrUpdateUnschedulablePod(q, q.newPodInfo(&unschedulablePod))
	addOrUpdateUnschedulablePod(q, q.newPodInfo(&highPriorityPod))
	q.MoveAllToActiveOrBackoffQueue("test")
	if q.activeQ.Len() != 3 {
		t.Error("Expected all items to be in activeQ.")
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

	q := createAndRunPriorityQueue(newDefaultFramework())
	q.Add(&medPriorityPod)
	// Add a couple of pods to the unschedulableQ.
	addOrUpdateUnschedulablePod(q, q.newPodInfo(&unschedulablePod))
	addOrUpdateUnschedulablePod(q, q.newPodInfo(affinityPod))
	// Simulate addition of an assigned pod. The pod has matching labels for
	// affinityPod. So, affinityPod should go to activeQ.
	q.AssignedPodAdded(&labelPod)
	if getUnschedulablePod(q, affinityPod) != nil {
		t.Error("affinityPod is still in the unschedulableQ.")
	}
	if _, exists, _ := q.activeQ.Get(newPodInfoNoTimestamp(affinityPod)); !exists {
		t.Error("affinityPod is not moved to activeQ.")
	}
	// Check that the other pod is still in the unschedulableQ.
	if getUnschedulablePod(q, &unschedulablePod) == nil {
		t.Error("unschedulablePod is not in the unschedulableQ.")
	}
}

func TestPriorityQueue_NominatedPodsForNode(t *testing.T) {
	q := createAndRunPriorityQueue(newDefaultFramework())
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

	q := createAndRunPriorityQueue(newDefaultFramework())
	q.Add(&medPriorityPod)
	addOrUpdateUnschedulablePod(q, q.newPodInfo(&unschedulablePod))
	addOrUpdateUnschedulablePod(q, q.newPodInfo(&highPriorityPod))
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
	q := createAndRunPriorityQueue(newDefaultFramework())
	if err := q.Add(&medPriorityPod); err != nil {
		t.Errorf("add failed: %v", err)
	}
	// Update unschedulablePod on a different node than specified in the pod.
	q.UpdateNominatedPodForNode(&unschedulablePod, "node5")

	// Update nominated node name of a pod on a node that is not specified in the pod object.
	q.UpdateNominatedPodForNode(&highPriorityPod, "node2")
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
	if !reflect.DeepEqual(q.nominatedPods, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after adding pods. Expected: %v, got: %v", expectedNominatedPods, q.nominatedPods)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &medPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Pod.Name)
	}
	// List of nominated pods shouldn't change after popping them from the queue.
	if !reflect.DeepEqual(q.nominatedPods, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after popping pods. Expected: %v, got: %v", expectedNominatedPods, q.nominatedPods)
	}
	// Update one of the nominated pods that doesn't have nominatedNodeName in the
	// pod object. It should be updated correctly.
	q.UpdateNominatedPodForNode(&highPriorityPod, "node4")
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
	if !reflect.DeepEqual(q.nominatedPods, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after updating pods. Expected: %v, got: %v", expectedNominatedPods, q.nominatedPods)
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
	if !reflect.DeepEqual(q.nominatedPods, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after deleting pods. Expected: %v, got: %v", expectedNominatedPods, q.nominatedPods)
	}
}

func TestPriorityQueue_NewWithOptions(t *testing.T) {
	q := createAndRunPriorityQueue(
		newDefaultFramework(),
		WithPodInitialBackoffDuration(2*time.Second),
		WithPodMaxBackoffDuration(20*time.Second),
	)

	if q.podBackoff.initialDuration != 2*time.Second {
		t.Errorf("Unexpected pod backoff initial duration. Expected: %v, got: %v", 2*time.Second, q.podBackoff.initialDuration)
	}

	if q.podBackoff.maxDuration != 20*time.Second {
		t.Errorf("Unexpected pod backoff max duration. Expected: %v, got: %v", 2*time.Second, q.podBackoff.maxDuration)
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
		expectedMapAfterAdd    map[string]*framework.PodInfo
		podsToUpdate           []*v1.Pod
		expectedMapAfterUpdate map[string]*framework.PodInfo
		podsToDelete           []*v1.Pod
		expectedMapAfterDelete map[string]*framework.PodInfo
	}{
		{
			name:      "create, update, delete subset of pods",
			podsToAdd: []*v1.Pod{pods[0], pods[1], pods[2], pods[3]},
			expectedMapAfterAdd: map[string]*framework.PodInfo{
				util.GetPodFullName(pods[0]): {Pod: pods[0]},
				util.GetPodFullName(pods[1]): {Pod: pods[1]},
				util.GetPodFullName(pods[2]): {Pod: pods[2]},
				util.GetPodFullName(pods[3]): {Pod: pods[3]},
			},
			podsToUpdate: []*v1.Pod{updatedPods[0]},
			expectedMapAfterUpdate: map[string]*framework.PodInfo{
				util.GetPodFullName(pods[0]): {Pod: updatedPods[0]},
				util.GetPodFullName(pods[1]): {Pod: pods[1]},
				util.GetPodFullName(pods[2]): {Pod: pods[2]},
				util.GetPodFullName(pods[3]): {Pod: pods[3]},
			},
			podsToDelete: []*v1.Pod{pods[0], pods[1]},
			expectedMapAfterDelete: map[string]*framework.PodInfo{
				util.GetPodFullName(pods[2]): {Pod: pods[2]},
				util.GetPodFullName(pods[3]): {Pod: pods[3]},
			},
		},
		{
			name:      "create, update, delete all",
			podsToAdd: []*v1.Pod{pods[0], pods[3]},
			expectedMapAfterAdd: map[string]*framework.PodInfo{
				util.GetPodFullName(pods[0]): {Pod: pods[0]},
				util.GetPodFullName(pods[3]): {Pod: pods[3]},
			},
			podsToUpdate: []*v1.Pod{updatedPods[3]},
			expectedMapAfterUpdate: map[string]*framework.PodInfo{
				util.GetPodFullName(pods[0]): {Pod: pods[0]},
				util.GetPodFullName(pods[3]): {Pod: updatedPods[3]},
			},
			podsToDelete:           []*v1.Pod{pods[0], pods[3]},
			expectedMapAfterDelete: map[string]*framework.PodInfo{},
		},
		{
			name:      "delete non-existing and existing pods",
			podsToAdd: []*v1.Pod{pods[1], pods[2]},
			expectedMapAfterAdd: map[string]*framework.PodInfo{
				util.GetPodFullName(pods[1]): {Pod: pods[1]},
				util.GetPodFullName(pods[2]): {Pod: pods[2]},
			},
			podsToUpdate: []*v1.Pod{updatedPods[1]},
			expectedMapAfterUpdate: map[string]*framework.PodInfo{
				util.GetPodFullName(pods[1]): {Pod: updatedPods[1]},
				util.GetPodFullName(pods[2]): {Pod: pods[2]},
			},
			podsToDelete: []*v1.Pod{pods[2], pods[3]},
			expectedMapAfterDelete: map[string]*framework.PodInfo{
				util.GetPodFullName(pods[1]): {Pod: updatedPods[1]},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upm := newUnschedulablePodsMap(nil)
			for _, p := range test.podsToAdd {
				upm.addOrUpdate(newPodInfoNoTimestamp(p))
			}
			if !reflect.DeepEqual(upm.podInfoMap, test.expectedMapAfterAdd) {
				t.Errorf("Unexpected map after adding pods. Expected: %v, got: %v",
					test.expectedMapAfterAdd, upm.podInfoMap)
			}

			if len(test.podsToUpdate) > 0 {
				for _, p := range test.podsToUpdate {
					upm.addOrUpdate(newPodInfoNoTimestamp(p))
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
	tests := []struct {
		name        string
		q           SchedulingQueue
		expectedErr error
	}{
		{
			name:        "PriorityQueue close",
			q:           createAndRunPriorityQueue(newDefaultFramework()),
			expectedErr: fmt.Errorf(queueClosed),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			wg := sync.WaitGroup{}
			wg.Add(1)
			go func() {
				defer wg.Done()
				pod, err := test.q.Pop()
				if err.Error() != test.expectedErr.Error() {
					t.Errorf("Expected err %q from Pop() if queue is closed, but got %q", test.expectedErr.Error(), err.Error())
				}
				if pod != nil {
					t.Errorf("Expected pod nil from Pop() if queue is closed, but got: %v", pod)
				}
			}()
			test.q.Close()
			wg.Wait()
		})
	}
}

// TestRecentlyTriedPodsGoBack tests that pods which are recently tried and are
// unschedulable go behind other pods with the same priority. This behavior
// ensures that an unschedulable pod does not block head of the queue when there
// are frequent events that move pods to the active queue.
func TestRecentlyTriedPodsGoBack(t *testing.T) {
	q := createAndRunPriorityQueue(newDefaultFramework())
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
	q := createAndRunPriorityQueue(newDefaultFramework())

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
	q.AddUnschedulableIfNotPresent(newPodInfoNoTimestamp(&unschedulablePod), q.SchedulingCycle())
	// Clear its backoff to simulate backoff its expiration
	q.clearPodBackoff(&unschedulablePod)
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
	q.AddUnschedulableIfNotPresent(newPodInfoNoTimestamp(&unschedulablePod), q.SchedulingCycle())
	// Clear its backoff to simulate its backoff expiration
	q.clearPodBackoff(&unschedulablePod)
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
	q := createAndRunPriorityQueue(newDefaultFramework())

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
	q := createAndRunPriorityQueue(newDefaultFramework())
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

	highPodInfo := q.newPodInfo(&highPod)
	highPodInfo.Timestamp = time.Now().Add(-1 * unschedulableQTimeInterval)
	midPodInfo := q.newPodInfo(&midPod)
	midPodInfo.Timestamp = time.Now().Add(-1 * unschedulableQTimeInterval)
	addOrUpdateUnschedulablePod(q, highPodInfo)
	addOrUpdateUnschedulablePod(q, midPodInfo)

	if p, err := q.Pop(); err != nil || p.Pod != &highPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Pod.Name)
	}
	if p, err := q.Pop(); err != nil || p.Pod != &midPod {
		t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Pod.Name)
	}
}

type operation func(queue *PriorityQueue, pInfo *framework.PodInfo)

var (
	add = func(queue *PriorityQueue, pInfo *framework.PodInfo) {
		queue.Add(pInfo.Pod)
	}
	addUnschedulablePodBackToUnschedulableQ = func(queue *PriorityQueue, pInfo *framework.PodInfo) {
		queue.AddUnschedulableIfNotPresent(pInfo, 0)
	}
	addUnschedulablePodBackToBackoffQ = func(queue *PriorityQueue, pInfo *framework.PodInfo) {
		queue.AddUnschedulableIfNotPresent(pInfo, -1)
	}
	addPodActiveQ = func(queue *PriorityQueue, pInfo *framework.PodInfo) {
		queue.lock.Lock()
		queue.activeQ.Add(pInfo)
		queue.lock.Unlock()
	}
	updatePodActiveQ = func(queue *PriorityQueue, pInfo *framework.PodInfo) {
		queue.lock.Lock()
		queue.activeQ.Update(pInfo)
		queue.lock.Unlock()
	}
	addPodUnschedulableQ = func(queue *PriorityQueue, pInfo *framework.PodInfo) {
		queue.lock.Lock()
		// Update pod condition to unschedulable.
		podutil.UpdatePodCondition(&pInfo.Pod.Status, &v1.PodCondition{
			Type:    v1.PodScheduled,
			Status:  v1.ConditionFalse,
			Reason:  v1.PodReasonUnschedulable,
			Message: "fake scheduling failure",
		})
		queue.unschedulableQ.addOrUpdate(pInfo)
		queue.lock.Unlock()
	}
	addPodBackoffQ = func(queue *PriorityQueue, pInfo *framework.PodInfo) {
		queue.lock.Lock()
		queue.podBackoffQ.Add(pInfo)
		queue.lock.Unlock()
	}
	moveAllToActiveOrBackoffQ = func(queue *PriorityQueue, _ *framework.PodInfo) {
		queue.MoveAllToActiveOrBackoffQueue("test")
	}
	backoffPod = func(queue *PriorityQueue, pInfo *framework.PodInfo) {
		queue.backoffPod(pInfo.Pod)
	}
	flushBackoffQ = func(queue *PriorityQueue, _ *framework.PodInfo) {
		queue.clock.(*clock.FakeClock).Step(2 * time.Second)
		queue.flushBackoffQCompleted()
	}
	moveClockForward = func(queue *PriorityQueue, _ *framework.PodInfo) {
		queue.clock.(*clock.FakeClock).Step(2 * time.Second)
	}
)

// TestPodTimestamp tests the operations related to PodInfo.
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
	pInfo1 := &framework.PodInfo{
		Pod:       pod1,
		Timestamp: timestamp,
	}
	pInfo2 := &framework.PodInfo{
		Pod:       pod2,
		Timestamp: timestamp.Add(time.Second),
	}

	tests := []struct {
		name       string
		operations []operation
		operands   []*framework.PodInfo
		expected   []*framework.PodInfo
	}{
		{
			name: "add two pod to activeQ and sort them by the timestamp",
			operations: []operation{
				addPodActiveQ,
				addPodActiveQ,
			},
			operands: []*framework.PodInfo{pInfo2, pInfo1},
			expected: []*framework.PodInfo{pInfo1, pInfo2},
		},
		{
			name: "update two pod to activeQ and sort them by the timestamp",
			operations: []operation{
				updatePodActiveQ,
				updatePodActiveQ,
			},
			operands: []*framework.PodInfo{pInfo2, pInfo1},
			expected: []*framework.PodInfo{pInfo1, pInfo2},
		},
		{
			name: "add two pod to unschedulableQ then move them to activeQ and sort them by the timestamp",
			operations: []operation{
				addPodUnschedulableQ,
				addPodUnschedulableQ,
				moveAllToActiveOrBackoffQ,
			},
			operands: []*framework.PodInfo{pInfo2, pInfo1, nil},
			expected: []*framework.PodInfo{pInfo1, pInfo2},
		},
		{
			name: "add one pod to BackoffQ and move it to activeQ",
			operations: []operation{
				addPodActiveQ,
				addPodBackoffQ,
				backoffPod,
				flushBackoffQ,
				moveAllToActiveOrBackoffQ,
			},
			operands: []*framework.PodInfo{pInfo2, pInfo1, pInfo1, nil, nil},
			expected: []*framework.PodInfo{pInfo1, pInfo2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			queue := createAndRunPriorityQueue(newDefaultFramework(), WithClock(clock.NewFakeClock(timestamp)))
			var podInfoList []*framework.PodInfo

			for i, op := range test.operations {
				op(queue, test.operands[i])
			}

			for i := 0; i < len(test.expected); i++ {
				if pInfo, err := queue.activeQ.Pop(); err != nil {
					t.Errorf("Error while popping the head of the queue: %v", err)
				} else {
					podInfoList = append(podInfoList, pInfo.(*framework.PodInfo))
				}
			}

			if !reflect.DeepEqual(test.expected, podInfoList) {
				t.Errorf("Unexpected PodInfo list. Expected: %v, got: %v",
					test.expected, podInfoList)
			}
		})
	}
}

// TestPendingPodsMetric tests Prometheus metrics related with pending pods
func TestPendingPodsMetric(t *testing.T) {
	total := 50
	timestamp := time.Now()
	metrics.Register()
	var pInfos = make([]*framework.PodInfo, 0, total)
	for i := 1; i <= total; i++ {
		p := &framework.PodInfo{
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
		name        string
		operations  []operation
		operands    [][]*framework.PodInfo
		metricsName string
		wants       string
	}{
		{
			name: "add pods to activeQ and unschedulableQ",
			operations: []operation{
				addPodActiveQ,
				addPodUnschedulableQ,
			},
			operands: [][]*framework.PodInfo{
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
				backoffPod,
				addPodBackoffQ,
				addPodUnschedulableQ,
			},
			operands: [][]*framework.PodInfo{
				pInfos[:15],
				pInfos[15:40],
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
				moveAllToActiveOrBackoffQ,
			},
			operands: [][]*framework.PodInfo{
				pInfos[:total],
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
				backoffPod,
				addPodUnschedulableQ,
				moveAllToActiveOrBackoffQ,
			},
			operands: [][]*framework.PodInfo{
				pInfos[:20],
				pInfos[:total],
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
				backoffPod,
				addPodUnschedulableQ,
				addPodActiveQ,
				moveAllToActiveOrBackoffQ,
				flushBackoffQ,
			},
			operands: [][]*framework.PodInfo{
				pInfos[:20],
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
			queue := createAndRunPriorityQueue(newDefaultFramework(), WithClock(clock.NewFakeClock(timestamp)))
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
	queue := createAndRunPriorityQueue(newDefaultFramework(), WithClock(c))
	queue.Add(pod)
	pInfo, err := queue.Pop()
	if err != nil {
		t.Fatalf("Failed to pop a pod %v", err)
	}
	checkPerPodSchedulingMetrics("Attempt once", t, pInfo, 1, timestamp)

	// Case 2: A pod is created and scheduled after 2 attempts. The queue operations are
	// Add -> Pop -> AddUnschedulableIfNotPresent -> flushUnschedulableQLeftover -> Pop.
	c = clock.NewFakeClock(timestamp)
	queue = createAndRunPriorityQueue(newDefaultFramework(), WithClock(c))
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
	queue = createAndRunPriorityQueue(newDefaultFramework(), WithClock(c))
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
	var pInfos = make([]*framework.PodInfo, 0, 3)
	for i := 1; i <= 3; i++ {
		p := &framework.PodInfo{
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
			queue := NewPriorityQueue(newDefaultFramework(), WithClock(clock.NewFakeClock(timestamp)))
			queue.Close()
			queue.Run()
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

func checkPerPodSchedulingMetrics(name string, t *testing.T, pInfo *framework.PodInfo, wantAttemtps int, wantInitialAttemptTs time.Time) {
	if pInfo.Attempts != wantAttemtps {
		t.Errorf("[%s] Pod schedule attempt unexpected, got %v, want %v", name, pInfo.Attempts, wantAttemtps)
	}
	if pInfo.InitialAttemptTimestamp != wantInitialAttemptTs {
		t.Errorf("[%s] Pod initial schedule attempt timestamp unexpected, got %v, want %v", name, pInfo.InitialAttemptTimestamp, wantInitialAttemptTs)
	}
}

func createAndRunPriorityQueue(fwk framework.Framework, opts ...Option) *PriorityQueue {
	q := NewPriorityQueue(fwk, opts...)
	q.Run()
	return q
}
