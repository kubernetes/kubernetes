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
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

var negPriority, lowPriority, midPriority, highPriority, veryHighPriority = int32(-100), int32(0), int32(100), int32(1000), int32(10000)
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

func addOrUpdateUnschedulablePod(p *PriorityQueue, pod *v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.unschedulableQ.addOrUpdate(pod)
}

func getUnschedulablePod(p *PriorityQueue, pod *v1.Pod) *v1.Pod {
	p.lock.Lock()
	defer p.lock.Unlock()
	return p.unschedulableQ.get(pod)
}

func TestPriorityQueue_Add(t *testing.T) {
	q := NewPriorityQueue(nil)
	q.Add(&medPriorityPod)
	q.Add(&unschedulablePod)
	q.Add(&highPriorityPod)
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
	if p, err := q.Pop(); err != nil || p != &highPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Name)
	}
	if p, err := q.Pop(); err != nil || p != &medPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Name)
	}
	if p, err := q.Pop(); err != nil || p != &unschedulablePod {
		t.Errorf("Expected: %v after Pop, but got: %v", unschedulablePod.Name, p.Name)
	}
	if len(q.nominatedPods.nominatedPods["node1"]) != 2 {
		t.Errorf("Expected medPriorityPod and unschedulablePod to be still present in nomindatePods: %v", q.nominatedPods.nominatedPods["node1"])
	}
}

func TestPriorityQueue_AddIfNotPresent(t *testing.T) {
	q := NewPriorityQueue(nil)
	addOrUpdateUnschedulablePod(q, &highPriNominatedPod)
	q.AddIfNotPresent(&highPriNominatedPod) // Must not add anything.
	q.AddIfNotPresent(&medPriorityPod)
	q.AddIfNotPresent(&unschedulablePod)
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
	if p, err := q.Pop(); err != nil || p != &medPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Name)
	}
	if p, err := q.Pop(); err != nil || p != &unschedulablePod {
		t.Errorf("Expected: %v after Pop, but got: %v", unschedulablePod.Name, p.Name)
	}
	if len(q.nominatedPods.nominatedPods["node1"]) != 2 {
		t.Errorf("Expected medPriorityPod and unschedulablePod to be still present in nomindatePods: %v", q.nominatedPods.nominatedPods["node1"])
	}
	if getUnschedulablePod(q, &highPriNominatedPod) != &highPriNominatedPod {
		t.Errorf("Pod %v was not found in the unschedulableQ.", highPriNominatedPod.Name)
	}
}

func TestPriorityQueue_AddUnschedulableIfNotPresent(t *testing.T) {
	q := NewPriorityQueue(nil)
	q.Add(&highPriNominatedPod)
	q.AddUnschedulableIfNotPresent(&highPriNominatedPod, q.SchedulingCycle()) // Must not add anything.
	q.AddUnschedulableIfNotPresent(&medPriorityPod, q.SchedulingCycle())      // This should go to activeQ.
	q.AddUnschedulableIfNotPresent(&unschedulablePod, q.SchedulingCycle())
	expectedNominatedPods := &nominatedPodMap{
		nominatedPodToNode: map[types.UID]string{
			medPriorityPod.UID:      "node1",
			unschedulablePod.UID:    "node1",
			highPriNominatedPod.UID: "node1",
		},
		nominatedPods: map[string][]*v1.Pod{
			"node1": {&highPriNominatedPod, &medPriorityPod, &unschedulablePod},
		},
	}
	if !reflect.DeepEqual(q.nominatedPods, expectedNominatedPods) {
		t.Errorf("Unexpected nominated map after adding pods. Expected: %v, got: %v", expectedNominatedPods, q.nominatedPods)
	}
	if p, err := q.Pop(); err != nil || p != &highPriNominatedPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriNominatedPod.Name, p.Name)
	}
	if p, err := q.Pop(); err != nil || p != &medPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Name)
	}
	if len(q.nominatedPods.nominatedPods) != 1 {
		t.Errorf("Expected nomindatePods to have one element: %v", q.nominatedPods)
	}
	if getUnschedulablePod(q, &unschedulablePod) != &unschedulablePod {
		t.Errorf("Pod %v was not found in the unschedulableQ.", unschedulablePod.Name)
	}
}

// TestPriorityQueue_AddUnschedulableIfNotPresent_Async tests scenario when
// AddUnschedulableIfNotPresent is called asynchronously pods in and before
// current scheduling cycle will be put back to activeQueue if we were trying
// to schedule them when we received move request.
func TestPriorityQueue_AddUnschedulableIfNotPresent_Async(t *testing.T) {
	q := NewPriorityQueue(nil)
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
		if !reflect.DeepEqual(&expectedPods[i], p) {
			t.Errorf("Unexpected pod. Expected: %v, got: %v", &expectedPods[i], p)
		}
	}

	// move all pods to active queue when we were trying to schedule them
	q.MoveAllToActiveQueue()
	moveReqChan := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(totalNum - 1)
	// mark pods[1] ~ pods[totalNum-1] as unschedulable, fire goroutines to add them back later
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
		cycle := q.SchedulingCycle()
		go func() {
			<-moveReqChan
			q.AddUnschedulableIfNotPresent(unschedulablePod, cycle)
			wg.Done()
		}()
	}
	firstPod, _ := q.Pop()
	if !reflect.DeepEqual(&expectedPods[0], firstPod) {
		t.Errorf("Unexpected pod. Expected: %v, got: %v", &expectedPods[0], firstPod)
	}
	// close moveReqChan here to make sure q.AddUnschedulableIfNotPresent is called after another pod is popped
	close(moveReqChan)
	wg.Wait()
	// all other pods should be in active queue again
	for i := 1; i < totalNum; i++ {
		if _, exists, _ := q.activeQ.Get(&expectedPods[i]); !exists {
			t.Errorf("Expected %v to be added to activeQ.", expectedPods[i].Name)
		}
	}
}

func TestPriorityQueue_Pop(t *testing.T) {
	q := NewPriorityQueue(nil)
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		if p, err := q.Pop(); err != nil || p != &medPriorityPod {
			t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Name)
		}
		if len(q.nominatedPods.nominatedPods["node1"]) != 1 {
			t.Errorf("Expected medPriorityPod to be present in nomindatePods: %v", q.nominatedPods.nominatedPods["node1"])
		}
	}()
	q.Add(&medPriorityPod)
	wg.Wait()
}

func TestPriorityQueue_Update(t *testing.T) {
	q := NewPriorityQueue(nil)
	q.Update(nil, &highPriorityPod)
	if _, exists, _ := q.activeQ.Get(&highPriorityPod); !exists {
		t.Errorf("Expected %v to be added to activeQ.", highPriorityPod.Name)
	}
	if len(q.nominatedPods.nominatedPods) != 0 {
		t.Errorf("Expected nomindatePods to be empty: %v", q.nominatedPods)
	}
	// Update highPriorityPod and add a nominatedNodeName to it.
	q.Update(&highPriorityPod, &highPriNominatedPod)
	if q.activeQ.data.Len() != 1 {
		t.Error("Expected only one item in activeQ.")
	}
	if len(q.nominatedPods.nominatedPods) != 1 {
		t.Errorf("Expected one item in nomindatePods map: %v", q.nominatedPods)
	}
	// Updating an unschedulable pod which is not in any of the two queues, should
	// add the pod to activeQ.
	q.Update(&unschedulablePod, &unschedulablePod)
	if _, exists, _ := q.activeQ.Get(&unschedulablePod); !exists {
		t.Errorf("Expected %v to be added to activeQ.", unschedulablePod.Name)
	}
	// Updating a pod that is already in activeQ, should not change it.
	q.Update(&unschedulablePod, &unschedulablePod)
	if len(q.unschedulableQ.pods) != 0 {
		t.Error("Expected unschedulableQ to be empty.")
	}
	if _, exists, _ := q.activeQ.Get(&unschedulablePod); !exists {
		t.Errorf("Expected: %v to be added to activeQ.", unschedulablePod.Name)
	}
	if p, err := q.Pop(); err != nil || p != &highPriNominatedPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Name)
	}
}

func TestPriorityQueue_Delete(t *testing.T) {
	q := NewPriorityQueue(nil)
	q.Update(&highPriorityPod, &highPriNominatedPod)
	q.Add(&unschedulablePod)
	q.Delete(&highPriNominatedPod)
	if _, exists, _ := q.activeQ.Get(&unschedulablePod); !exists {
		t.Errorf("Expected %v to be in activeQ.", unschedulablePod.Name)
	}
	if _, exists, _ := q.activeQ.Get(&highPriNominatedPod); exists {
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
	q := NewPriorityQueue(nil)
	q.Add(&medPriorityPod)
	addOrUpdateUnschedulablePod(q, &unschedulablePod)
	addOrUpdateUnschedulablePod(q, &highPriorityPod)
	q.MoveAllToActiveQueue()
	if q.activeQ.data.Len() != 3 {
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

	q := NewPriorityQueue(nil)
	q.Add(&medPriorityPod)
	// Add a couple of pods to the unschedulableQ.
	addOrUpdateUnschedulablePod(q, &unschedulablePod)
	addOrUpdateUnschedulablePod(q, affinityPod)
	// Simulate addition of an assigned pod. The pod has matching labels for
	// affinityPod. So, affinityPod should go to activeQ.
	q.AssignedPodAdded(&labelPod)
	if getUnschedulablePod(q, affinityPod) != nil {
		t.Error("affinityPod is still in the unschedulableQ.")
	}
	if _, exists, _ := q.activeQ.Get(affinityPod); !exists {
		t.Error("affinityPod is not moved to activeQ.")
	}
	// Check that the other pod is still in the unschedulableQ.
	if getUnschedulablePod(q, &unschedulablePod) == nil {
		t.Error("unschedulablePod is not in the unschedulableQ.")
	}
}

func TestPriorityQueue_NominatedPodsForNode(t *testing.T) {
	q := NewPriorityQueue(nil)
	q.Add(&medPriorityPod)
	q.Add(&unschedulablePod)
	q.Add(&highPriorityPod)
	if p, err := q.Pop(); err != nil || p != &highPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Name)
	}
	expectedList := []*v1.Pod{&medPriorityPod, &unschedulablePod}
	if !reflect.DeepEqual(expectedList, q.NominatedPodsForNode("node1")) {
		t.Error("Unexpected list of nominated Pods for node.")
	}
	if q.NominatedPodsForNode("node2") != nil {
		t.Error("Expected list of nominated Pods for node2 to be empty.")
	}
}

func TestPriorityQueue_UpdateNominatedPodForNode(t *testing.T) {
	q := NewPriorityQueue(nil)
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
	if p, err := q.Pop(); err != nil || p != &medPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Name)
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
		expectedMapAfterAdd    map[string]*v1.Pod
		podsToUpdate           []*v1.Pod
		expectedMapAfterUpdate map[string]*v1.Pod
		podsToDelete           []*v1.Pod
		expectedMapAfterDelete map[string]*v1.Pod
	}{
		{
			name:      "create, update, delete subset of pods",
			podsToAdd: []*v1.Pod{pods[0], pods[1], pods[2], pods[3]},
			expectedMapAfterAdd: map[string]*v1.Pod{
				util.GetPodFullName(pods[0]): pods[0],
				util.GetPodFullName(pods[1]): pods[1],
				util.GetPodFullName(pods[2]): pods[2],
				util.GetPodFullName(pods[3]): pods[3],
			},
			podsToUpdate: []*v1.Pod{updatedPods[0]},
			expectedMapAfterUpdate: map[string]*v1.Pod{
				util.GetPodFullName(pods[0]): updatedPods[0],
				util.GetPodFullName(pods[1]): pods[1],
				util.GetPodFullName(pods[2]): pods[2],
				util.GetPodFullName(pods[3]): pods[3],
			},
			podsToDelete: []*v1.Pod{pods[0], pods[1]},
			expectedMapAfterDelete: map[string]*v1.Pod{
				util.GetPodFullName(pods[2]): pods[2],
				util.GetPodFullName(pods[3]): pods[3],
			},
		},
		{
			name:      "create, update, delete all",
			podsToAdd: []*v1.Pod{pods[0], pods[3]},
			expectedMapAfterAdd: map[string]*v1.Pod{
				util.GetPodFullName(pods[0]): pods[0],
				util.GetPodFullName(pods[3]): pods[3],
			},
			podsToUpdate: []*v1.Pod{updatedPods[3]},
			expectedMapAfterUpdate: map[string]*v1.Pod{
				util.GetPodFullName(pods[0]): pods[0],
				util.GetPodFullName(pods[3]): updatedPods[3],
			},
			podsToDelete:           []*v1.Pod{pods[0], pods[3]},
			expectedMapAfterDelete: map[string]*v1.Pod{},
		},
		{
			name:      "delete non-existing and existing pods",
			podsToAdd: []*v1.Pod{pods[1], pods[2]},
			expectedMapAfterAdd: map[string]*v1.Pod{
				util.GetPodFullName(pods[1]): pods[1],
				util.GetPodFullName(pods[2]): pods[2],
			},
			podsToUpdate: []*v1.Pod{updatedPods[1]},
			expectedMapAfterUpdate: map[string]*v1.Pod{
				util.GetPodFullName(pods[1]): updatedPods[1],
				util.GetPodFullName(pods[2]): pods[2],
			},
			podsToDelete: []*v1.Pod{pods[2], pods[3]},
			expectedMapAfterDelete: map[string]*v1.Pod{
				util.GetPodFullName(pods[1]): updatedPods[1],
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			upm := newUnschedulablePodsMap()
			for _, p := range test.podsToAdd {
				upm.addOrUpdate(p)
			}
			if !reflect.DeepEqual(upm.pods, test.expectedMapAfterAdd) {
				t.Errorf("Unexpected map after adding pods. Expected: %v, got: %v",
					test.expectedMapAfterAdd, upm.pods)
			}

			if len(test.podsToUpdate) > 0 {
				for _, p := range test.podsToUpdate {
					upm.addOrUpdate(p)
				}
				if !reflect.DeepEqual(upm.pods, test.expectedMapAfterUpdate) {
					t.Errorf("Unexpected map after updating pods. Expected: %v, got: %v",
						test.expectedMapAfterUpdate, upm.pods)
				}
			}
			for _, p := range test.podsToDelete {
				upm.delete(p)
			}
			if !reflect.DeepEqual(upm.pods, test.expectedMapAfterDelete) {
				t.Errorf("Unexpected map after deleting pods. Expected: %v, got: %v",
					test.expectedMapAfterDelete, upm.pods)
			}
			upm.clear()
			if len(upm.pods) != 0 {
				t.Errorf("Expected the map to be empty, but has %v elements.", len(upm.pods))
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
			name:        "FIFO close",
			q:           NewFIFO(),
			expectedErr: fmt.Errorf(queueClosed),
		},
		{
			name:        "PriorityQueue close",
			q:           NewPriorityQueue(nil),
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
	q := NewPriorityQueue(nil)
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
	podutil.UpdatePodCondition(&p1.Status, &v1.PodCondition{
		Type:          v1.PodScheduled,
		Status:        v1.ConditionFalse,
		Reason:        v1.PodReasonUnschedulable,
		Message:       "fake scheduling failure",
		LastProbeTime: metav1.Now(),
	})
	// Put in the unschedulable queue.
	q.AddUnschedulableIfNotPresent(p1, q.SchedulingCycle())
	// Move all unschedulable pods to the active queue.
	q.MoveAllToActiveQueue()
	// Simulation is over. Now let's pop all pods. The pod popped first should be
	// the last one we pop here.
	for i := 0; i < 5; i++ {
		p, err := q.Pop()
		if err != nil {
			t.Errorf("Error while popping pods from the queue: %v", err)
		}
		if (i == 4) != (p1 == p) {
			t.Errorf("A pod tried before is not the last pod popped: i: %v, pod name: %v", i, p.Name)
		}
	}
}

// TestPodFailedSchedulingMultipleTimesDoesNotBlockNewerPod tests
// that a pod determined as unschedulable multiple times doesn't block any newer pod.
// This behavior ensures that an unschedulable pod does not block head of the queue when there
// are frequent events that move pods to the active queue.
func TestPodFailedSchedulingMultipleTimesDoesNotBlockNewerPod(t *testing.T) {
	q := NewPriorityQueue(nil)

	// Add an unschedulable pod to a priority queue.
	// This makes a situation that the pod was tried to schedule
	// and had been determined unschedulable so far.
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
		Type:          v1.PodScheduled,
		Status:        v1.ConditionFalse,
		Reason:        v1.PodReasonUnschedulable,
		Message:       "fake scheduling failure",
		LastProbeTime: metav1.Now(),
	})
	// Put in the unschedulable queue
	q.AddUnschedulableIfNotPresent(&unschedulablePod, q.SchedulingCycle())
	// Move all unschedulable pods to the active queue.
	q.MoveAllToActiveQueue()

	// Simulate a pod being popped by the scheduler,
	// At this time, unschedulable pod should be popped.
	p1, err := q.Pop()
	if err != nil {
		t.Errorf("Error while popping the head of the queue: %v", err)
	}
	if p1 != &unschedulablePod {
		t.Errorf("Expected that test-pod-unscheduled was popped, got %v", p1.Name)
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
		Type:          v1.PodScheduled,
		Status:        v1.ConditionFalse,
		Reason:        v1.PodReasonUnschedulable,
		Message:       "fake scheduling failure",
		LastProbeTime: metav1.Now(),
	})
	// And then, put unschedulable pod to the unschedulable queue
	q.AddUnschedulableIfNotPresent(&unschedulablePod, q.SchedulingCycle())
	// Move all unschedulable pods to the active queue.
	q.MoveAllToActiveQueue()

	// At this time, newerPod should be popped
	// because it is the oldest tried pod.
	p2, err2 := q.Pop()
	if err2 != nil {
		t.Errorf("Error while popping the head of the queue: %v", err2)
	}
	if p2 != &newerPod {
		t.Errorf("Expected that test-newer-pod was popped, got %v", p2.Name)
	}
}

// TestHighProirotyFlushUnschedulableQLeftover tests that pods will be moved to
// activeQ after one minutes if it is in unschedulableQ
func TestHighProirotyFlushUnschedulableQLeftover(t *testing.T) {
	q := NewPriorityQueue(nil)
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

	addOrUpdateUnschedulablePod(q, &highPod)
	addOrUpdateUnschedulablePod(q, &midPod)

	// Update pod condition to highPod.
	podutil.UpdatePodCondition(&highPod.Status, &v1.PodCondition{
		Type:          v1.PodScheduled,
		Status:        v1.ConditionFalse,
		Reason:        v1.PodReasonUnschedulable,
		Message:       "fake scheduling failure",
		LastProbeTime: metav1.Time{Time: time.Now().Add(-1 * unschedulableQTimeInterval)},
	})

	// Update pod condition to midPod.
	podutil.UpdatePodCondition(&midPod.Status, &v1.PodCondition{
		Type:          v1.PodScheduled,
		Status:        v1.ConditionFalse,
		Reason:        v1.PodReasonUnschedulable,
		Message:       "fake scheduling failure",
		LastProbeTime: metav1.Time{Time: time.Now().Add(-1 * unschedulableQTimeInterval)},
	})

	if p, err := q.Pop(); err != nil || p != &highPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Name)
	}
	if p, err := q.Pop(); err != nil || p != &midPod {
		t.Errorf("Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Name)
	}
}
