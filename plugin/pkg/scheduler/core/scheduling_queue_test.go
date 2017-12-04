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

package core

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler/util"
)

var mediumPriority = (lowPriority + highPriority) / 2
var highPriorityPod, medPriorityPod, unschedulablePod = v1.Pod{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "hpp",
		Namespace: "ns1",
	},
	Spec: v1.PodSpec{
		Priority: &highPriority,
	},
},
	v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mpp",
			Namespace: "ns2",
			Annotations: map[string]string{
				NominatedNodeAnnotationKey: "node1", "annot2": "val2",
			},
		},
		Spec: v1.PodSpec{
			Priority: &mediumPriority,
		},
	},
	v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "up",
			Namespace: "ns1",
			Annotations: map[string]string{
				NominatedNodeAnnotationKey: "node1", "annot2": "val2",
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
		},
	}

func TestPriorityQueue_Add(t *testing.T) {
	q := NewPriorityQueue()
	q.Add(&medPriorityPod)
	q.Add(&unschedulablePod)
	q.Add(&highPriorityPod)
	p, err := q.Pop()
	require.NoError(t, err, "Unexpected error: %v", err)
	require.Equal(t, &highPriorityPod, p, "Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Name)

	p, err = q.Pop()
	require.NoError(t, err, "Unexpected error: %v", err)
	require.Equal(t, &medPriorityPod, p, "Expected: %v after Pop, but got: %v", medPriorityPod.Name, p.Name)

	p, err = q.Pop()
	require.NoError(t, err, "Unexpected error: %v", err)
	require.Equal(t, &unschedulablePod, p, "Expected: %v after Pop, but got: %v", unschedulablePod.Name, p.Name)
}

func TestPriorityQueue_Pop(t *testing.T) {
	q := NewPriorityQueue()
	go func() {
		p, err := q.Pop()
		require.NoError(t, err, "Unexpected error: %v", err)
		require.Equal(t, &highPriorityPod, p, "Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Name)
	}()
	q.Add(&highPriorityPod)
}

func TestPriorityQueue_Update(t *testing.T) {
	q := NewPriorityQueue()
	q.Update(&highPriorityPod)
	if _, exists, _ := q.activeQ.Get(&highPriorityPod); !exists {
		t.Errorf("Expected %v to be added to activeQ.", highPriorityPod.Name)
	}
	q.Update(&highPriorityPod)
	require.Equal(t, 1, q.activeQ.data.Len(), "Expected only one item in activeQ.")
	// Updating an unschedulable pod which is not in any of the two queues, should
	// add the pod to activeQ.
	q.Update(&unschedulablePod)
	if _, exists, _ := q.activeQ.Get(&unschedulablePod); !exists {
		t.Errorf("Expected %v to be added to activeQ.", unschedulablePod.Name)
	}
	// Updating a pod that is already in unschedulableQ, should move the pod to
	// activeQ.
	q.Update(&unschedulablePod)
	if len(q.unschedulableQ.pods) != 0 {
		t.Error("Expected unschedulableQ to be empty.")
	}
	if _, exists, _ := q.activeQ.Get(&unschedulablePod); !exists {
		t.Errorf("Expected: %v to be added to activeQ.", unschedulablePod.Name)
	}
	if p, err := q.Pop(); err != nil || p != &highPriorityPod {
		t.Errorf("Expected: %v after Pop, but got: %v", highPriorityPod.Name, p.Name)
	}
}

func TestPriorityQueue_Delete(t *testing.T) {
	q := NewPriorityQueue()
	q.Update(&highPriorityPod)
	q.Add(&unschedulablePod)
	q.Delete(&highPriorityPod)
	_, exists, _ := q.activeQ.Get(&unschedulablePod)
	require.True(t, exists, "Expected %v to be in activeQ.", unschedulablePod.Name)

	_, exists, _ = q.activeQ.Get(&highPriorityPod)
	require.False(t, exists, "Didn't expect %v to be in activeQ.", highPriorityPod.Name)
}

func TestPriorityQueue_MoveAllToActiveQueue(t *testing.T) {
	q := NewPriorityQueue()
	q.Add(&medPriorityPod)
	q.unschedulableQ.Add(&unschedulablePod)
	q.unschedulableQ.Add(&highPriorityPod)
	q.MoveAllToActiveQueue()
	require.Equal(t, 3, q.activeQ.data.Len(), "Expected all items to be in activeQ.")
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

	q := NewPriorityQueue()
	q.Add(&medPriorityPod)
	// Add a couple of pods to the unschedulableQ.
	q.unschedulableQ.Add(&unschedulablePod)
	q.unschedulableQ.Add(affinityPod)
	// Simulate addition of an assigned pod. The pod has matching labels for
	// affinityPod. So, affinityPod should go to activeQ.
	q.AssignedPodAdded(&labelPod)
	require.Nil(t, q.unschedulableQ.Get(affinityPod), "affinityPod is still in the unschedulableQ.")
	_, exists, _ := q.activeQ.Get(affinityPod)
	require.True(t, exists, "affinityPod is not moved to activeQ.")
	// Check that the other pod is still in the unschedulableQ.
	require.NotNil(t, q.unschedulableQ.Get(&unschedulablePod), "unschedulablePod is not in the unschedulableQ.")
}

func TestUnschedulablePodsMap(t *testing.T) {
	var pods = []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "p0",
				Namespace: "ns1",
				Annotations: map[string]string{
					NominatedNodeAnnotationKey: "node1", "annot2": "val2",
				},
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
					NominatedNodeAnnotationKey: "node3", "annot2": "val2", "annot3": "val3",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "p3",
				Namespace: "ns4",
				Annotations: map[string]string{
					NominatedNodeAnnotationKey: "node1",
				},
			},
		},
	}
	var updatedPods = make([]*v1.Pod, len(pods))
	updatedPods[0] = pods[0].DeepCopy()
	updatedPods[0].Annotations[NominatedNodeAnnotationKey] = "node3"
	updatedPods[1] = pods[1].DeepCopy()
	updatedPods[1].Annotations[NominatedNodeAnnotationKey] = "node3"
	updatedPods[3] = pods[3].DeepCopy()
	delete(updatedPods[3].Annotations, NominatedNodeAnnotationKey)

	tests := []struct {
		podsToAdd                    []*v1.Pod
		expectedMapAfterAdd          map[string]*v1.Pod
		expectedNominatedAfterAdd    map[string][]string
		podsToUpdate                 []*v1.Pod
		expectedMapAfterUpdate       map[string]*v1.Pod
		expectedNominatedAfterUpdate map[string][]string
		podsToDelete                 []*v1.Pod
		expectedMapAfterDelete       map[string]*v1.Pod
		expectedNominatedAfterDelete map[string][]string
	}{
		{
			podsToAdd: []*v1.Pod{pods[0], pods[1], pods[2], pods[3]},
			expectedMapAfterAdd: map[string]*v1.Pod{
				util.GetPodFullName(pods[0]): pods[0],
				util.GetPodFullName(pods[1]): pods[1],
				util.GetPodFullName(pods[2]): pods[2],
				util.GetPodFullName(pods[3]): pods[3],
			},
			expectedNominatedAfterAdd: map[string][]string{
				"node1": {util.GetPodFullName(pods[0]), util.GetPodFullName(pods[3])},
				"node3": {util.GetPodFullName(pods[2])},
			},
			podsToUpdate: []*v1.Pod{updatedPods[0]},
			expectedMapAfterUpdate: map[string]*v1.Pod{
				util.GetPodFullName(pods[0]): updatedPods[0],
				util.GetPodFullName(pods[1]): pods[1],
				util.GetPodFullName(pods[2]): pods[2],
				util.GetPodFullName(pods[3]): pods[3],
			},
			expectedNominatedAfterUpdate: map[string][]string{
				"node1": {util.GetPodFullName(pods[3])},
				"node3": {util.GetPodFullName(pods[2]), util.GetPodFullName(pods[0])},
			},
			podsToDelete: []*v1.Pod{pods[0], pods[1]},
			expectedMapAfterDelete: map[string]*v1.Pod{
				util.GetPodFullName(pods[2]): pods[2],
				util.GetPodFullName(pods[3]): pods[3],
			},
			expectedNominatedAfterDelete: map[string][]string{
				"node1": {util.GetPodFullName(pods[3])},
				"node3": {util.GetPodFullName(pods[2])},
			},
		},
		{
			podsToAdd: []*v1.Pod{pods[0], pods[3]},
			expectedMapAfterAdd: map[string]*v1.Pod{
				util.GetPodFullName(pods[0]): pods[0],
				util.GetPodFullName(pods[3]): pods[3],
			},
			expectedNominatedAfterAdd: map[string][]string{
				"node1": {util.GetPodFullName(pods[0]), util.GetPodFullName(pods[3])},
			},
			podsToUpdate: []*v1.Pod{updatedPods[3]},
			expectedMapAfterUpdate: map[string]*v1.Pod{
				util.GetPodFullName(pods[0]): pods[0],
				util.GetPodFullName(pods[3]): updatedPods[3],
			},
			expectedNominatedAfterUpdate: map[string][]string{
				"node1": {util.GetPodFullName(pods[0])},
			},
			podsToDelete:                 []*v1.Pod{pods[0], pods[3]},
			expectedMapAfterDelete:       map[string]*v1.Pod{},
			expectedNominatedAfterDelete: map[string][]string{},
		},
		{
			podsToAdd: []*v1.Pod{pods[1], pods[2]},
			expectedMapAfterAdd: map[string]*v1.Pod{
				util.GetPodFullName(pods[1]): pods[1],
				util.GetPodFullName(pods[2]): pods[2],
			},
			expectedNominatedAfterAdd: map[string][]string{
				"node3": {util.GetPodFullName(pods[2])},
			},
			podsToUpdate: []*v1.Pod{updatedPods[1]},
			expectedMapAfterUpdate: map[string]*v1.Pod{
				util.GetPodFullName(pods[1]): updatedPods[1],
				util.GetPodFullName(pods[2]): pods[2],
			},
			expectedNominatedAfterUpdate: map[string][]string{
				"node3": {util.GetPodFullName(pods[2]), util.GetPodFullName(updatedPods[1])},
			},
			podsToDelete: []*v1.Pod{pods[2], pods[3]},
			expectedMapAfterDelete: map[string]*v1.Pod{
				util.GetPodFullName(pods[1]): updatedPods[1],
			},
			expectedNominatedAfterDelete: map[string][]string{
				"node3": {util.GetPodFullName(updatedPods[1])},
			},
		},
	}

	for i, test := range tests {
		upm := newUnschedulablePodsMap()
		for _, p := range test.podsToAdd {
			upm.Add(p)
		}
		require.True(t, reflect.DeepEqual(upm.pods, test.expectedMapAfterAdd),
			"#%d: Unexpected map after adding pods. Expected: %v, got: %v",
			i, test.expectedMapAfterAdd, upm.pods)
		require.True(t, reflect.DeepEqual(upm.nominatedPods, test.expectedNominatedAfterAdd),
			"#%d: Unexpected nominated map after adding pods. Expected: %v, got: %v",
			i, test.expectedNominatedAfterAdd, upm.nominatedPods)
		if len(test.podsToUpdate) > 0 {
			for _, p := range test.podsToUpdate {
				upm.Update(p)
			}
			require.True(t, reflect.DeepEqual(upm.pods, test.expectedMapAfterUpdate),
				"#%d: Unexpected map after updating pods. Expected: %v, got: %v",
				i, test.expectedMapAfterUpdate, upm.pods)
			require.True(t, reflect.DeepEqual(upm.nominatedPods, test.expectedNominatedAfterUpdate),
				"#%d: Unexpected nominated map after updating pods. Expected: %v, got: %v",
				i, test.expectedNominatedAfterUpdate, upm.nominatedPods)
		}
		for _, p := range test.podsToDelete {
			upm.Delete(p)
		}
		require.True(t, reflect.DeepEqual(upm.pods, test.expectedMapAfterDelete),
			"#%d: Unexpected map after deleting pods. Expected: %v, got: %v",
			i, test.expectedMapAfterDelete, upm.pods)
		require.True(t, reflect.DeepEqual(upm.nominatedPods, test.expectedNominatedAfterDelete),
			"#%d: Unexpected nominated map after deleting pods. Expected: %v, got: %v",
			i, test.expectedNominatedAfterDelete, upm.nominatedPods)
		upm.Clear()
		require.Len(t, upm.pods, 0, "Expected the map to be empty, but has %v elements.", len(upm.pods))
	}
}
