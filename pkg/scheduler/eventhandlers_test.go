/*
Copyright 2019 The Kubernetes Authors.

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

package scheduler

import (
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	fakecache "k8s.io/kubernetes/pkg/scheduler/internal/cache/fake"
	"k8s.io/kubernetes/pkg/scheduler/internal/queue"
)

func TestSkipPodUpdate(t *testing.T) {
	for _, test := range []struct {
		pod              *v1.Pod
		isAssumedPodFunc func(*v1.Pod) bool
		getPodFunc       func(*v1.Pod) *v1.Pod
		expected         bool
		name             string
	}{
		{
			name: "Non-assumed pod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod-0",
				},
			},
			isAssumedPodFunc: func(*v1.Pod) bool { return false },
			getPodFunc: func(*v1.Pod) *v1.Pod {
				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod-0",
					},
				}
			},
			expected: false,
		},
		{
			name: "with changes on ResourceVersion, Spec.NodeName and/or Annotations",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "pod-0",
					Annotations:     map[string]string{"a": "b"},
					ResourceVersion: "0",
				},
				Spec: v1.PodSpec{
					NodeName: "node-0",
				},
			},
			isAssumedPodFunc: func(*v1.Pod) bool {
				return true
			},
			getPodFunc: func(*v1.Pod) *v1.Pod {
				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:            "pod-0",
						Annotations:     map[string]string{"c": "d"},
						ResourceVersion: "1",
					},
					Spec: v1.PodSpec{
						NodeName: "node-1",
					},
				}
			},
			expected: true,
		},
		{
			name: "with ServerSideApply changes on Annotations",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "pod-0",
					Annotations:     map[string]string{"a": "b"},
					ResourceVersion: "0",
					ManagedFields: []metav1.ManagedFieldsEntry{
						{
							Manager:    "some-actor",
							Operation:  metav1.ManagedFieldsOperationApply,
							APIVersion: "v1",
							FieldsType: "FieldsV1",
							FieldsV1: &metav1.FieldsV1{
								Raw: []byte(`
									"f:metadata": {
									  "f:annotations": {
										"f:a: {}
									  }
									}
								`),
							},
						},
					},
				},
				Spec: v1.PodSpec{
					NodeName: "node-0",
				},
			},
			isAssumedPodFunc: func(*v1.Pod) bool {
				return true
			},
			getPodFunc: func(*v1.Pod) *v1.Pod {
				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:            "pod-0",
						Annotations:     map[string]string{"a": "c", "d": "e"},
						ResourceVersion: "1",
						ManagedFields: []metav1.ManagedFieldsEntry{
							{
								Manager:    "some-actor",
								Operation:  metav1.ManagedFieldsOperationApply,
								APIVersion: "v1",
								FieldsType: "FieldsV1",
								FieldsV1: &metav1.FieldsV1{
									Raw: []byte(`
										"f:metadata": {
										  "f:annotations": {
											"f:a: {}
											"f:d: {}
										  }
										}
									`),
								},
							},
							{
								Manager:    "some-actor",
								Operation:  metav1.ManagedFieldsOperationApply,
								APIVersion: "v1",
								FieldsType: "FieldsV1",
								FieldsV1: &metav1.FieldsV1{
									Raw: []byte(`
										"f:metadata": {
										  "f:annotations": {
											"f:a: {}
										  }
										}
									`),
								},
							},
						},
					},
					Spec: v1.PodSpec{
						NodeName: "node-1",
					},
				}
			},
			expected: true,
		},
		{
			name: "with changes on Labels",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "pod-0",
					Labels: map[string]string{"a": "b"},
				},
			},
			isAssumedPodFunc: func(*v1.Pod) bool {
				return true
			},
			getPodFunc: func(*v1.Pod) *v1.Pod {
				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "pod-0",
						Labels: map[string]string{"c": "d"},
					},
				}
			},
			expected: false,
		},
		{
			name: "with changes on Finalizers",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "pod-0",
					Finalizers: []string{"a", "b"},
				},
			},
			isAssumedPodFunc: func(*v1.Pod) bool {
				return true
			},
			getPodFunc: func(*v1.Pod) *v1.Pod {
				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "pod-0",
						Finalizers: []string{"c", "d"},
					},
				}
			},
			expected: true,
		},
		{
			name: "with changes on Conditions",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod-0",
				},
				Status: v1.PodStatus{
					Conditions: []v1.PodCondition{
						{Type: "foo"},
					},
				},
			},
			isAssumedPodFunc: func(*v1.Pod) bool {
				return true
			},
			getPodFunc: func(*v1.Pod) *v1.Pod {
				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod-0",
					},
				}
			},
			expected: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			c := &Scheduler{
				SchedulerCache: &fakecache.Cache{
					IsAssumedPodFunc: test.isAssumedPodFunc,
					GetPodFunc:       test.getPodFunc,
				},
			}
			got := c.skipPodUpdate(test.pod)
			if got != test.expected {
				t.Errorf("skipPodUpdate() = %t, expected = %t", got, test.expected)
			}
		})
	}
}

func TestNodeAllocatableChanged(t *testing.T) {
	newQuantity := func(value int64) resource.Quantity {
		return *resource.NewQuantity(value, resource.BinarySI)
	}
	for _, test := range []struct {
		Name           string
		Changed        bool
		OldAllocatable v1.ResourceList
		NewAllocatable v1.ResourceList
	}{
		{
			Name:           "no allocatable resources changed",
			Changed:        false,
			OldAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
			NewAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
		},
		{
			Name:           "new node has more allocatable resources",
			Changed:        true,
			OldAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
			NewAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024), v1.ResourceStorage: newQuantity(1024)},
		},
	} {
		t.Run(test.Name, func(t *testing.T) {
			oldNode := &v1.Node{Status: v1.NodeStatus{Allocatable: test.OldAllocatable}}
			newNode := &v1.Node{Status: v1.NodeStatus{Allocatable: test.NewAllocatable}}
			changed := nodeAllocatableChanged(newNode, oldNode)
			if changed != test.Changed {
				t.Errorf("nodeAllocatableChanged should be %t, got %t", test.Changed, changed)
			}
		})
	}
}

func TestNodeLabelsChanged(t *testing.T) {
	for _, test := range []struct {
		Name      string
		Changed   bool
		OldLabels map[string]string
		NewLabels map[string]string
	}{
		{
			Name:      "no labels changed",
			Changed:   false,
			OldLabels: map[string]string{"foo": "bar"},
			NewLabels: map[string]string{"foo": "bar"},
		},
		// Labels changed.
		{
			Name:      "new node has more labels",
			Changed:   true,
			OldLabels: map[string]string{"foo": "bar"},
			NewLabels: map[string]string{"foo": "bar", "test": "value"},
		},
	} {
		t.Run(test.Name, func(t *testing.T) {
			oldNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: test.OldLabels}}
			newNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: test.NewLabels}}
			changed := nodeLabelsChanged(newNode, oldNode)
			if changed != test.Changed {
				t.Errorf("Test case %q failed: should be %t, got %t", test.Name, test.Changed, changed)
			}
		})
	}
}

func TestNodeTaintsChanged(t *testing.T) {
	for _, test := range []struct {
		Name      string
		Changed   bool
		OldTaints []v1.Taint
		NewTaints []v1.Taint
	}{
		{
			Name:      "no taint changed",
			Changed:   false,
			OldTaints: []v1.Taint{{Key: "key", Value: "value"}},
			NewTaints: []v1.Taint{{Key: "key", Value: "value"}},
		},
		{
			Name:      "taint value changed",
			Changed:   true,
			OldTaints: []v1.Taint{{Key: "key", Value: "value1"}},
			NewTaints: []v1.Taint{{Key: "key", Value: "value2"}},
		},
	} {
		t.Run(test.Name, func(t *testing.T) {
			oldNode := &v1.Node{Spec: v1.NodeSpec{Taints: test.OldTaints}}
			newNode := &v1.Node{Spec: v1.NodeSpec{Taints: test.NewTaints}}
			changed := nodeTaintsChanged(newNode, oldNode)
			if changed != test.Changed {
				t.Errorf("Test case %q failed: should be %t, not %t", test.Name, test.Changed, changed)
			}
		})
	}
}

func TestNodeConditionsChanged(t *testing.T) {
	nodeConditionType := reflect.TypeOf(v1.NodeCondition{})
	if nodeConditionType.NumField() != 6 {
		t.Errorf("NodeCondition type has changed. The nodeConditionsChanged() function must be reevaluated.")
	}

	for _, test := range []struct {
		Name          string
		Changed       bool
		OldConditions []v1.NodeCondition
		NewConditions []v1.NodeCondition
	}{
		{
			Name:          "no condition changed",
			Changed:       false,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue}},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue}},
		},
		{
			Name:          "only LastHeartbeatTime changed",
			Changed:       false,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue, LastHeartbeatTime: metav1.Unix(1, 0)}},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue, LastHeartbeatTime: metav1.Unix(2, 0)}},
		},
		{
			Name:          "new node has more healthy conditions",
			Changed:       true,
			OldConditions: []v1.NodeCondition{},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
		},
		{
			Name:          "new node has less unhealthy conditions",
			Changed:       true,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue}},
			NewConditions: []v1.NodeCondition{},
		},
		{
			Name:          "condition status changed",
			Changed:       true,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
		},
	} {
		t.Run(test.Name, func(t *testing.T) {
			oldNode := &v1.Node{Status: v1.NodeStatus{Conditions: test.OldConditions}}
			newNode := &v1.Node{Status: v1.NodeStatus{Conditions: test.NewConditions}}
			changed := nodeConditionsChanged(newNode, oldNode)
			if changed != test.Changed {
				t.Errorf("Test case %q failed: should be %t, got %t", test.Name, test.Changed, changed)
			}
		})
	}
}

func TestUpdatePodInCache(t *testing.T) {
	ttl := 10 * time.Second
	nodeName := "node"

	tests := []struct {
		name   string
		oldObj interface{}
		newObj interface{}
	}{
		{
			name:   "pod updated with the same UID",
			oldObj: withPodName(podWithPort("oldUID", nodeName, 80), "pod"),
			newObj: withPodName(podWithPort("oldUID", nodeName, 8080), "pod"),
		},
		{
			name:   "pod updated with different UIDs",
			oldObj: withPodName(podWithPort("oldUID", nodeName, 80), "pod"),
			newObj: withPodName(podWithPort("newUID", nodeName, 8080), "pod"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stopCh := make(chan struct{})
			defer close(stopCh)
			schedulerCache := cache.New(ttl, stopCh)
			schedulerQueue := queue.NewPriorityQueue(nil)
			sched := &Scheduler{
				SchedulerCache:  schedulerCache,
				SchedulingQueue: schedulerQueue,
			}
			sched.addPodToCache(tt.oldObj)
			sched.updatePodInCache(tt.oldObj, tt.newObj)
			pod, err := sched.SchedulerCache.GetPod(tt.newObj.(*v1.Pod))
			if err != nil {
				t.Errorf("Failed to get pod from scheduler: %v", err)
			}
			if pod.UID != tt.newObj.(*v1.Pod).UID {
				t.Errorf("Want pod UID %v, got %v", tt.newObj.(*v1.Pod).UID, pod.UID)
			}
		})
	}
}

func withPodName(pod *v1.Pod, name string) *v1.Pod {
	pod.Name = name
	return pod
}
