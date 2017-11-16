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

package util

import (
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/apis/scheduling"
)

// TestGetPodPriority tests GetPodPriority function.
func TestGetPodPriority(t *testing.T) {
	p := int32(20)
	tests := []struct {
		name             string
		pod              *v1.Pod
		expectedPriority int32
	}{
		{
			name: "no priority pod resolves to static default priority",
			pod: &v1.Pod{
				Spec: v1.PodSpec{Containers: []v1.Container{
					{Name: "container", Image: "image"}},
				},
			},
			expectedPriority: scheduling.DefaultPriorityWhenNoDefaultClassExists,
		},
		{
			name: "pod with priority resolves correctly",
			pod: &v1.Pod{
				Spec: v1.PodSpec{Containers: []v1.Container{
					{Name: "container", Image: "image"}},
					Priority: &p,
				},
			},
			expectedPriority: p,
		},
	}
	for _, test := range tests {
		if GetPodPriority(test.pod) != test.expectedPriority {
			t.Errorf("expected pod priority: %v, got %v", test.expectedPriority, GetPodPriority(test.pod))
		}

	}
}

// TestSortableList tests SortableList by storing pods in the list and sorting
// them by their priority.
func TestSortableList(t *testing.T) {
	higherPriority := func(pod1, pod2 interface{}) bool {
		return GetPodPriority(pod1.(*v1.Pod)) > GetPodPriority(pod2.(*v1.Pod))
	}
	podList := SortableList{CompFunc: higherPriority}
	// Add a few Pods with different priorities from lowest to highest priority.
	for i := 0; i < 10; i++ {
		var p int32 = int32(i)
		pod := &v1.Pod{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "container",
						Image: "image",
					},
				},
				Priority: &p,
			},
		}
		podList.Items = append(podList.Items, pod)
	}
	podList.Sort()
	if len(podList.Items) != 10 {
		t.Errorf("expected length of list was 10, got: %v", len(podList.Items))
	}
	var prevPriority = int32(10)
	for _, p := range podList.Items {
		if *p.(*v1.Pod).Spec.Priority >= prevPriority {
			t.Errorf("Pods are not soreted. Current pod pririty is %v, while previous one was %v.", *p.(*v1.Pod).Spec.Priority, prevPriority)
		}
	}
}
