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
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	extenderv1 "k8s.io/kubernetes/pkg/scheduler/apis/extender/v1"
)

func TestGetPodFullName(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test",
			Name:      "pod",
		},
	}
	got := GetPodFullName(pod)
	expected := fmt.Sprintf("%s_%s", pod.Name, pod.Namespace)
	if got != expected {
		t.Errorf("Got wrong full name, got: %s, expected: %s", got, expected)
	}
}

func newPriorityPodWithStartTime(name string, priority int32, startTime time.Time) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Priority: &priority,
		},
		Status: v1.PodStatus{
			StartTime: &metav1.Time{Time: startTime},
		},
	}
}

func TestGetEarliestPodStartTime(t *testing.T) {
	currentTime := time.Now()
	pod1 := newPriorityPodWithStartTime("pod1", 1, currentTime.Add(time.Second))
	pod2 := newPriorityPodWithStartTime("pod2", 2, currentTime.Add(time.Second))
	pod3 := newPriorityPodWithStartTime("pod3", 2, currentTime)
	victims := &extenderv1.Victims{
		Pods: []*v1.Pod{pod1, pod2, pod3},
	}
	startTime := GetEarliestPodStartTime(victims)
	if !startTime.Equal(pod3.Status.StartTime) {
		t.Errorf("Got wrong earliest pod start time")
	}

	pod1 = newPriorityPodWithStartTime("pod1", 2, currentTime)
	pod2 = newPriorityPodWithStartTime("pod2", 2, currentTime.Add(time.Second))
	pod3 = newPriorityPodWithStartTime("pod3", 2, currentTime.Add(2*time.Second))
	victims = &extenderv1.Victims{
		Pods: []*v1.Pod{pod1, pod2, pod3},
	}
	startTime = GetEarliestPodStartTime(victims)
	if !startTime.Equal(pod1.Status.StartTime) {
		t.Errorf("Got wrong earliest pod start time, got %v, expected %v", startTime, pod1.Status.StartTime)
	}
}

func TestMoreImportantPod(t *testing.T) {
	currentTime := time.Now()
	pod1 := newPriorityPodWithStartTime("pod1", 1, currentTime)
	pod2 := newPriorityPodWithStartTime("pod2", 2, currentTime.Add(time.Second))
	pod3 := newPriorityPodWithStartTime("pod3", 2, currentTime)

	tests := map[string]struct {
		p1       *v1.Pod
		p2       *v1.Pod
		expected bool
	}{
		"Pod with higher priority": {
			p1:       pod1,
			p2:       pod2,
			expected: false,
		},
		"Pod with older created time": {
			p1:       pod2,
			p2:       pod3,
			expected: false,
		},
		"Pods with same start time": {
			p1:       pod3,
			p2:       pod1,
			expected: true,
		},
	}

	for k, v := range tests {
		got := MoreImportantPod(v.p1, v.p2)
		if got != v.expected {
			t.Errorf("%s failed, expected %t but got %t", k, v.expected, got)
		}
	}
}
