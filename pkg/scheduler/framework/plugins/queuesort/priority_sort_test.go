/*
Copyright 2020 The Kubernetes Authors.

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

package queuesort

import (
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

func TestLess(t *testing.T) {
	prioritySort := &PrioritySort{}
	var lowPriority, highPriority = int32(10), int32(100)
	t1 := time.Now()
	t2 := t1.Add(time.Second)
	for _, tt := range []struct {
		name     string
		p1       *framework.QueuedPodInfo
		p2       *framework.QueuedPodInfo
		expected bool
	}{
		{
			name: "p1.priority less than p2.priority",
			p1: &framework.QueuedPodInfo{
				Pod: &v1.Pod{
					Spec: v1.PodSpec{
						Priority: &lowPriority,
					},
				},
			},
			p2: &framework.QueuedPodInfo{
				Pod: &v1.Pod{
					Spec: v1.PodSpec{
						Priority: &highPriority,
					},
				},
			},
			expected: false, // p2 should be ahead of p1 in the queue
		},
		{
			name: "p1.priority greater than p2.priority",
			p1: &framework.QueuedPodInfo{
				Pod: &v1.Pod{
					Spec: v1.PodSpec{
						Priority: &highPriority,
					},
				},
			},
			p2: &framework.QueuedPodInfo{
				Pod: &v1.Pod{
					Spec: v1.PodSpec{
						Priority: &lowPriority,
					},
				},
			},
			expected: true, // p1 should be ahead of p2 in the queue
		},
		{
			name: "equal priority. p1 is added to schedulingQ earlier than p2",
			p1: &framework.QueuedPodInfo{
				Pod: &v1.Pod{
					Spec: v1.PodSpec{
						Priority: &highPriority,
					},
				},
				Timestamp: t1,
			},
			p2: &framework.QueuedPodInfo{
				Pod: &v1.Pod{
					Spec: v1.PodSpec{
						Priority: &highPriority,
					},
				},
				Timestamp: t2,
			},
			expected: true, // p1 should be ahead of p2 in the queue
		},
		{
			name: "equal priority. p2 is added to schedulingQ earlier than p1",
			p1: &framework.QueuedPodInfo{
				Pod: &v1.Pod{
					Spec: v1.PodSpec{
						Priority: &highPriority,
					},
				},
				Timestamp: t2,
			},
			p2: &framework.QueuedPodInfo{
				Pod: &v1.Pod{
					Spec: v1.PodSpec{
						Priority: &highPriority,
					},
				},
				Timestamp: t1,
			},
			expected: false, // p2 should be ahead of p1 in the queue
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := prioritySort.Less(tt.p1, tt.p2); got != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, got)
			}
		})
	}
}
