/*
Copyright 2023 The Kubernetes Authors.

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

package podutils

import (
	"sort"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestActivePods(t *testing.T) {
	time1 := metav1.Now()
	time2 := metav1.NewTime(time1.Add(1 * time.Second))
	time3 := metav1.NewTime(time1.Add(2 * time.Second))

	tests := []struct {
		name string
		pod1 *corev1.Pod
		pod2 *corev1.Pod
	}{
		{
			name: "unassigned pod should sort before assigned pod",
			pod1: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "unassignedPod",
					Namespace: "default",
				},
			},
			pod2: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "assignedPod",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
			},
		},
		{
			name: "pending pod should sort before unknown pod",
			pod1: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pendingPod",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodPending,
				},
			},
			pod2: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "unknownPod",
					Namespace:         "default",
					CreationTimestamp: time1,
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodUnknown,
				},
			},
		},
		{
			name: "unknown pod should sort before running pod",
			pod1: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "unknownPod",
					Namespace:         "default",
					CreationTimestamp: time1,
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodUnknown,
				},
			},
			pod2: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "runningPod",
					Namespace:         "default",
					CreationTimestamp: time1,
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
				},
			},
		},
		{
			name: "unready pod should sort before ready pod",
			pod1: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "unreadyPod",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
				},
			},
			pod2: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "readyPod",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					Conditions: []corev1.PodCondition{
						{
							Type:               corev1.PodReady,
							Status:             corev1.ConditionTrue,
							LastTransitionTime: time1,
						},
					},
				},
			},
		},
		{
			name: "pod with deletion timestamp should sort before pod without deletion timestamp",
			pod1: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "readyPodDeleting",
					Namespace:         "default",
					DeletionTimestamp: &time2,
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					Conditions: []corev1.PodCondition{
						{
							Type:               corev1.PodReady,
							Status:             corev1.ConditionTrue,
							LastTransitionTime: time1,
						},
					},
				},
			},
			pod2: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "readyPod",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					Conditions: []corev1.PodCondition{
						{
							Type:               corev1.PodReady,
							Status:             corev1.ConditionTrue,
							LastTransitionTime: time1,
						},
					},
				},
			},
		},
		{
			name: "older deletion timestamp should sort before newer deletion timestamp",
			pod1: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "readyPodDeletingOlder",
					Namespace:         "default",
					DeletionTimestamp: &time2,
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					Conditions: []corev1.PodCondition{
						{
							Type:               corev1.PodReady,
							Status:             corev1.ConditionTrue,
							LastTransitionTime: time1,
						},
					},
				},
			},
			pod2: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "readyPodDeletingNewer",
					Namespace:         "default",
					DeletionTimestamp: &time3,
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					Conditions: []corev1.PodCondition{
						{
							Type:               corev1.PodReady,
							Status:             corev1.ConditionTrue,
							LastTransitionTime: time1,
						},
					},
				},
			},
		},
		{
			name: "newer ready timestamp should sort before older ready timestamp",
			pod1: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "newerReadyPod",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					Conditions: []corev1.PodCondition{
						{
							Type:               corev1.PodReady,
							Status:             corev1.ConditionTrue,
							LastTransitionTime: time3,
						},
					},
				},
			},
			pod2: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "olderReadyPod",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					Conditions: []corev1.PodCondition{
						{
							Type:               corev1.PodReady,
							Status:             corev1.ConditionTrue,
							LastTransitionTime: time2,
						},
					},
				},
			},
		},
		{
			name: "higher restart count should sort before lower restart count",
			pod1: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "podWithMoreRestarts",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					Conditions: []corev1.PodCondition{
						{
							Type:               corev1.PodReady,
							Status:             corev1.ConditionTrue,
							LastTransitionTime: time1,
						},
					},
					ContainerStatuses: []corev1.ContainerStatus{
						{
							RestartCount: 3,
						},
					},
				},
			},
			pod2: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "podWithLessRestarts",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					Conditions: []corev1.PodCondition{
						{
							Type:               corev1.PodReady,
							Status:             corev1.ConditionTrue,
							LastTransitionTime: time1,
						},
					},
					ContainerStatuses: []corev1.ContainerStatus{
						{
							RestartCount: 2,
						},
					},
				},
			},
		},
		{
			name: "newer creation timestamp should sort before older creation timestamp",
			pod1: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "newerCreationPod",
					Namespace:         "default",
					CreationTimestamp: time3,
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					Conditions: []corev1.PodCondition{
						{
							Type:               corev1.PodReady,
							Status:             corev1.ConditionTrue,
							LastTransitionTime: time1,
						},
					},
				},
			},
			pod2: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "olderCreationPod",
					Namespace:         "default",
					CreationTimestamp: time2,
				},
				Spec: corev1.PodSpec{
					NodeName: "node1",
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					Conditions: []corev1.PodCondition{
						{
							Type:               corev1.PodReady,
							Status:             corev1.ConditionTrue,
							LastTransitionTime: time1,
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test that the pods are sorted in the correct order when pod1 is first and pod2 is second.
			pods := ActivePods{tt.pod1, tt.pod2}
			sort.Sort(pods)
			if pods[0] != tt.pod1 || pods[1] != tt.pod2 {
				t.Errorf("Incorrect ActivePods sorting, expected pod1 to be first")
			}

			// Test that the pods are sorted in the correct order when pod2 is first and pod1 is second.
			pods = ActivePods{tt.pod2, tt.pod1}
			sort.Sort(pods)
			if pods[0] != tt.pod1 || pods[1] != tt.pod2 {
				t.Errorf("Incorrect ActivePods sorting, expected pod1 to be first")
			}
		})
	}
}
