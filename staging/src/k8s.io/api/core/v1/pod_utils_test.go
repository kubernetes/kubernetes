/*
Copyright 2026 The Kubernetes Authors.

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

package v1

import (
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGetConditionStatus(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod"},
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionTrue},
				{Type: corev1.PodInitialized, Status: corev1.ConditionFalse},
			},
		},
	}

	analyzer := NewPodConditionAnalyzer(pod)

	if analyzer.GetConditionStatus(corev1.PodReady) != corev1.ConditionTrue {
		t.Error("Expected PodReady to be True")
	}
	if analyzer.GetConditionStatus(corev1.PodInitialized) != corev1.ConditionFalse {
		t.Error("Expected PodInitialized to be False")
	}
	if analyzer.GetConditionStatus("Unknown") != corev1.ConditionUnknown {
		t.Error("Expected unknown condition to return Unknown")
	}
}

func TestIsReady(t *testing.T) {
	tests := []struct {
		name     string
		status   corev1.ConditionStatus
		expected bool
	}{
		{"ready", corev1.ConditionTrue, true},
		{"not ready", corev1.ConditionFalse, false},
		{"unknown", corev1.ConditionUnknown, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := &corev1.Pod{
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{Type: corev1.PodReady, Status: tt.status},
					},
				},
			}
			analyzer := NewPodConditionAnalyzer(pod)
			if analyzer.IsReady() != tt.expected {
				t.Errorf("Expected IsReady() = %v", tt.expected)
			}
		})
	}
}

func TestGetUnhealthyContainers(t *testing.T) {
	pod := &corev1.Pod{
		Status: corev1.PodStatus{
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "ready", Ready: true},
				{Name: "not-ready", Ready: false},
				{Name: "also-ready", Ready: true},
			},
		},
	}

	analyzer := NewPodConditionAnalyzer(pod)
	unhealthy := analyzer.GetUnhealthyContainers()

	if len(unhealthy) != 1 {
		t.Errorf("Expected 1 unhealthy container, got %d", len(unhealthy))
	}
	if unhealthy[0].Name != "not-ready" {
		t.Errorf("Expected not-ready, got %s", unhealthy[0].Name)
	}
}

func TestGetRestartingContainers(t *testing.T) {
	pod := &corev1.Pod{
		Status: corev1.PodStatus{
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "low", RestartCount: 2},
				{Name: "high", RestartCount: 10},
				{Name: "medium", RestartCount: 5},
			},
		},
	}

	analyzer := NewPodConditionAnalyzer(pod)
	restarting := analyzer.GetRestartingContainers(5)

	if len(restarting) != 2 {
		t.Errorf("Expected 2 restarting containers, got %d", len(restarting))
	}
}

func TestGetTotalRestartCount(t *testing.T) {
	pod := &corev1.Pod{
		Status: corev1.PodStatus{
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "a", RestartCount: 3},
				{Name: "b", RestartCount: 7},
			},
		},
	}

	analyzer := NewPodConditionAnalyzer(pod)
	total := analyzer.GetTotalRestartCount()

	if total != 10 {
		t.Errorf("Expected 10, got %d", total)
	}
}

func TestGetConditionAge(t *testing.T) {
	past := time.Now().Add(-5 * time.Minute)
	pod := &corev1.Pod{
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, LastTransitionTime: metav1.Time{Time: past}},
			},
		},
	}

	analyzer := NewPodConditionAnalyzer(pod)
	age := analyzer.GetConditionAge(corev1.PodReady)

	if age < 4*time.Minute || age > 6*time.Minute {
		t.Errorf("Expected ~5 minutes, got %v", age)
	}
}

func TestGetSummary(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionTrue},
			},
			ContainerStatuses: []corev1.ContainerStatus{
				{Name: "app", Ready: true, RestartCount: 0},
			},
		},
	}

	analyzer := NewPodConditionAnalyzer(pod)
	summary := analyzer.GetSummary()

	if summary == "" {
		t.Error("Expected non-empty summary")
	}
}