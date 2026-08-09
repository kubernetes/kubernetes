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
	"fmt"
	"sort"
	"time"

	corev1 "k8s.io/api/core/v1"
)

// PodConditionAnalyzer provides utilities for analyzing pod conditions.
type PodConditionAnalyzer struct {
	pod *corev1.Pod
}

// NewPodConditionAnalyzer creates a new analyzer for the given pod.
func NewPodConditionAnalyzer(pod *corev1.Pod) *PodConditionAnalyzer {
	return &PodConditionAnalyzer{pod: pod}
}

// GetConditionStatus returns the status of a specific condition.
func (a *PodConditionAnalyzer) GetConditionStatus(conditionType corev1.PodConditionType) corev1.ConditionStatus {
	for _, condition := range a.pod.Status.Conditions {
		if condition.Type == conditionType {
			return condition.Status
		}
	}
	return corev1.ConditionUnknown
}

// IsReady returns true if the pod has a Ready condition with status True.
func (a *PodConditionAnalyzer) IsReady() bool {
	return a.GetConditionStatus(corev1.PodReady) == corev1.ConditionTrue
}

// IsInitialized returns true if the pod has been initialized.
func (a *PodConditionAnalyzer) IsInitialized() bool {
	return a.GetConditionStatus(corev1.PodInitialized) == corev1.ConditionTrue
}

// IsPodScheduled returns true if the pod has been scheduled to a node.
func (a *PodConditionAnalyzer) IsPodScheduled() bool {
	return a.GetConditionStatus(corev1.PodScheduled) == corev1.ConditionTrue
}

// GetConditionAge returns the duration since a condition was last transitioned.
func (a *PodConditionAnalyzer) GetConditionAge(conditionType corev1.PodConditionType) time.Duration {
	for _, condition := range a.pod.Status.Conditions {
		if condition.Type == conditionType {
			return time.Since(condition.LastTransitionTime.Time)
		}
	}
	return 0
}

// GetUnhealthyContainers returns containers that are not in a ready state.
func (a *PodConditionAnalyzer) GetUnhealthyContainers() []corev1.ContainerStatus {
	var unhealthy []corev1.ContainerStatus
	for _, status := range a.pod.Status.ContainerStatuses {
		if !status.Ready {
			unhealthy = append(unhealthy, status)
		}
	}
	return unhealthy
}

// GetRestartingContainers returns containers that have restarted more than the given threshold.
func (a *PodConditionAnalyzer) GetRestartingContainers(threshold int32) []corev1.ContainerStatus {
	var restarting []corev1.ContainerStatus
	for _, status := range a.pod.Status.ContainerStatuses {
		if status.RestartCount > threshold {
			restarting = append(restarting, status)
		}
	}
	return restarting
}

// GetWaitingContainers returns containers that are in a waiting state.
func (a *PodConditionAnalyzer) GetWaitingContainers() []corev1.ContainerStatus {
	var waiting []corev1.ContainerStatus
	for _, status := range a.pod.Status.ContainerStatuses {
		if status.State.Waiting != nil {
			waiting = append(waiting, status)
		}
	}
	return waiting
}

// GetSummary returns a human-readable summary of the pod status.
func (a *PodConditionAnalyzer) GetSummary() string {
	summary := fmt.Sprintf("Pod: %s/%s\n", a.pod.Namespace, a.pod.Name)
	summary += fmt.Sprintf("Phase: %s\n", a.pod.Status.Phase)
	summary += fmt.Sprintf("Ready: %v\n", a.IsReady())
	summary += fmt.Sprintf("Restart Count: %d\n", a.GetTotalRestartCount())

	if len(a.pod.Status.Conditions) > 0 {
		summary += "Conditions:\n"
		conditions := a.pod.Status.Conditions
		sort.Slice(conditions, func(i, j int) bool {
			return conditions[i].Type < conditions[j].Type
		})
		for _, c := range conditions {
			summary += fmt.Sprintf("  %s: %s (last transition: %s ago)\n",
				c.Type, c.Status, a.GetConditionAge(c.Type))
		}
	}

	unhealthy := a.GetUnhealthyContainers()
	if len(unhealthy) > 0 {
		summary += fmt.Sprintf("Unhealthy Containers: %d\n", len(unhealthy))
		for _, status := range unhealthy {
			summary += fmt.Sprintf("  - %s\n", status.Name)
		}
	}

	return summary
}

// GetTotalRestartCount returns the total restart count across all containers.
func (a *PodConditionAnalyzer) GetTotalRestartCount() int32 {
	var total int32
	for _, status := range a.pod.Status.ContainerStatuses {
		total += status.RestartCount
	}
	return total
}

// GetConditionMessages returns messages from all conditions.
func (a *PodConditionAnalyzer) GetConditionMessages() []string {
	var messages []string
	for _, condition := range a.pod.Status.Conditions {
		if condition.Message != "" {
			messages = append(messages, fmt.Sprintf("[%s] %s", condition.Type, condition.Message))
		}
	}
	return messages
}