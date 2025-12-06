/*
Copyright 2024 The Kubernetes Authors.

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

package metricsutil

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
)

// UtilizationStatus represents the resource utilization state of a pod
type UtilizationStatus string

const (
	StatusOK              UtilizationStatus = "OK"
	StatusOverProvisioned UtilizationStatus = "OVER-PROVISIONED"
	StatusNearLimit       UtilizationStatus = "NEAR-LIMIT"
	StatusNoLimit         UtilizationStatus = "NO-LIMIT"
	StatusNoRequest       UtilizationStatus = "NO-REQUEST"
)

// Thresholds for status determination
const (
	OverProvisionedThreshold = 25 // Below 25% of request = over-provisioned
	NearLimitThreshold       = 90 // Above 90% of limit = near limit
)

// PodUtilization holds computed utilization data for a single pod
type PodUtilization struct {
	// Pod identification
	Name      string
	Namespace string

	// Raw metrics (from Metrics API)
	CPUUsage    resource.Quantity
	MemoryUsage resource.Quantity

	// Resource specs (from Pod API)
	CPURequest    resource.Quantity
	CPULimit      resource.Quantity
	MemoryRequest resource.Quantity
	MemoryLimit   resource.Quantity

	// Computed percentages (-1 means not applicable/no request or limit set)
	CPURequestPercent    int64
	CPULimitPercent      int64
	MemoryRequestPercent int64
	MemoryLimitPercent   int64

	// Computed status
	Status UtilizationStatus

	// Flags for missing data
	HasCPURequest    bool
	HasCPULimit      bool
	HasMemoryRequest bool
	HasMemoryLimit   bool
}

// CalculatePodUtilization computes utilization metrics for a pod
func CalculatePodUtilization(
	podMetrics *metricsapi.PodMetrics,
	podSpec *corev1.Pod,
	measuredResources []corev1.ResourceName,
) *PodUtilization {
	util := &PodUtilization{
		Name:      podMetrics.Name,
		Namespace: podMetrics.Namespace,
	}

	// Aggregate container metrics
	podUsage := getPodMetrics(podMetrics, measuredResources)
	util.CPUUsage = podUsage[corev1.ResourceCPU]
	util.MemoryUsage = podUsage[corev1.ResourceMemory]

	// Get pod requests and limits
	if podSpec != nil {
		reqs, limits := podRequestsAndLimitsFromSpec(podSpec)

		// CPU
		if req, ok := reqs[corev1.ResourceCPU]; ok && !req.IsZero() {
			util.CPURequest = req
			util.HasCPURequest = true
			util.CPURequestPercent = calculatePercent(util.CPUUsage.MilliValue(), req.MilliValue())
		} else {
			util.CPURequestPercent = -1
		}

		if lim, ok := limits[corev1.ResourceCPU]; ok && !lim.IsZero() {
			util.CPULimit = lim
			util.HasCPULimit = true
			util.CPULimitPercent = calculatePercent(util.CPUUsage.MilliValue(), lim.MilliValue())
		} else {
			util.CPULimitPercent = -1
		}

		// Memory
		if req, ok := reqs[corev1.ResourceMemory]; ok && !req.IsZero() {
			util.MemoryRequest = req
			util.HasMemoryRequest = true
			util.MemoryRequestPercent = calculatePercent(util.MemoryUsage.Value(), req.Value())
		} else {
			util.MemoryRequestPercent = -1
		}

		if lim, ok := limits[corev1.ResourceMemory]; ok && !lim.IsZero() {
			util.MemoryLimit = lim
			util.HasMemoryLimit = true
			util.MemoryLimitPercent = calculatePercent(util.MemoryUsage.Value(), lim.Value())
		} else {
			util.MemoryLimitPercent = -1
		}
	} else {
		// No pod spec available
		util.CPURequestPercent = -1
		util.CPULimitPercent = -1
		util.MemoryRequestPercent = -1
		util.MemoryLimitPercent = -1
	}

	// Determine status
	util.Status = determineStatus(util)

	return util
}

// calculatePercent calculates percentage, returning -1 if denominator is 0
func calculatePercent(numerator, denominator int64) int64 {
	if denominator == 0 {
		return -1
	}
	return (numerator * 100) / denominator
}

// determineStatus determines the utilization status based on thresholds
func determineStatus(util *PodUtilization) UtilizationStatus {
	// Check for missing limits (highest priority concern)
	if !util.HasCPULimit && !util.HasMemoryLimit {
		return StatusNoLimit
	}

	// Check for missing requests
	if !util.HasCPURequest && !util.HasMemoryRequest {
		return StatusNoRequest
	}

	// Check for near-limit conditions (risk)
	if (util.HasCPULimit && util.CPULimitPercent >= NearLimitThreshold) ||
		(util.HasMemoryLimit && util.MemoryLimitPercent >= NearLimitThreshold) {
		return StatusNearLimit
	}

	// Check for over-provisioning (waste)
	cpuOverProvisioned := util.HasCPURequest && util.CPURequestPercent >= 0 &&
		util.CPURequestPercent < OverProvisionedThreshold
	memOverProvisioned := util.HasMemoryRequest && util.MemoryRequestPercent >= 0 &&
		util.MemoryRequestPercent < OverProvisionedThreshold

	if cpuOverProvisioned || memOverProvisioned {
		return StatusOverProvisioned
	}

	return StatusOK
}

// podRequestsAndLimitsFromSpec extracts requests and limits from pod spec
func podRequestsAndLimitsFromSpec(pod *corev1.Pod) (reqs, limits corev1.ResourceList) {
	reqs = make(corev1.ResourceList)
	limits = make(corev1.ResourceList)

	for _, container := range pod.Spec.Containers {
		addResourceList(reqs, container.Resources.Requests)
		addResourceList(limits, container.Resources.Limits)
	}

	// Also consider init containers (take max)
	for _, container := range pod.Spec.InitContainers {
		maxResourceList(reqs, container.Resources.Requests)
		maxResourceList(limits, container.Resources.Limits)
	}

	return reqs, limits
}

// addResourceList adds the resources in newList to list
func addResourceList(list, newList corev1.ResourceList) {
	for name, quantity := range newList {
		if value, ok := list[name]; ok {
			value.Add(quantity)
			list[name] = value
		} else {
			list[name] = quantity.DeepCopy()
		}
	}
}

// maxResourceList sets list[name] = max(list[name], newList[name]) for each name
func maxResourceList(list, newList corev1.ResourceList) {
	for name, quantity := range newList {
		if value, ok := list[name]; ok {
			if quantity.Cmp(value) > 0 {
				list[name] = quantity.DeepCopy()
			}
		} else {
			list[name] = quantity.DeepCopy()
		}
	}
}
