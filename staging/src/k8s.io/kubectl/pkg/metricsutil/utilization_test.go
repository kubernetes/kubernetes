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
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
)

func TestCalculatePercent(t *testing.T) {
	tests := []struct {
		name        string
		numerator   int64
		denominator int64
		expected    int64
	}{
		{"50%", 50, 100, 50},
		{"100%", 100, 100, 100},
		{"0%", 0, 100, 0},
		{"200%", 200, 100, 200},
		{"zero denominator", 50, 0, -1},
		{"both zero", 0, 0, -1},
		{"25%", 25, 100, 25},
		{"10%", 100, 1000, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculatePercent(tt.numerator, tt.denominator)
			if result != tt.expected {
				t.Errorf("calculatePercent(%d, %d) = %d, expected %d",
					tt.numerator, tt.denominator, result, tt.expected)
			}
		})
	}
}

func TestDetermineStatus(t *testing.T) {
	tests := []struct {
		name     string
		util     *PodUtilization
		expected UtilizationStatus
	}{
		{
			name: "OK status",
			util: &PodUtilization{
				HasCPURequest:        true,
				HasCPULimit:          true,
				HasMemoryRequest:     true,
				HasMemoryLimit:       true,
				CPURequestPercent:    50,
				CPULimitPercent:      25,
				MemoryRequestPercent: 50,
				MemoryLimitPercent:   25,
			},
			expected: StatusOK,
		},
		{
			name: "Near limit - CPU",
			util: &PodUtilization{
				HasCPURequest:        true,
				HasCPULimit:          true,
				HasMemoryRequest:     true,
				HasMemoryLimit:       true,
				CPURequestPercent:    95,
				CPULimitPercent:      95,
				MemoryRequestPercent: 50,
				MemoryLimitPercent:   25,
			},
			expected: StatusNearLimit,
		},
		{
			name: "Near limit - Memory",
			util: &PodUtilization{
				HasCPURequest:        true,
				HasCPULimit:          true,
				HasMemoryRequest:     true,
				HasMemoryLimit:       true,
				CPURequestPercent:    50,
				CPULimitPercent:      25,
				MemoryRequestPercent: 95,
				MemoryLimitPercent:   95,
			},
			expected: StatusNearLimit,
		},
		{
			name: "Over-provisioned - CPU",
			util: &PodUtilization{
				HasCPURequest:        true,
				HasCPULimit:          true,
				HasMemoryRequest:     true,
				HasMemoryLimit:       true,
				CPURequestPercent:    10,
				CPULimitPercent:      5,
				MemoryRequestPercent: 50,
				MemoryLimitPercent:   25,
			},
			expected: StatusOverProvisioned,
		},
		{
			name: "Over-provisioned - Memory",
			util: &PodUtilization{
				HasCPURequest:        true,
				HasCPULimit:          true,
				HasMemoryRequest:     true,
				HasMemoryLimit:       true,
				CPURequestPercent:    50,
				CPULimitPercent:      25,
				MemoryRequestPercent: 10,
				MemoryLimitPercent:   5,
			},
			expected: StatusOverProvisioned,
		},
		{
			name: "No limit",
			util: &PodUtilization{
				HasCPURequest:        true,
				HasCPULimit:          false,
				HasMemoryRequest:     true,
				HasMemoryLimit:       false,
				CPURequestPercent:    50,
				CPULimitPercent:      -1,
				MemoryRequestPercent: 50,
				MemoryLimitPercent:   -1,
			},
			expected: StatusNoLimit,
		},
		{
			name: "No request",
			util: &PodUtilization{
				HasCPURequest:        false,
				HasCPULimit:          true,
				HasMemoryRequest:     false,
				HasMemoryLimit:       true,
				CPURequestPercent:    -1,
				CPULimitPercent:      50,
				MemoryRequestPercent: -1,
				MemoryLimitPercent:   50,
			},
			expected: StatusNoRequest,
		},
		{
			name: "No limits takes precedence over no requests",
			util: &PodUtilization{
				HasCPURequest:        false,
				HasCPULimit:          false,
				HasMemoryRequest:     false,
				HasMemoryLimit:       false,
				CPURequestPercent:    -1,
				CPULimitPercent:      -1,
				MemoryRequestPercent: -1,
				MemoryLimitPercent:   -1,
			},
			expected: StatusNoLimit,
		},
		{
			name: "Near limit takes precedence over over-provisioned",
			util: &PodUtilization{
				HasCPURequest:        true,
				HasCPULimit:          true,
				HasMemoryRequest:     true,
				HasMemoryLimit:       true,
				CPURequestPercent:    10, // Over-provisioned
				CPULimitPercent:      5,
				MemoryRequestPercent: 95, // Near limit
				MemoryLimitPercent:   95,
			},
			expected: StatusNearLimit,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := determineStatus(tt.util)
			if result != tt.expected {
				t.Errorf("determineStatus() = %s, expected %s", result, tt.expected)
			}
		})
	}
}

func TestCalculatePodUtilization(t *testing.T) {
	measuredResources := []corev1.ResourceName{corev1.ResourceCPU, corev1.ResourceMemory}

	tests := []struct {
		name                  string
		metrics               *metricsapi.PodMetrics
		podSpec               *corev1.Pod
		expectedCPUReqPercent int64
		expectedMemReqPercent int64
		expectedStatus        UtilizationStatus
	}{
		{
			name: "pod with requests and limits - 50% usage",
			metrics: &metricsapi.PodMetrics{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Containers: []metricsapi.ContainerMetrics{
					{
						Name: "container1",
						Usage: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("128Mi"),
						},
					},
				},
			},
			podSpec: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "container1",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("200m"),
									corev1.ResourceMemory: resource.MustParse("256Mi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("500m"),
									corev1.ResourceMemory: resource.MustParse("512Mi"),
								},
							},
						},
					},
				},
			},
			expectedCPUReqPercent: 50,
			expectedMemReqPercent: 50,
			expectedStatus:        StatusOK,
		},
		{
			name: "pod with low utilization - over-provisioned",
			metrics: &metricsapi.PodMetrics{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Containers: []metricsapi.ContainerMetrics{
					{
						Name: "container1",
						Usage: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("10m"),
							corev1.ResourceMemory: resource.MustParse("25Mi"),
						},
					},
				},
			},
			podSpec: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "container1",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("1000m"),
									corev1.ResourceMemory: resource.MustParse("1Gi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("2000m"),
									corev1.ResourceMemory: resource.MustParse("2Gi"),
								},
							},
						},
					},
				},
			},
			expectedCPUReqPercent: 1,
			expectedMemReqPercent: 2,
			expectedStatus:        StatusOverProvisioned,
		},
		{
			name: "pod without requests",
			metrics: &metricsapi.PodMetrics{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Containers: []metricsapi.ContainerMetrics{
					{
						Name: "container1",
						Usage: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("128Mi"),
						},
					},
				},
			},
			podSpec: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:      "container1",
							Resources: corev1.ResourceRequirements{},
						},
					},
				},
			},
			expectedCPUReqPercent: -1,
			expectedMemReqPercent: -1,
			expectedStatus:        StatusNoLimit, // Both no limits and no requests
		},
		{
			name: "nil pod spec",
			metrics: &metricsapi.PodMetrics{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Containers: []metricsapi.ContainerMetrics{
					{
						Name: "container1",
						Usage: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("128Mi"),
						},
					},
				},
			},
			podSpec:               nil,
			expectedCPUReqPercent: -1,
			expectedMemReqPercent: -1,
			expectedStatus:        StatusNoLimit,
		},
		{
			name: "pod with multiple containers",
			metrics: &metricsapi.PodMetrics{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Containers: []metricsapi.ContainerMetrics{
					{
						Name: "container1",
						Usage: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("128Mi"),
						},
					},
					{
						Name: "container2",
						Usage: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("128Mi"),
						},
					},
				},
			},
			podSpec: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "container1",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("200m"),
									corev1.ResourceMemory: resource.MustParse("256Mi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("400m"),
									corev1.ResourceMemory: resource.MustParse("512Mi"),
								},
							},
						},
						{
							Name: "container2",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("200m"),
									corev1.ResourceMemory: resource.MustParse("256Mi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("400m"),
									corev1.ResourceMemory: resource.MustParse("512Mi"),
								},
							},
						},
					},
				},
			},
			// Total usage: 200m CPU, 256Mi memory
			// Total request: 400m CPU, 512Mi memory
			// Percentage: 50% for both
			expectedCPUReqPercent: 50,
			expectedMemReqPercent: 50,
			expectedStatus:        StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CalculatePodUtilization(tt.metrics, tt.podSpec, measuredResources)

			if result.CPURequestPercent != tt.expectedCPUReqPercent {
				t.Errorf("CPURequestPercent = %d, expected %d",
					result.CPURequestPercent, tt.expectedCPUReqPercent)
			}
			if result.MemoryRequestPercent != tt.expectedMemReqPercent {
				t.Errorf("MemoryRequestPercent = %d, expected %d",
					result.MemoryRequestPercent, tt.expectedMemReqPercent)
			}
			if result.Status != tt.expectedStatus {
				t.Errorf("Status = %s, expected %s", result.Status, tt.expectedStatus)
			}
		})
	}
}

func TestAddResourceList(t *testing.T) {
	tests := []struct {
		name     string
		list     corev1.ResourceList
		newList  corev1.ResourceList
		expected corev1.ResourceList
	}{
		{
			name:     "add to empty list",
			list:     corev1.ResourceList{},
			newList:  corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
			expected: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
		},
		{
			name:     "add to existing resource",
			list:     corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
			newList:  corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("200m")},
			expected: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("300m")},
		},
		{
			name:     "add new resource type",
			list:     corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
			newList:  corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("256Mi")},
			expected: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m"), corev1.ResourceMemory: resource.MustParse("256Mi")},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			addResourceList(tt.list, tt.newList)
			for name, expected := range tt.expected {
				if got, ok := tt.list[name]; !ok || got.Cmp(expected) != 0 {
					t.Errorf("resource %s = %v, expected %v", name, got, expected)
				}
			}
		})
	}
}

func TestMaxResourceList(t *testing.T) {
	tests := []struct {
		name     string
		list     corev1.ResourceList
		newList  corev1.ResourceList
		expected corev1.ResourceList
	}{
		{
			name:     "max to empty list",
			list:     corev1.ResourceList{},
			newList:  corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
			expected: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
		},
		{
			name:     "max when new is larger",
			list:     corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
			newList:  corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("200m")},
			expected: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("200m")},
		},
		{
			name:     "max when existing is larger",
			list:     corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("300m")},
			newList:  corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("200m")},
			expected: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("300m")},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			maxResourceList(tt.list, tt.newList)
			for name, expected := range tt.expected {
				if got, ok := tt.list[name]; !ok || got.Cmp(expected) != 0 {
					t.Errorf("resource %s = %v, expected %v", name, got, expected)
				}
			}
		})
	}
}

func TestPodUtilizationSorter(t *testing.T) {
	utilizations := []*PodUtilization{
		{Name: "pod1", Namespace: "default", CPUUsage: resource.MustParse("100m"), MemoryUsage: resource.MustParse("128Mi"), CPURequestPercent: 50, MemoryRequestPercent: 50},
		{Name: "pod2", Namespace: "default", CPUUsage: resource.MustParse("200m"), MemoryUsage: resource.MustParse("256Mi"), CPURequestPercent: 80, MemoryRequestPercent: 80},
		{Name: "pod3", Namespace: "default", CPUUsage: resource.MustParse("50m"), MemoryUsage: resource.MustParse("64Mi"), CPURequestPercent: 20, MemoryRequestPercent: 20},
		{Name: "pod4", Namespace: "default", CPUUsage: resource.MustParse("150m"), MemoryUsage: resource.MustParse("192Mi"), CPURequestPercent: -1, MemoryRequestPercent: -1}, // No requests
	}

	tests := []struct {
		name          string
		sortBy        string
		expectedOrder []string
	}{
		{
			name:          "sort by cpu-util descending",
			sortBy:        "cpu-util",
			expectedOrder: []string{"pod2", "pod1", "pod3", "pod4"}, // pod4 goes last due to no request
		},
		{
			name:          "sort by mem-util descending",
			sortBy:        "mem-util",
			expectedOrder: []string{"pod2", "pod1", "pod3", "pod4"},
		},
		{
			name:          "sort by cpu descending",
			sortBy:        "cpu",
			expectedOrder: []string{"pod2", "pod4", "pod1", "pod3"},
		},
		{
			name:          "sort by memory descending",
			sortBy:        "memory",
			expectedOrder: []string{"pod2", "pod4", "pod1", "pod3"},
		},
		{
			name:          "default sort by name",
			sortBy:        "",
			expectedOrder: []string{"pod1", "pod2", "pod3", "pod4"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Copy utilizations to avoid mutating between tests
			copyUtils := make([]*PodUtilization, len(utilizations))
			for i, u := range utilizations {
				copyU := *u
				copyUtils[i] = &copyU
			}

			sorter := NewPodUtilizationSorter(copyUtils, false, tt.sortBy)
			for i := 0; i < len(copyUtils)-1; i++ {
				for j := i + 1; j < len(copyUtils); j++ {
					if sorter.Less(j, i) {
						sorter.Swap(i, j)
					}
				}
			}

			for i, expected := range tt.expectedOrder {
				if copyUtils[i].Name != expected {
					t.Errorf("position %d: got %s, expected %s", i, copyUtils[i].Name, expected)
				}
			}
		})
	}
}
