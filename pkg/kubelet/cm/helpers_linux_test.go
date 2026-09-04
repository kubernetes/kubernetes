//go:build linux

/*
Copyright 2015 The Kubernetes Authors.

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

package cm

import (
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/utils/ptr"
)

// getResourceList returns a ResourceList with the
// specified cpu and memory resource values
func getResourceList(cpu, memory string) v1.ResourceList {
	res := v1.ResourceList{}
	if cpu != "" {
		res[v1.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[v1.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

// getResourceRequirements returns a ResourceRequirements object
func getResourceRequirements(requests, limits v1.ResourceList) v1.ResourceRequirements {
	res := v1.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

func TestResourceConfigForPod(t *testing.T) {
	const Mi = int64(1024 * 1024)
	const Gi = int64(1024 * 1024 * 1024)
	defaultQuotaPeriod := uint64(100 * time.Millisecond / time.Microsecond) // in microseconds
	tunedQuotaPeriod := uint64(5 * time.Millisecond / time.Microsecond)     // in microseconds
	tunedQuota := int64(1 * time.Millisecond / time.Microsecond)

	minShares := uint64(MinShares)
	burstableShares := MilliCPUToShares(100)
	memoryQuantity := resource.MustParse("200Mi")
	burstableMemory := memoryQuantity.Value()
	burstablePartialShares := MilliCPUToShares(200)
	burstableQuota := MilliCPUToQuota(200, int64(defaultQuotaPeriod))
	guaranteedShares := MilliCPUToShares(100)
	guaranteedQuota := MilliCPUToQuota(100, int64(defaultQuotaPeriod))
	guaranteedTunedQuota := MilliCPUToQuota(100, int64(tunedQuotaPeriod))
	memoryQuantity = resource.MustParse("100Mi")
	cpuNoLimit := int64(-1)
	guaranteedMemory := memoryQuantity.Value()
	testCases := []struct {
		description                        string
		pod                                *v1.Pod
		expected                           *ResourceConfig
		enforceCPULimits                   bool
		quotaPeriod                        uint64 // in microseconds
		podLevelResourcesEnabled           bool
		draNodeAllocatableResourcesEnabled bool
	}{
		{
			description: "besteffort",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("", ""), getResourceList("", "")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &minShares},
		},
		{
			description: "burstable-no-limits",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("", "")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstableShares},
		},
		{
			description: "burstable-with-limits",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &burstableQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory},
		},
		{
			description: "burstable-with-limits-no-cpu-enforcement",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
					},
				},
			},
			enforceCPULimits: false,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &cpuNoLimit, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory},
		},
		{
			description: "burstable-partial-limits",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("", "")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstablePartialShares},
		},
		{
			description: "burstable-with-limits-with-tuned-quota",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      tunedQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &tunedQuota, CPUPeriod: &tunedQuotaPeriod, Memory: &burstableMemory},
		},
		{
			description: "burstable-with-limits-no-cpu-enforcement-with-tuned-quota",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
					},
				},
			},
			enforceCPULimits: false,
			quotaPeriod:      tunedQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &cpuNoLimit, CPUPeriod: &tunedQuotaPeriod, Memory: &burstableMemory},
		},
		{
			description: "burstable-partial-limits-with-tuned-quota",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("", "")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      tunedQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstablePartialShares},
		},
		{
			description: "guaranteed",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &guaranteedQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &guaranteedMemory},
		},
		{
			description: "guaranteed-no-cpu-enforcement",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
						},
					},
				},
			},
			enforceCPULimits: false,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &cpuNoLimit, CPUPeriod: &defaultQuotaPeriod, Memory: &guaranteedMemory},
		},
		{
			description: "guaranteed-with-tuned-quota",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      tunedQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &guaranteedTunedQuota, CPUPeriod: &tunedQuotaPeriod, Memory: &guaranteedMemory},
		},
		{
			description: "guaranteed-no-cpu-enforcement-with-tuned-quota",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
						},
					},
				},
			},
			enforceCPULimits: false,
			quotaPeriod:      tunedQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &cpuNoLimit, CPUPeriod: &tunedQuotaPeriod, Memory: &guaranteedMemory},
		},
		{
			description: "burstable-partial-limits-with-init-containers",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100m"), getResourceList("100m", "100Mi")),
						},
						{
							Resources: getResourceRequirements(getResourceList("100m", "100m"), getResourceList("", "")),
						},
					},
					InitContainers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100m"), getResourceList("100m", "100Mi")),
						},
						{
							Resources: getResourceRequirements(getResourceList("100m", "100m"), getResourceList("", "")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      tunedQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstablePartialShares},
		},
		{
			description: "besteffort-with-pod-level-resources-enabled",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("", ""),
						Limits:   getResourceList("", ""),
					},
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("", ""), getResourceList("", "")),
						},
					},
				},
			},
			podLevelResourcesEnabled: true,
			enforceCPULimits:         true,
			quotaPeriod:              defaultQuotaPeriod,
			expected:                 &ResourceConfig{CPUShares: &minShares},
		},
		{
			description: "burstable-with-pod-level-requests",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("100m", "100Mi"),
					},
					Containers: []v1.Container{
						{
							Name: "Container with no resources",
						},
					},
				},
			},
			podLevelResourcesEnabled: true,
			enforceCPULimits:         true,
			quotaPeriod:              defaultQuotaPeriod,
			expected:                 &ResourceConfig{CPUShares: &burstableShares},
		},
		{
			description: "burstable-with-pod-and-container-level-requests",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("100m", "100Mi"),
					},
					Containers: []v1.Container{
						{
							Name:      "Container with resources",
							Resources: getResourceRequirements(getResourceList("10m", "50Mi"), getResourceList("", "")),
						},
					},
				},
			},
			podLevelResourcesEnabled: true,
			enforceCPULimits:         true,
			quotaPeriod:              defaultQuotaPeriod,
			expected:                 &ResourceConfig{CPUShares: &burstableShares},
		},
		{
			description: "burstable-with-pod-level-resources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("100m", "100Mi"),
						Limits:   getResourceList("200m", "200Mi"),
					},
					Containers: []v1.Container{
						{
							Name: "Container with no resources",
						},
					},
				},
			},
			podLevelResourcesEnabled: true,
			enforceCPULimits:         true,
			quotaPeriod:              defaultQuotaPeriod,
			expected:                 &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &burstableQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory},
		},
		{
			description: "burstable-with-pod-and-container-level-resources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("100m", "100Mi"),
						Limits:   getResourceList("200m", "200Mi"),
					},
					Containers: []v1.Container{
						{
							Name:      "Container with resources",
							Resources: getResourceRequirements(getResourceList("10m", "50Mi"), getResourceList("50m", "100Mi")),
						},
					},
				},
			},
			podLevelResourcesEnabled: true,
			enforceCPULimits:         true,
			quotaPeriod:              defaultQuotaPeriod,
			expected:                 &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &burstableQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory},
		},
		{
			description: "burstable-with-partial-pod-level-resources-limits",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("200m", "300Mi"),
					},
					Containers: []v1.Container{
						{
							Name:      "Container with guaranteed resources",
							Resources: getResourceRequirements(getResourceList("200m", "200Mi"), getResourceList("200m", "200Mi")),
						},
					},
				},
			},
			podLevelResourcesEnabled: true,
			enforceCPULimits:         true,
			quotaPeriod:              defaultQuotaPeriod,
			expected:                 &ResourceConfig{CPUShares: &burstablePartialShares, CPUQuota: &burstableQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory},
		},
		{
			description: "guaranteed-with-pod-level-resources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("100m", "100Mi"),
						Limits:   getResourceList("100m", "100Mi"),
					},
					Containers: []v1.Container{
						{
							Name: "Container with no resources",
						},
					},
				},
			},
			podLevelResourcesEnabled: true,
			enforceCPULimits:         true,
			quotaPeriod:              defaultQuotaPeriod,
			expected:                 &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &guaranteedQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &guaranteedMemory},
		},
		{
			description: "guaranteed-with-pod-and-container-level-resources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("100m", "100Mi"),
						Limits:   getResourceList("100m", "100Mi"),
					},
					Containers: []v1.Container{
						{
							Name:      "Container with resources",
							Resources: getResourceRequirements(getResourceList("10m", "50Mi"), getResourceList("50m", "100Mi")),
						},
					},
				},
			},
			podLevelResourcesEnabled: true,
			enforceCPULimits:         true,
			quotaPeriod:              defaultQuotaPeriod,
			expected:                 &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &guaranteedQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &guaranteedMemory},
		},
		{
			description: "guaranteed-pod-level-resources-with-init-containers",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("100m", "100Mi"),
						Limits:   getResourceList("100m", "100Mi"),
					},
					Containers: []v1.Container{
						{
							Name:      "Container with resources",
							Resources: getResourceRequirements(getResourceList("10m", "50Mi"), getResourceList("50m", "50Mi")),
						},
					},
					InitContainers: []v1.Container{
						{
							Name:      "Container with resources",
							Resources: getResourceRequirements(getResourceList("10m", "50Mi"), getResourceList("50m", "50Mi")),
						},
					},
				},
			},
			podLevelResourcesEnabled: true,
			enforceCPULimits:         true,
			quotaPeriod:              defaultQuotaPeriod,
			expected:                 &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &guaranteedQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &guaranteedMemory},
		},
		{
			description: "burstable-with-dra-direct-claims",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(
								getResourceList("1", "200Mi"), // requests
								getResourceList("3", "500Mi"), // limits
							),
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "direct-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("2"))},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			enforceCPULimits:                   true,
			quotaPeriod:                        defaultQuotaPeriod,
			// CPUShares: c1 request (1) + direct claim (2)
			// CPUQuota: c1 limit (3) + direct claim (2)
			expected: &ResourceConfig{CPUShares: new(MilliCPUToShares((1 + 2) * 1000)), CPUQuota: new(MilliCPUToQuota((3+2)*1000, int64(defaultQuotaPeriod))), CPUPeriod: &defaultQuotaPeriod, Memory: new(500 * Mi)},
		},
		{
			description: "burstable-with-dra-overhead-claims",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(
								getResourceList("1", "200Mi"), // requests
								getResourceList("3", "500Mi"), // limits
							),
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "overhead-claim",
							Containers:        []string{"c1"},
							Overhead: []v1.NodeAllocatableOverheadResources{
								{
									Name:         v1.ResourceMemory,
									PerPod:       new(resource.MustParse("300Mi")),
									PerContainer: new(resource.MustParse("100Mi")),
								},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			enforceCPULimits:                   true,
			quotaPeriod:                        defaultQuotaPeriod,
			// Memory: c1 limit (500) + perPodRef overhead (300) + perContainerRef overhead * 1 container (100 * 1)
			expected: &ResourceConfig{CPUShares: new(MilliCPUToShares(1 * 1000)), CPUQuota: new(MilliCPUToQuota(3*1000, int64(defaultQuotaPeriod))), CPUPeriod: &defaultQuotaPeriod, Memory: new((500 + 300 + 100) * Mi)},
		},
		{
			description: "burstable-with-dra-combined-mapping-and-overhead-claims",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(
								getResourceList("1", "200Mi"), // requests
								getResourceList("3", "500Mi"), // limits
							),
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "combined-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("2"))},
								{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("100Mi"))},
							},
							Overhead: []v1.NodeAllocatableOverheadResources{
								{
									Name:         v1.ResourceCPU,
									PerPod:       new(resource.MustParse("1")),
									PerContainer: new(resource.MustParse("1500m")),
								},
								{
									Name:         v1.ResourceMemory,
									PerPod:       new(resource.MustParse("300Mi")),
									PerContainer: new(resource.MustParse("100Mi")),
								},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			enforceCPULimits:                   true,
			quotaPeriod:                        defaultQuotaPeriod,
			// CPUShares: c1 request (1) + mapping (2) + overhead PerPod (1) + overhead PerContainer (1.5) = 5.5 CPUs
			// CPUQuota: c1 limit (3) + mapping (2) + overhead PerPod (1) + overhead PerContainer (1.5) = 7.5 CPUs
			// Memory: c1 limit (500) + mapping (100) + overhead PerPod (300) + overhead PerContainer (100) = 1000 Mi
			expected: &ResourceConfig{
				CPUShares: new(MilliCPUToShares(5500)),
				CPUQuota:  new(MilliCPUToQuota(7500, int64(defaultQuotaPeriod))),
				CPUPeriod: &defaultQuotaPeriod,
				Memory:    new(1000 * Mi),
			},
		},
		{
			description: "burstable-with-dra-feature-gate-disabled",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(
								getResourceList("1", "200Mi"), // requests
								getResourceList("3", "500Mi"), // limits
							),
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "direct-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("2"))},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: false,
			enforceCPULimits:                   true,
			quotaPeriod:                        defaultQuotaPeriod,
			// DRA values not added
			expected: &ResourceConfig{CPUShares: new(MilliCPUToShares(1 * 1000)), CPUQuota: new(MilliCPUToQuota(3*1000, int64(defaultQuotaPeriod))), CPUPeriod: &defaultQuotaPeriod, Memory: new(500 * Mi)},
		},
		{
			description: "burstable-with-dra-hugepages-only",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(
								getResourceList("1", "200Mi"), // requests
								getResourceList("3", "500Mi"), // limits
							),
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "hugepage-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceName(v1.ResourceHugePagesPrefix + "1Gi"), Quantity: new(resource.MustParse("2Gi"))},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			enforceCPULimits:                   true,
			quotaPeriod:                        defaultQuotaPeriod,
			expected: &ResourceConfig{
				CPUShares: new(MilliCPUToShares(1 * 1000)),
				CPUQuota:  new(MilliCPUToQuota(3*1000, int64(defaultQuotaPeriod))),
				CPUPeriod: &defaultQuotaPeriod,
				Memory:    ptr.To(500 * Mi),
				HugePageLimit: map[int64]int64{
					// HugePageLimit: DRA claim hugepage allocation of 2 x 1Gi
					1 * Gi: 2 * Gi,
				},
			},
		},
		{
			description: "burstable-with-dra-and-pod-level-resources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("4", "600Mi"),
						Limits:   getResourceList("8", "900Mi"),
					},
					Containers: []v1.Container{
						{
							Name: "c1",
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "direct-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("2"))},
								{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("300Mi"))},
							},
						},
					},
				},
			},
			podLevelResourcesEnabled:           true,
			draNodeAllocatableResourcesEnabled: true,
			enforceCPULimits:                   true,
			quotaPeriod:                        defaultQuotaPeriod,
			// pod-level resources override DRA CPU/Memory
			expected: &ResourceConfig{
				CPUShares: new(MilliCPUToShares(4 * 1000)),
				CPUQuota:  new(MilliCPUToQuota(8*1000, int64(defaultQuotaPeriod))),
				CPUPeriod: &defaultQuotaPeriod,
				Memory:    ptr.To(900 * Mi),
			},
		},
		{
			description: "burstable-with-multiple-dra-claims",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
							Resources: getResourceRequirements(
								getResourceList("100m", "200Mi"), // requests
								getResourceList("300m", "500Mi"), // limits
							),
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "direct-claim-1",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("100m"))},
							},
						},
						{
							ResourceClaimName: "direct-claim-2",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("200m"))},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			enforceCPULimits:                   true,
			quotaPeriod:                        defaultQuotaPeriod,
			// CPUShares: c1 request (100) + claim 1 CPU (100) + claim 2 CPU (200)
			// CPUQuota: c1 limit (300) + claim 1 CPU (100) + claim 2 CPU (200)
			expected: &ResourceConfig{
				CPUShares: new(MilliCPUToShares(100 + 100 + 200)),
				CPUQuota:  new(MilliCPUToQuota(300+100+200, int64(defaultQuotaPeriod))),
				CPUPeriod: &defaultQuotaPeriod,
				Memory:    ptr.To(500 * Mi),
			},
		},
		{
			description: "burstable-with-multiple-containers-referencing-overhead-claim",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
							Resources: getResourceRequirements(
								getResourceList("100m", "200Mi"), // requests
								getResourceList("300m", "500Mi"), // limits
							),
						},
						{
							Name: "c2",
							Resources: getResourceRequirements(
								getResourceList("200m", "300Mi"), // requests
								getResourceList("400m", "600Mi"), // limits
							),
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "overhead-claim",
							Containers:        []string{"c1", "c2"},
							Overhead: []v1.NodeAllocatableOverheadResources{
								{
									Name:         v1.ResourceMemory,
									PerPod:       new(resource.MustParse("200Mi")),
									PerContainer: new(resource.MustParse("100Mi")),
								},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			enforceCPULimits:                   true,
			quotaPeriod:                        defaultQuotaPeriod,
			expected: &ResourceConfig{
				CPUShares: new(MilliCPUToShares(100 + 200)),
				CPUQuota:  new(MilliCPUToQuota(300+400, int64(defaultQuotaPeriod))),
				CPUPeriod: &defaultQuotaPeriod,
				// Memory: c1 limit (500) + c2 limit (600) + perPodRef (200) + perContainerRef * 2 containers (100 * 2)
				Memory: ptr.To((500 + 600 + 200 + 100*2) * Mi),
			},
		},
		{
			description: "burstable-with-direct-claim-shared-by-two-containers",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
							Resources: getResourceRequirements(
								getResourceList("1", "200Mi"), // requests
								getResourceList("3", "500Mi"), // limits
							),
						},
						{
							Name: "c2",
							Resources: getResourceRequirements(
								getResourceList("2", "300Mi"), // requests
								getResourceList("4", "600Mi"), // limits
							),
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "shared-direct-claim",
							Containers:        []string{"c1", "c2"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("2"))},
								{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("400Mi"))},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			enforceCPULimits:                   true,
			quotaPeriod:                        defaultQuotaPeriod,
			expected: &ResourceConfig{
				// CPUShares: c1 request (1) + c2 request (2) + direct claim CPU (2) (shared direct resources are not multiplied)
				CPUShares: new(MilliCPUToShares((1 + 2 + 2) * 1000)),
				// CPUQuota: c1 limit (3) + c2 limit (4) + direct claim CPU (2)
				CPUQuota:  new(MilliCPUToQuota((3+4+2)*1000, int64(defaultQuotaPeriod))),
				CPUPeriod: &defaultQuotaPeriod,
				// Memory: c1 limit (500) + c2 limit (600) + direct claim memory (400)
				Memory: ptr.To((500 + 600 + 400) * Mi),
			},
		},
		{
			description: "besteffort-with-dra-direct-claims",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "direct-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("2"))},
								{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("400Mi"))},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			enforceCPULimits:                   true,
			quotaPeriod:                        defaultQuotaPeriod,
			// set min shares with DRA and without spec requests
			expected: &ResourceConfig{
				CPUShares: new(uint64(MinShares)),
				Memory:    nil,
			},
		},
		{
			description: "guaranteed-with-dra-direct-claims",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
							Resources: getResourceRequirements(
								getResourceList("1", "300Mi"), // requests
								getResourceList("1", "300Mi"), // limits
							),
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "direct-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("2"))},
								{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("400Mi"))},
							},
						},
					},
				},
			},
			draNodeAllocatableResourcesEnabled: true,
			enforceCPULimits:                   true,
			quotaPeriod:                        defaultQuotaPeriod,
			expected: &ResourceConfig{
				// CPUShares: c1 request (1) + direct claim CPU (2)
				CPUShares: new(MilliCPUToShares((1 + 2) * 1000)),
				// CPUQuota: c1 limit (1) + direct claim CPU (2)
				CPUQuota:  new(MilliCPUToQuota((1+2)*1000, int64(defaultQuotaPeriod))),
				CPUPeriod: &defaultQuotaPeriod,
				// Memory: c1 limit (300) + direct claim memory (400)
				Memory: ptr.To((300 + 400) * Mi),
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, testCase.podLevelResourcesEnabled)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.DRANodeAllocatableResources, testCase.draNodeAllocatableResourcesEnabled)
			actual := ResourceConfigForPod(testCase.pod, testCase.enforceCPULimits, testCase.quotaPeriod, false, kubeletconfig.NoneMemoryReservationPolicy)
			if !reflect.DeepEqual(actual.CPUPeriod, testCase.expected.CPUPeriod) {
				t.Errorf("cpu period not as expected. Expected: %v, Actual:%v", *testCase.expected.CPUPeriod, *actual.CPUPeriod)
			}
			if !reflect.DeepEqual(actual.CPUQuota, testCase.expected.CPUQuota) {
				t.Errorf("cpu quota not as expected. Expected: %v, Actual:%v", *testCase.expected.CPUQuota, *actual.CPUQuota)
			}
			if !reflect.DeepEqual(actual.CPUShares, testCase.expected.CPUShares) {
				t.Errorf("cpu shares not as expected. Expected: %v, Actual:%v", *testCase.expected.CPUShares, *actual.CPUShares)
			}
			if !reflect.DeepEqual(actual.Memory, testCase.expected.Memory) {
				t.Errorf("memory not as expected. Expected: %v, Actual:%v", *testCase.expected.Memory, *actual.Memory)
			}
			if len(testCase.expected.HugePageLimit) > 0 || len(actual.HugePageLimit) > 0 {
				if !reflect.DeepEqual(actual.HugePageLimit, testCase.expected.HugePageLimit) {
					t.Errorf("hugepage limit not as expected. Expected: %v, Actual:%v", testCase.expected.HugePageLimit, actual.HugePageLimit)
				}
			}
		})
	}
}

func TestMilliCPUToQuota(t *testing.T) {
	testCases := []struct {
		input  int64
		quota  int64
		period uint64
	}{
		{
			input:  int64(0),
			quota:  int64(0),
			period: uint64(0),
		},
		{
			input:  int64(5),
			quota:  int64(1000),
			period: uint64(100000),
		},
		{
			input:  int64(9),
			quota:  int64(1000),
			period: uint64(100000),
		},
		{
			input:  int64(10),
			quota:  int64(1000),
			period: uint64(100000),
		},
		{
			input:  int64(200),
			quota:  int64(20000),
			period: uint64(100000),
		},
		{
			input:  int64(500),
			quota:  int64(50000),
			period: uint64(100000),
		},
		{
			input:  int64(1000),
			quota:  int64(100000),
			period: uint64(100000),
		},
		{
			input:  int64(1500),
			quota:  int64(150000),
			period: uint64(100000),
		},
	}
	for _, testCase := range testCases {
		quota := MilliCPUToQuota(testCase.input, int64(testCase.period))
		if quota != testCase.quota {
			t.Errorf("Input %v and %v, expected quota %v, but got quota %v", testCase.input, testCase.period, testCase.quota, quota)
		}
	}
}

func TestHugePageLimits(t *testing.T) {
	Mi := int64(1024 * 1024)
	type inputStruct struct {
		key   string
		input string
	}

	testCases := []struct {
		name     string
		inputs   []inputStruct
		expected map[int64]int64
	}{
		{
			name: "no valid hugepages",
			inputs: []inputStruct{
				{
					key:   "2Mi",
					input: "128",
				},
			},
			expected: map[int64]int64{},
		},
		{
			name: "2Mi only",
			inputs: []inputStruct{
				{
					key:   v1.ResourceHugePagesPrefix + "2Mi",
					input: "128",
				},
			},
			expected: map[int64]int64{2 * Mi: 128},
		},
		{
			name: "2Mi and 4Mi",
			inputs: []inputStruct{
				{
					key:   v1.ResourceHugePagesPrefix + "2Mi",
					input: "128",
				},
				{
					key:   v1.ResourceHugePagesPrefix + strconv.FormatInt(2*Mi, 10),
					input: "256",
				},
				{
					key:   v1.ResourceHugePagesPrefix + "4Mi",
					input: "512",
				},
				{
					key:   "4Mi",
					input: "1024",
				},
			},
			expected: map[int64]int64{2 * Mi: 384, 4 * Mi: 512},
		},
	}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			resourceList := v1.ResourceList{}

			for _, input := range testcase.inputs {
				value, err := resource.ParseQuantity(input.input)
				if err != nil {
					t.Fatalf("error in parsing hugepages, value: %s", input.input)
				} else {
					resourceList[v1.ResourceName(input.key)] = value
				}
			}

			resultValue := HugePageLimits(resourceList)

			if !reflect.DeepEqual(testcase.expected, resultValue) {
				t.Errorf("unexpected result for HugePageLimits(), expected: %v, actual: %v", testcase.expected, resultValue)
			}

			// ensure ResourceConfigForPod uses HugePageLimits correctly internally
			p := v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: resourceList,
							},
						},
					},
				},
			}
			resultValuePod := ResourceConfigForPod(&p, false, 0, false, kubeletconfig.NoneMemoryReservationPolicy)
			if !reflect.DeepEqual(testcase.expected, resultValuePod.HugePageLimit) {
				t.Errorf("unexpected result for ResourceConfigForPod(), expected: %v, actual: %v", testcase.expected, resultValuePod)
			}
		})
	}
}

func TestResourceConfigForPodWithEnforceMemoryQoS(t *testing.T) {
	defaultQuotaPeriod := uint64(100 * time.Millisecond / time.Microsecond) // in microseconds
	tunedQuotaPeriod := uint64(5 * time.Millisecond / time.Microsecond)     // in microseconds

	minShares := uint64(MinShares)
	burstableShares := MilliCPUToShares(100)
	memoryQuantity := resource.MustParse("200Mi")
	burstableMemory := memoryQuantity.Value()
	burstablePartialShares := MilliCPUToShares(200)
	burstableQuota := MilliCPUToQuota(200, int64(defaultQuotaPeriod))
	guaranteedShares := MilliCPUToShares(100)
	guaranteedQuota := MilliCPUToQuota(100, int64(defaultQuotaPeriod))
	guaranteedTunedQuota := MilliCPUToQuota(100, int64(tunedQuotaPeriod))
	memoryQuantity = resource.MustParse("100Mi")
	cpuNoLimit := int64(-1)
	guaranteedMemory := memoryQuantity.Value()
	testCases := map[string]struct {
		pod              *v1.Pod
		expected         *ResourceConfig
		enforceCPULimits bool
		quotaPeriod      uint64 // in microseconds
	}{
		"besteffort": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("", ""), getResourceList("", "")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &minShares},
		},
		"burstable-no-limits": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("", "")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstableShares, Unified: map[string]string{"memory.low": "104857600"}},
		},
		"burstable-with-limits": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &burstableQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory, Unified: map[string]string{"memory.low": "104857600"}},
		},
		"burstable-with-limits-no-cpu-enforcement": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
					},
				},
			},
			enforceCPULimits: false,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &cpuNoLimit, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory, Unified: map[string]string{"memory.low": "104857600"}},
		},
		"burstable-partial-limits": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("", "")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstablePartialShares, Unified: map[string]string{"memory.low": "209715200"}},
		},
		"burstable-with-limits-with-tuned-quota": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      tunedQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &burstableQuota, CPUPeriod: &tunedQuotaPeriod, Memory: &burstableMemory, Unified: map[string]string{"memory.low": "104857600"}},
		},
		"burstable-with-limits-no-cpu-enforcement-with-tuned-quota": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
					},
				},
			},
			enforceCPULimits: false,
			quotaPeriod:      tunedQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &cpuNoLimit, CPUPeriod: &tunedQuotaPeriod, Memory: &burstableMemory, Unified: map[string]string{"memory.low": "104857600"}},
		},
		"burstable-partial-limits-with-tuned-quota": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
						},
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("", "")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      tunedQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &burstablePartialShares, Unified: map[string]string{"memory.low": "209715200"}},
		},
		"guaranteed": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &guaranteedQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &guaranteedMemory, Unified: map[string]string{"memory.min": "104857600"}},
		},
		"guaranteed-no-cpu-enforcement": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
						},
					},
				},
			},
			enforceCPULimits: false,
			quotaPeriod:      defaultQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &cpuNoLimit, CPUPeriod: &defaultQuotaPeriod, Memory: &guaranteedMemory, Unified: map[string]string{"memory.min": "104857600"}},
		},
		"guaranteed-with-tuned-quota": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
						},
					},
				},
			},
			enforceCPULimits: true,
			quotaPeriod:      tunedQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &guaranteedTunedQuota, CPUPeriod: &tunedQuotaPeriod, Memory: &guaranteedMemory, Unified: map[string]string{"memory.min": "104857600"}},
		},
		"guaranteed-no-cpu-enforcement-with-tuned-quota": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
						},
					},
				},
			},
			enforceCPULimits: false,
			quotaPeriod:      tunedQuotaPeriod,
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &cpuNoLimit, CPUPeriod: &tunedQuotaPeriod, Memory: &guaranteedMemory, Unified: map[string]string{"memory.min": "104857600"}},
		},
	}

	for testName, testCase := range testCases {

		actual := ResourceConfigForPod(testCase.pod, testCase.enforceCPULimits, testCase.quotaPeriod, true, kubeletconfig.TieredReservationMemoryReservationPolicy)

		if !reflect.DeepEqual(actual.Unified, testCase.expected.Unified) {
			t.Errorf("unexpected result, test: %v, unified not as expected", testName)
		}
	}
}

func TestApplyPodLevelMemoryHigh(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)

	t.Run("cpu-only pod-level resources does not set memory.high", func(t *testing.T) {
		pod := &v1.Pod{
			Spec: v1.PodSpec{
				Resources: &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("2"),
					},
				},
				Containers: []v1.Container{
					{Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", ""))},
				},
			},
		}
		rc := &ResourceConfig{}
		ApplyPodLevelMemoryHigh(pod, rc, 0.9)
		if rc.Unified != nil {
			t.Errorf("expected no Unified map, got %v", rc.Unified)
		}
	})

	// Overhead is included via PodResourcesOptions{ExcludeOverhead: false} (the default).
	t.Run("pod overhead is included in memory.high calculation", func(t *testing.T) {
		pod := &v1.Pod{
			Spec: v1.PodSpec{
				Overhead: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("64Mi"),
				},
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse("128Mi"),
						v1.ResourceCPU:    resource.MustParse("1"),
					},
					Limits: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse("256Mi"),
						v1.ResourceCPU:    resource.MustParse("1"),
					},
				},
				Containers: []v1.Container{
					{Resources: getResourceRequirements(getResourceList("1", "128Mi"), getResourceList("1", "256Mi"))},
				},
			},
		}
		rc := &ResourceConfig{}
		ApplyPodLevelMemoryHigh(pod, rc, 0.9)
		if rc.Unified == nil {
			t.Fatal("expected Unified map to be set")
		}
		// Without overhead: request=128Mi, limit=256Mi → memory.high based on 128Mi..256Mi
		rcNoOverhead := &ResourceConfig{}
		podNoOverhead := pod.DeepCopy()
		podNoOverhead.Spec.Overhead = nil
		ApplyPodLevelMemoryHigh(podNoOverhead, rcNoOverhead, 0.9)
		if rcNoOverhead.Unified == nil {
			t.Fatal("expected Unified map without overhead")
		}
		withOverhead, _ := strconv.ParseInt(rc.Unified[Cgroup2MemoryHigh], 10, 64)
		withoutOverhead, _ := strconv.ParseInt(rcNoOverhead.Unified[Cgroup2MemoryHigh], 10, 64)
		if withOverhead <= withoutOverhead {
			t.Errorf("memory.high with overhead (%d) should be greater than without overhead (%d)",
				withOverhead, withoutOverhead)
		}
	})

	t.Run("mixed declared/undeclared memory limits does not set memory.high", func(t *testing.T) {
		pod := &v1.Pod{
			Spec: v1.PodSpec{
				Resources: &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("2"),
					},
				},
				Containers: []v1.Container{
					{Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi"))},
					{Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("", ""))},
				},
			},
		}
		rc := &ResourceConfig{}
		ApplyPodLevelMemoryHigh(pod, rc, 0.9)
		if rc.Unified != nil {
			t.Errorf("expected no Unified map for mixed limits, got %v", rc.Unified)
		}
	})
}

func TestCPUSharesEqualAfterV2RoundTrip(t *testing.T) {
	testCases := []struct {
		name            string
		allocatedShares uint64
		readbackShares  uint64
		expected        bool
	}{
		{
			// 50m -> shares=51 -> weight=2 -> shares=28
			name:            "matches lossy cgroup v2 readback for 50m",
			allocatedShares: MilliCPUToShares(50),
			readbackShares:  28,
			expected:        true,
		},
		{
			name:            "does not match nearby non-roundtrip readback",
			allocatedShares: MilliCPUToShares(50),
			readbackShares:  27,
			expected:        false,
		},
		{
			// 100m -> shares=102 -> weight=4 -> shares=80
			name:            "matches lossy cgroup v2 readback for 100m",
			allocatedShares: MilliCPUToShares(100),
			readbackShares:  80,
			expected:        true,
		},
		{
			name:            "does not match identity when conversion is lossy",
			allocatedShares: MilliCPUToShares(100),
			readbackShares:  MilliCPUToShares(100),
			expected:        false,
		},
		{
			name:            "does not match unrelated higher readback",
			allocatedShares: MilliCPUToShares(50),
			readbackShares:  MilliCPUToShares(100),
			expected:        false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := CPUSharesEqualAfterV2RoundTrip(tc.allocatedShares, tc.readbackShares)
			if got != tc.expected {
				t.Fatalf("CPUSharesEqualAfterV2RoundTrip(%d, %d) = %t, want %t",
					tc.allocatedShares, tc.readbackShares, got, tc.expected)
			}
		})
	}
}

// TestMilliCPUToShares verifies conversion from Kubernetes milliCPU to Linux CFS cpu.shares.
// The formula (milliCPU * 1024 / 1000) derives from Docker's convention where 1 CPU = 1024
// shares (kernel default for a new cgroup). Bounds are enforced per the Linux kernel:
//   - MinShares (2): kernel/sched/sched.h line 427
//   - MaxShares (262144): kernel/sched/sched.h line 428
//
// Zero milliCPU returns MinShares (not 0) because the kernel treats 0 as "use default (1024)",
// and Kubernetes needs a distinct floor value to represent "no CPU request".
func TestMilliCPUToShares(t *testing.T) {
	testCases := []struct {
		name     string
		input    int64
		expected uint64
	}{
		{name: "zero milliCPU returns MinShares", input: 0, expected: MinShares},
		{name: "1 milliCPU rounds down to MinShares", input: 1, expected: MinShares},
		{name: "2 milliCPU computes to exactly MinShares (2*1024/1000=2)", input: 2, expected: MinShares},
		{name: "3 milliCPU computes to 3 shares (3*1024/1000=3)", input: 3, expected: 3},
		{name: "100 milliCPU maps to 102 shares", input: 100, expected: 102},
		{name: "500 milliCPU maps to 512 shares", input: 500, expected: 512},
		{name: "1000 milliCPU (1 CPU) maps to 1024 shares", input: 1000, expected: 1024},
		{name: "255999 milliCPU just below MaxShares", input: 255999, expected: uint64(255999 * 1024 / 1000)},
		{name: "256000 milliCPU hits exactly MaxShares", input: 256000, expected: MaxShares},
		{name: "256001 milliCPU clamped to MaxShares", input: 256001, expected: MaxShares},
		{name: "1000000 milliCPU clamped to MaxShares", input: 1000000, expected: MaxShares},
		{name: "negative milliCPU clamped to MinShares", input: -1, expected: MinShares},
		{name: "large negative clamped to MinShares", input: -1000000, expected: MinShares},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expected, MilliCPUToShares(tc.input))
		})
	}
}

// TestSharesToMilliCPU verifies the inverse conversion from Linux CFS cpu.shares back to
// Kubernetes milliCPU. Uses ceiling division — ceil(shares * 1000 / 1024) — to avoid
// losing precision on round-trips. Shares below MinShares return 0, indicating "unset"
// per kubelet convention for pods without explicit CPU requests.
func TestSharesToMilliCPU(t *testing.T) {
	testCases := []struct {
		name     string
		input    int64
		expected int64
	}{
		{name: "zero shares returns 0 (below MinShares threshold)", input: 0, expected: 0},
		{name: "1 share returns 0 (below MinShares)", input: 1, expected: 0},
		{name: "MinShares (2) returns ceil(2*1000/1024)=2", input: int64(MinShares), expected: 2},
		{name: "3 shares returns ceil(3*1000/1024)=3", input: 3, expected: 3},
		{name: "102 shares returns ceil(102*1000/1024)=100", input: 102, expected: 100},
		{name: "512 shares returns ceil(512*1000/1024)=500", input: 512, expected: 500},
		{name: "1024 shares returns ceil(1024*1000/1024)=1000", input: 1024, expected: 1000},
		{name: "MaxShares (262144) returns ceil(262144*1000/1024)=256000", input: int64(MaxShares), expected: 256000},
		{name: "negative shares returns 0", input: -1, expected: 0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expected, SharesToMilliCPU(tc.input))
		})
	}
}

// TestMilliCPUToSharesRoundTrip verifies lossless conversion milliCPU → shares → milliCPU
// for standard values. This property is critical because kubelet writes cpu.shares during
// pod admission (MilliCPUToShares) and reads them back for status reporting (SharesToMilliCPU).
func TestMilliCPUToSharesRoundTrip(t *testing.T) {
	testCases := []struct {
		name     string
		milliCPU int64
	}{
		{name: "100m", milliCPU: 100},
		{name: "250m", milliCPU: 250},
		{name: "500m", milliCPU: 500},
		{name: "1000m (1 CPU)", milliCPU: 1000},
		{name: "2000m", milliCPU: 2000},
		{name: "4000m", milliCPU: 4000},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			shares := MilliCPUToShares(tc.milliCPU)
			recovered := SharesToMilliCPU(int64(shares))
			assert.Equal(t, tc.milliCPU, recovered)
		})
	}
}

// TestQuotaToMilliCPU verifies conversion from CFS bandwidth control quota/period to
// Kubernetes milliCPU. Per the CFS bandwidth control specification
// (Documentation/scheduler/sched-bwc.txt), quota=-1 means "unlimited" (no CPU limit)
// and is mapped to 0 milliCPU. The formula (quota * 1000 / period) uses integer division,
// so sub-milliCPU fractions are truncated.
func TestQuotaToMilliCPU(t *testing.T) {
	testCases := []struct {
		name     string
		quota    int64
		period   int64
		expected int64
	}{
		{name: "unlimited quota (-1) returns 0", quota: -1, period: 100000, expected: 0},
		{name: "zero quota returns 0", quota: 0, period: 100000, expected: 0},
		{name: "quota=1000 period=100000 returns 10 (MinMilliCPULimit)", quota: 1000, period: 100000, expected: 10},
		{name: "quota=5000 period=100000 returns 50", quota: 5000, period: 100000, expected: 50},
		{name: "quota=50000 period=100000 returns 500", quota: 50000, period: 100000, expected: 500},
		{name: "quota=100000 period=100000 returns 1000 (1 CPU)", quota: 100000, period: 100000, expected: 1000},
		{name: "quota=200000 period=100000 returns 2000 (2 CPUs)", quota: 200000, period: 100000, expected: 2000},
		{name: "non-standard period: quota=5000 period=5000 returns 1000", quota: 5000, period: 5000, expected: 1000},
		{name: "sub-milliCPU truncation: quota=1 period=100000 returns 0", quota: 1, period: 100000, expected: 0},
		{name: "near-minimum: quota=100 period=100000 returns 1", quota: 100, period: 100000, expected: 1},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expected, QuotaToMilliCPU(tc.quota, tc.period))
		})
	}
}

// TestCPURequestsFromConfig verifies reconstruction of Kubernetes CPU request quantities
// from a ResourceConfig's CPUShares field. Used by kubelet to report actual allocated
// resources in pod status. Returns nil for nil config or zero shares, indicating no
// CPU request was set.
func TestCPURequestsFromConfig(t *testing.T) {
	testCases := []struct {
		name             string
		config           *ResourceConfig
		expectedMilliCPU *int64
	}{
		{name: "nil config returns nil", config: nil, expectedMilliCPU: nil},
		{name: "zero shares returns nil", config: &ResourceConfig{CPUShares: ptr.To[uint64](0)}, expectedMilliCPU: nil},
		{name: "MinShares (2) returns 2m", config: &ResourceConfig{CPUShares: ptr.To[uint64](MinShares)}, expectedMilliCPU: ptr.To[int64](2)},
		{name: "1024 shares returns 1000m", config: &ResourceConfig{CPUShares: ptr.To[uint64](1024)}, expectedMilliCPU: ptr.To[int64](1000)},
		{name: "512 shares returns 500m", config: &ResourceConfig{CPUShares: ptr.To[uint64](512)}, expectedMilliCPU: ptr.To[int64](500)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := CPURequestsFromConfig(tc.config)
			if tc.expectedMilliCPU == nil {
				assert.Nil(t, result)
			} else {
				assert.NotNil(t, result)
				assert.Equal(t, *tc.expectedMilliCPU, result.MilliValue())
			}
		})
	}
}

// TestCPULimitsFromConfig verifies reconstruction of Kubernetes CPU limit quantities from
// ResourceConfig's CPUQuota and CPUPeriod fields. Quota=-1 represents "unlimited" per
// CFS bandwidth control and returns nil. Zero period is treated as invalid (returns nil)
// to avoid division by zero.
func TestCPULimitsFromConfig(t *testing.T) {
	testCases := []struct {
		name             string
		config           *ResourceConfig
		expectedMilliCPU *int64
	}{
		{name: "nil config returns nil", config: nil, expectedMilliCPU: nil},
		{name: "zero period returns nil", config: &ResourceConfig{CPUQuota: ptr.To[int64](100000), CPUPeriod: ptr.To[uint64](0)}, expectedMilliCPU: nil},
		{name: "unlimited quota (-1) returns nil", config: &ResourceConfig{CPUQuota: ptr.To[int64](-1), CPUPeriod: ptr.To[uint64](100000)}, expectedMilliCPU: nil},
		{name: "50000/100000 returns 500m", config: &ResourceConfig{CPUQuota: ptr.To[int64](50000), CPUPeriod: ptr.To[uint64](100000)}, expectedMilliCPU: ptr.To[int64](500)},
		{name: "100000/100000 returns 1000m", config: &ResourceConfig{CPUQuota: ptr.To[int64](100000), CPUPeriod: ptr.To[uint64](100000)}, expectedMilliCPU: ptr.To[int64](1000)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := CPULimitsFromConfig(tc.config)
			if tc.expectedMilliCPU == nil {
				assert.Nil(t, result)
			} else {
				assert.NotNil(t, result)
				assert.Equal(t, *tc.expectedMilliCPU, result.MilliValue())
			}
		})
	}
}

// TestMemoryLimitsFromConfig verifies reconstruction of Kubernetes memory limit quantities
// from ResourceConfig's Memory field. Zero or negative values return nil per kubelet
// convention for absent memory limits (kernel uses -1 for unlimited memory cgroup).
func TestMemoryLimitsFromConfig(t *testing.T) {
	testCases := []struct {
		name          string
		config        *ResourceConfig
		expectedBytes *int64
	}{
		{name: "nil config returns nil", config: nil, expectedBytes: nil},
		{name: "zero memory returns nil", config: &ResourceConfig{Memory: ptr.To[int64](0)}, expectedBytes: nil},
		{name: "negative memory returns nil", config: &ResourceConfig{Memory: ptr.To[int64](-1)}, expectedBytes: nil},
		{name: "1Gi memory returns quantity", config: &ResourceConfig{Memory: ptr.To[int64](1073741824)}, expectedBytes: ptr.To[int64](1073741824)},
		{name: "512Mi memory returns quantity", config: &ResourceConfig{Memory: ptr.To[int64](536870912)}, expectedBytes: ptr.To[int64](536870912)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := MemoryLimitsFromConfig(tc.config)
			if tc.expectedBytes == nil {
				assert.Nil(t, result)
			} else {
				assert.NotNil(t, result)
				assert.Equal(t, *tc.expectedBytes, result.Value())
			}
		})
	}
}

// TestNodeAllocatableRoot verifies construction of the node allocatable cgroup path.
// Per the node allocatable design (KEP-726), kubelet isolates pod workloads under a
// "kubepods" subtree when cgroupsPerQOS is enabled. The cgroupfs driver uses literal
// path components; the systemd driver converts to systemd slice naming convention
// (component.slice). This separation ensures system-critical processes are protected
// from pod resource consumption.
func TestNodeAllocatableRoot(t *testing.T) {
	testCases := []struct {
		name          string
		cgroupRoot    string
		cgroupsPerQOS bool
		cgroupDriver  string
		expected      string
	}{
		{name: "cgroupfs with cgroupsPerQOS from root", cgroupRoot: "/", cgroupsPerQOS: true, cgroupDriver: "cgroupfs", expected: "/kubepods"},
		{name: "cgroupfs without cgroupsPerQOS from root", cgroupRoot: "/", cgroupsPerQOS: false, cgroupDriver: "cgroupfs", expected: "/"},
		{name: "cgroupfs with cgroupsPerQOS custom root", cgroupRoot: "/custom", cgroupsPerQOS: true, cgroupDriver: "cgroupfs", expected: "/custom/kubepods"},
		{name: "cgroupfs without cgroupsPerQOS custom root", cgroupRoot: "/custom", cgroupsPerQOS: false, cgroupDriver: "cgroupfs", expected: "/custom"},
		{name: "systemd with cgroupsPerQOS from root", cgroupRoot: "/", cgroupsPerQOS: true, cgroupDriver: "systemd", expected: "/kubepods.slice"},
		{name: "empty root with cgroupsPerQOS cgroupfs", cgroupRoot: "", cgroupsPerQOS: true, cgroupDriver: "cgroupfs", expected: "/kubepods"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expected, NodeAllocatableRoot(tc.cgroupRoot, tc.cgroupsPerQOS, tc.cgroupDriver))
		})
	}
}

// TestGetCgroupProcs verifies reading PIDs from a cgroup's cgroup.procs file.
// Per cgroup lifecycle semantics, a missing procs file is not an error (the cgroup
// may have been removed between listing and reading). Malformed content or permission
// errors are surfaced as errors to distinguish configuration problems from normal lifecycle.
func TestGetCgroupProcs(t *testing.T) {
	testCases := []struct {
		name        string
		setup       func(t *testing.T) string
		expected    []int
		expectError bool
	}{
		{
			name: "non-existent cgroup.procs returns empty slice",
			setup: func(t *testing.T) string {
				return t.TempDir()
			},
			expected:    []int{},
			expectError: false,
		},
		{
			name: "valid PIDs parsed correctly",
			setup: func(t *testing.T) string {
				dir := t.TempDir()
				err := os.WriteFile(filepath.Join(dir, "cgroup.procs"), []byte("1\n22\n333\n4444\n"), 0644)
				require.NoError(t, err)
				return dir
			},
			expected:    []int{1, 22, 333, 4444},
			expectError: false,
		},
		{
			name: "empty file returns empty slice",
			setup: func(t *testing.T) string {
				dir := t.TempDir()
				err := os.WriteFile(filepath.Join(dir, "cgroup.procs"), []byte(""), 0644)
				require.NoError(t, err)
				return dir
			},
			expected:    []int{},
			expectError: false,
		},
		{
			name: "empty lines between PIDs are skipped",
			setup: func(t *testing.T) string {
				dir := t.TempDir()
				err := os.WriteFile(filepath.Join(dir, "cgroup.procs"), []byte("1\n\n2\n\n3\n"), 0644)
				require.NoError(t, err)
				return dir
			},
			expected:    []int{1, 2, 3},
			expectError: false,
		},
		{
			name: "non-integer content returns error",
			setup: func(t *testing.T) string {
				dir := t.TempDir()
				err := os.WriteFile(filepath.Join(dir, "cgroup.procs"), []byte("1\nnot_a_pid\n3\n"), 0644)
				require.NoError(t, err)
				return dir
			},
			expected:    nil,
			expectError: true,
		},
		{
			name: "single PID without trailing newline",
			setup: func(t *testing.T) string {
				dir := t.TempDir()
				err := os.WriteFile(filepath.Join(dir, "cgroup.procs"), []byte("42"), 0644)
				require.NoError(t, err)
				return dir
			},
			expected:    []int{42},
			expectError: false,
		},
		{
			name: "permission denied returns error",
			setup: func(t *testing.T) string {
				dir := t.TempDir()
				procsPath := filepath.Join(dir, "cgroup.procs")
				err := os.WriteFile(procsPath, []byte("1\n"), 0644)
				require.NoError(t, err)
				err = os.Chmod(procsPath, 0000)
				require.NoError(t, err)
				t.Cleanup(func() {
					if err := os.Chmod(procsPath, 0644); err != nil {
						t.Logf("cleanup chmod failed: %v", err)
					}
				})
				return dir
			},
			expected:    nil,
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.name == "permission denied returns error" && os.Getuid() == 0 {
				t.Skip("skipping permission test when running as root")
			}
			dir := tc.setup(t)
			result, err := getCgroupProcs(dir)
			if tc.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tc.expected, result)
			}
		})
	}
}

// TestGetPodCgroupNameSuffix verifies generation of pod cgroup leaf names. Each pod gets
// a cgroup named "pod<UID>" as the leaf directory under its QoS-class cgroup. This naming
// convention allows kubelet to locate and manage per-pod cgroups throughout the pod
// lifecycle (creation, resource updates, cleanup on termination).
func TestGetPodCgroupNameSuffix(t *testing.T) {
	testCases := []struct {
		name     string
		podUID   types.UID
		expected string
	}{
		{name: "standard UID", podUID: types.UID("abc123-def456"), expected: "podabc123-def456"},
		{name: "full UUID format", podUID: types.UID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"), expected: "poda1b2c3d4-e5f6-7890-abcd-ef1234567890"},
		{name: "empty UID returns prefix only", podUID: types.UID(""), expected: "pod"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expected, GetPodCgroupNameSuffix(tc.podUID))
		})
	}
}
