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

package qos

import (
	"fmt"
	"strconv"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/apis/scheduling"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

const (
	standardMemoryAmount = 8000000000
)

var (
	cpuLimit = v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "cpu-limit",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU): resource.MustParse("10"),
						},
					},
				},
			},
		},
	}

	memoryLimitCPURequest = v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "memory-limit-cpu-request",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU): resource.MustParse("0"),
						},
						Limits: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
						},
					},
				},
			},
		},
	}

	zeroMemoryLimit = v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "zero-memory-limit",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("0"),
						},
					},
				},
			},
		},
	}

	noRequestLimit = v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      "no-request-limit",
					Resources: v1.ResourceRequirements{},
				},
			},
		},
	}

	equalRequestLimitCPUMemory = v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "equal-request-limit-cpu-memory",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("5m"),
						},
						Limits: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("5m"),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
						},
					},
				},
			},
		},
	}

	cpuUnlimitedMemoryLimitedWithRequests = v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "cpu-unlimited-memory-limited-with-requests",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): resource.MustParse(strconv.FormatInt(standardMemoryAmount/2, 10)),
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("5m"),
						},
						Limits: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
						},
					},
				},
			},
		},
	}

	requestNoLimit = v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "request-no-limit",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): resource.MustParse(strconv.FormatInt(standardMemoryAmount-1, 10)),
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("5m"),
						},
					},
				},
			},
		},
	}

	systemCritical = scheduling.SystemCriticalPriority

	clusterCritical = v1.Pod{
		Spec: v1.PodSpec{
			PriorityClassName: scheduling.SystemClusterCritical,
			Priority:          &systemCritical,
			Containers: []v1.Container{
				{
					Name:      "cluster-critical",
					Resources: v1.ResourceRequirements{},
				},
			},
		},
	}

	systemNodeCritical = scheduling.SystemCriticalPriority + 1000

	nodeCritical = v1.Pod{
		Spec: v1.PodSpec{
			PriorityClassName: scheduling.SystemNodeCritical,
			Priority:          &systemNodeCritical,
			Containers: []v1.Container{
				{
					Name:      "node-critical",
					Resources: v1.ResourceRequirements{},
				},
			},
		},
	}
	sampleDefaultMemRequest    = resource.MustParse(strconv.FormatInt(standardMemoryAmount/8, 10))
	sampleDefaultMemLimit      = resource.MustParse(strconv.FormatInt(1000+(standardMemoryAmount/8), 10))
	sampleDefaultPodMemRequest = resource.MustParse(strconv.FormatInt(standardMemoryAmount/4, 10))
	sampleDefaultPodMemLimit   = resource.MustParse(strconv.FormatInt(1000+(standardMemoryAmount/4), 10))

	sampleContainer = v1.Container{
		Name: "main-1",
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): sampleDefaultMemRequest,
			},
			Limits: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): sampleDefaultMemLimit,
			},
		},
	}

	burstableUniqueContainerPod = v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "burstable-unique-container",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): sampleDefaultMemRequest,
						},
						Limits: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): sampleDefaultMemLimit,
						},
					},
				},
			},
		},
	}

	sampleInitContainer = v1.Container{
		Name: "init-container",
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): sampleDefaultMemRequest,
			},
			Limits: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): sampleDefaultMemLimit,
			},
		},
	}
	restartPolicyAlways    = v1.ContainerRestartPolicyAlways
	sampleSidecarContainer = v1.Container{
		Name:          "sidecar-container",
		RestartPolicy: &restartPolicyAlways,
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): sampleDefaultMemRequest,
			},
			Limits: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): sampleDefaultMemLimit,
			},
		},
	}

	sampleSmallSidecarContainer = v1.Container{
		Name:          "sidecar-small-container",
		RestartPolicy: &restartPolicyAlways,
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): resource.MustParse(strconv.FormatInt(standardMemoryAmount/20, 10)),
			},
			Limits: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): sampleDefaultMemLimit,
			},
		},
	}

	sampleBigSidecarContainer = v1.Container{
		Name:          "sidecar-big-container",
		RestartPolicy: &restartPolicyAlways,
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): resource.MustParse(strconv.FormatInt(standardMemoryAmount/2, 10)),
			},
			Limits: v1.ResourceList{
				v1.ResourceName(v1.ResourceMemory): sampleDefaultMemLimit,
			},
		},
	}

	burstableMixedUniqueMainContainerPod = v1.Pod{
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				sampleInitContainer,
			}, Containers: []v1.Container{
				sampleContainer,
			},
		},
	}

	burstableMixedMultiContainerSameRequestPod = v1.Pod{
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				sampleInitContainer, sampleSidecarContainer,
			}, Containers: []v1.Container{
				sampleContainer,
			},
		},
	}

	burstableMixedMultiContainerSmallSidecarPod = v1.Pod{
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				sampleInitContainer, sampleSmallSidecarContainer,
			}, Containers: []v1.Container{
				sampleContainer,
			},
		},
	}

	burstableMixedMultiContainerBigSidecarContainerPod = v1.Pod{
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				sampleInitContainer, sampleBigSidecarContainer,
			}, Containers: []v1.Container{
				sampleContainer,
			},
		},
	}

	// Pod definitions with their resource specifications are defined in this section.
	// TODO(ndixita): cleanup the tests to create a method that generates pod
	// definitions based on input resource parameters, replacing the current
	// approach of individual pod variables.
	guaranteedPodResourcesNoContainerResources = v1.Pod{
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemRequest,
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemRequest,
				},
			},
			Containers: []v1.Container{
				{
					Name:      "no-request-limit-1",
					Resources: v1.ResourceRequirements{},
				},
				{
					Name:      "no-request-limit-2",
					Resources: v1.ResourceRequirements{},
				},
			},
		},
	}

	guaranteedPodResourcesEqualContainerRequests = v1.Pod{
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemRequest,
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemRequest,
				},
			},
			Containers: []v1.Container{
				{
					Name: "guaranteed-container-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
					},
				},
				{
					Name: "guaranteed-container-2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
					},
				},
			},
		},
	}

	guaranteedPodResourcesUnequalContainerRequests = v1.Pod{
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemRequest,
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemRequest,
				},
			},
			Containers: []v1.Container{
				{
					Name: "burstable-container",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("3m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemLimit,
						},
					},
				},
				{
					Name:      "best-effort-container",
					Resources: v1.ResourceRequirements{},
				},
			},
		},
	}

	burstablePodResourcesNoContainerResources = v1.Pod{
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemRequest,
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemLimit,
				},
			},
			Containers: []v1.Container{
				{
					Name:      "no-request-limit-1",
					Resources: v1.ResourceRequirements{},
				},
				{
					Name:      "no-request-limit-2",
					Resources: v1.ResourceRequirements{},
				},
			},
		},
	}

	burstablePodResourcesEqualContainerRequests = v1.Pod{
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemRequest,
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemLimit,
				},
			},
			Containers: []v1.Container{
				{
					Name: "guaranteed-container-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
					},
				},
				{
					Name: "guaranteed-container-2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
					},
				},
			},
		},
	}

	burstablePodResourcesUnequalContainerRequests = v1.Pod{
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemRequest,
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemLimit,
				},
			},
			Containers: []v1.Container{
				{
					Name: "burstable-container",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("3m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
					},
				},
				{
					Name:      "best-effort-container",
					Resources: v1.ResourceRequirements{},
				},
			},
		},
	}

	burstablePodResourcesNoContainerResourcesWithSidecar = v1.Pod{
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemRequest,
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemLimit,
				},
			},
			Containers: []v1.Container{
				{
					Name:      "no-request-limit",
					Resources: v1.ResourceRequirements{},
				},
			},
			InitContainers: []v1.Container{
				{
					Name:          "no-request-limit-sidecar",
					Resources:     v1.ResourceRequirements{},
					RestartPolicy: &restartPolicyAlways,
				},
			},
		},
	}

	burstablePodResourcesEqualContainerRequestsWithSidecar = v1.Pod{
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemRequest,
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemLimit,
				},
			},
			Containers: []v1.Container{
				{
					Name: "burstable-container",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemLimit,
						},
					},
				},
			},
			InitContainers: []v1.Container{
				{
					Name: "burstable-sidecar",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemRequest,
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultMemLimit,
						},
					},
					RestartPolicy: &restartPolicyAlways,
				},
			},
		},
	}

	burstablePodResourcesUnequalContainerRequestsWithSidecar = v1.Pod{
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: resource.MustParse("2000000000"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("5m"),
					v1.ResourceMemory: sampleDefaultPodMemLimit,
				},
			},
			Containers: []v1.Container{
				{
					Name: "burstable-container-1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: resource.MustParse("1000000000"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultPodMemLimit,
						},
					},
				},
				{
					Name: "burstable-container-2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: resource.MustParse("500000000"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultPodMemLimit,
						},
					},
				},
			},
			InitContainers: []v1.Container{
				{
					Name: "burstable-sidecar",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: resource.MustParse("200000000"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: sampleDefaultPodMemLimit,
						},
					},
					RestartPolicy: &restartPolicyAlways,
				},
			},
		},
	}
)

type lowHighOOMScoreAdjTest struct {
	lowOOMScoreAdj  int
	highOOMScoreAdj int
}
type oomTest struct {
	pod                               *v1.Pod
	memoryCapacity                    int64
	lowHighOOMScoreAdj                map[string]lowHighOOMScoreAdjTest // [container-name] : min and max oom_score_adj score the container should be assigned.
	podLevelResourcesFeatureEnabled   bool
	enableDRANodeAllocatableResources bool
}

type draMemAllocation struct {
	containers           []string
	mapping              string
	perContainerOverhead string
	perPodOverhead       string
}

type podParams struct {
	c1Mem       string
	c2Mem       string
	sidecarMem  string
	podLevelMem string
	draAlloc    draMemAllocation
}

func TestGetContainerOOMScoreAdjust(t *testing.T) {
	makeTestPodForDRA := func(p podParams) *v1.Pod {
		pod := &v1.Pod{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "c1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse(p.c1Mem),
							},
						},
					},
				},
			},
		}

		if p.c2Mem != "" {
			pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
				Name: "c2",
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse(p.c2Mem),
					},
				},
			})
		}

		if p.sidecarMem != "" {
			pod.Spec.InitContainers = append(pod.Spec.InitContainers, v1.Container{
				Name: "s1",
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse(p.sidecarMem),
					},
				},
				RestartPolicy: &restartPolicyAlways,
			})
		}

		if p.podLevelMem != "" {
			pod.Spec.Resources = &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse(p.podLevelMem),
				},
			}
		}

		if len(p.draAlloc.containers) > 0 {
			status := v1.NodeAllocatableResourceClaimStatus{
				ResourceClaimName: "dra-claim",
				Containers:        p.draAlloc.containers,
			}
			if p.draAlloc.mapping != "" {
				status.Mapping = []v1.NodeAllocatableMappedResources{
					{
						Name:     v1.ResourceMemory,
						Quantity: new(resource.MustParse(p.draAlloc.mapping)),
					},
				}
			}
			var overheads []v1.NodeAllocatableOverheadResources
			if p.draAlloc.perContainerOverhead != "" {
				overheads = append(overheads, v1.NodeAllocatableOverheadResources{
					Name:         v1.ResourceMemory,
					PerContainer: new(resource.MustParse(p.draAlloc.perContainerOverhead)),
				})
			}
			if p.draAlloc.perPodOverhead != "" {
				overheads = append(overheads, v1.NodeAllocatableOverheadResources{
					Name:   v1.ResourceMemory,
					PerPod: new(resource.MustParse(p.draAlloc.perPodOverhead)),
				})
			}
			status.Overhead = overheads
			pod.Status.NodeAllocatableResourceClaimStatuses = []v1.NodeAllocatableResourceClaimStatus{status}
		}

		return pod
	}

	oomTests := map[string]oomTest{
		"cpu-limit": {
			pod:            &cpuLimit,
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"cpu-limit": {lowOOMScoreAdj: 999, highOOMScoreAdj: 999},
			},
		},
		"memory-limit-cpu-request": {
			pod:            &memoryLimitCPURequest,
			memoryCapacity: 8000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"memory-limit-cpu-request": {lowOOMScoreAdj: 999, highOOMScoreAdj: 999},
			},
		},
		"zero-memory-limit": {
			pod:            &zeroMemoryLimit,
			memoryCapacity: 7230457451,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"zero-memory-limit": {lowOOMScoreAdj: 1000, highOOMScoreAdj: 1000},
			},
		},
		"no-request-limit": {
			pod:            &noRequestLimit,
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"no-request-limit": {lowOOMScoreAdj: 1000, highOOMScoreAdj: 1000},
			},
		},
		"equal-request-limit-cpu-memory": {
			pod:            &equalRequestLimitCPUMemory,
			memoryCapacity: 123456789,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"equal-request-limit-cpu-memory": {lowOOMScoreAdj: -997, highOOMScoreAdj: -997},
			},
		},
		"cpu-unlimited-memory-limited-with-requests": {
			pod:            &cpuUnlimitedMemoryLimitedWithRequests,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"cpu-unlimited-memory-limited-with-requests": {lowOOMScoreAdj: 495, highOOMScoreAdj: 505},
			},
		},
		"request-no-limit": {
			pod:            &requestNoLimit,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"request-no-limit": {lowOOMScoreAdj: 3, highOOMScoreAdj: 3},
			},
		},
		"cluster-critical": {
			pod:            &clusterCritical,
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"cluster-critical": {lowOOMScoreAdj: 1000, highOOMScoreAdj: 1000},
			},
		},
		"node-critical": {
			pod:            &nodeCritical,
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"node-critical": {lowOOMScoreAdj: -997, highOOMScoreAdj: -997},
			},
		},
		"burstable-unique-container-pod": {
			pod:            &burstableUniqueContainerPod,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"burstable-unique-container": {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
			},
		},
		"burstable-mixed-unique-main-container-pod": {
			pod:            &burstableMixedUniqueMainContainerPod,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"init-container": {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
				"main-1":         {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
			},
		},
		"burstable-mixed-multi-container-small-sidecar-pod": {
			pod:            &burstableMixedMultiContainerSmallSidecarPod,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"init-container":          {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
				"sidecar-small-container": {lowOOMScoreAdj: 875, highOOMScoreAdj: 875},
				"main-1":                  {lowOOMScoreAdj: 875, highOOMScoreAdj: 875},
			},
		},
		"burstable-mixed-multi-container-sample-request-pod": {
			pod:            &burstableMixedMultiContainerSameRequestPod,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"init-container":    {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
				"sidecar-container": {lowOOMScoreAdj: 875, highOOMScoreAdj: 875},
				"main-1":            {lowOOMScoreAdj: 875, highOOMScoreAdj: 875},
			},
		},
		"burstable-mixed-multi-container-big-sidecar-container-pod": {
			pod:            &burstableMixedMultiContainerBigSidecarContainerPod,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"init-container":        {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
				"sidecar-big-container": {lowOOMScoreAdj: 500, highOOMScoreAdj: 500},
				"main-1":                {lowOOMScoreAdj: 875, highOOMScoreAdj: 875},
			},
		},
		"guaranteed-pod-resources-no-container-resources": {
			pod: &guaranteedPodResourcesNoContainerResources,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"no-request-limit-1": {lowOOMScoreAdj: -997, highOOMScoreAdj: -997},
				"no-request-limit-2": {lowOOMScoreAdj: -997, highOOMScoreAdj: -997},
			},
			memoryCapacity:                  4000000000,
			podLevelResourcesFeatureEnabled: true,
		},
		"guaranteed-pod-resources-equal-container-resources": {
			pod: &guaranteedPodResourcesEqualContainerRequests,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"guaranteed-container-1": {lowOOMScoreAdj: -997, highOOMScoreAdj: -997},
				"guaranteed-container-2": {lowOOMScoreAdj: -997, highOOMScoreAdj: -997},
			},
			memoryCapacity:                  4000000000,
			podLevelResourcesFeatureEnabled: true,
		},
		"guaranteed-pod-resources-unequal-container-requests": {
			pod: &guaranteedPodResourcesUnequalContainerRequests,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"burstable-container":   {lowOOMScoreAdj: -997, highOOMScoreAdj: -997},
				"best-effort-container": {lowOOMScoreAdj: -997, highOOMScoreAdj: -997},
			},
			memoryCapacity:                  4000000000,
			podLevelResourcesFeatureEnabled: true,
		},
		"burstable-pod-resources-no-container-resources": {
			pod: &burstablePodResourcesNoContainerResources,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"no-request-limit-1": {lowOOMScoreAdj: 750, highOOMScoreAdj: 750},
				"no-request-limit-2": {lowOOMScoreAdj: 750, highOOMScoreAdj: 750},
			},
			memoryCapacity:                  4000000000,
			podLevelResourcesFeatureEnabled: true,
		},
		"burstable-pod-resources-equal-container-requests": {
			pod: &burstablePodResourcesEqualContainerRequests,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"guaranteed-container-1": {lowOOMScoreAdj: 750, highOOMScoreAdj: 750},
				"guaranteed-container-2": {lowOOMScoreAdj: 750, highOOMScoreAdj: 750},
			},
			memoryCapacity:                  4000000000,
			podLevelResourcesFeatureEnabled: true,
		},
		"burstable-pod-resources-unequal-container-requests": {
			pod: &burstablePodResourcesUnequalContainerRequests,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"burstable-container":   {lowOOMScoreAdj: 625, highOOMScoreAdj: 625},
				"best-effort-container": {lowOOMScoreAdj: 875, highOOMScoreAdj: 875},
			},
			memoryCapacity:                  4000000000,
			podLevelResourcesFeatureEnabled: true,
		},
		"burstable-pod-resources-no-container-resources-with-sidecar": {
			pod: &burstablePodResourcesNoContainerResourcesWithSidecar,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"no-request-limit":         {lowOOMScoreAdj: 750, highOOMScoreAdj: 750},
				"no-request-limit-sidecar": {lowOOMScoreAdj: 750, highOOMScoreAdj: 750},
			},
			memoryCapacity:                  4000000000,
			podLevelResourcesFeatureEnabled: true,
		},
		"burstable-pod-resources-equal-container-requests-with-sidecar": {
			pod: &burstablePodResourcesEqualContainerRequestsWithSidecar,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"burstable-container": {lowOOMScoreAdj: 750, highOOMScoreAdj: 750},
				"burstable-sidecar":   {lowOOMScoreAdj: 750, highOOMScoreAdj: 750},
			},
			memoryCapacity:                  4000000000,
			podLevelResourcesFeatureEnabled: true,
		},
		"burstable-pod-resources-unequal-container-requests-with-sidecar": {
			pod: &burstablePodResourcesUnequalContainerRequestsWithSidecar,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"burstable-container-1": {lowOOMScoreAdj: 725, highOOMScoreAdj: 725},
				"burstable-container-2": {lowOOMScoreAdj: 850, highOOMScoreAdj: 850},
				"burstable-sidecar":     {lowOOMScoreAdj: 850, highOOMScoreAdj: 850},
			},
			memoryCapacity:                  4000000000,
			podLevelResourcesFeatureEnabled: true,
		},
		"burstable-pod-with-dra-disabled": {
			pod: makeTestPodForDRA(podParams{
				c1Mem: "500Mi",
				draAlloc: draMemAllocation{
					containers:           []string{"c1"},
					perContainerOverhead: "500Mi",
				},
			}),
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				// OOMScoreAdj = 1000 - (1000 * 500Mi Spec / 4GB) = 869
				"c1": {lowOOMScoreAdj: 869, highOOMScoreAdj: 869},
			},
			enableDRANodeAllocatableResources: false,
		},
		"burstable-pod-with-dra-enabled": {
			pod: makeTestPodForDRA(podParams{
				c1Mem: "500Mi",
				draAlloc: draMemAllocation{
					containers:           []string{"c1"},
					perContainerOverhead: "500Mi",
				},
			}),
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				// OOMScoreAdj = 1000 - (1000 * (500Mi Spec + 500Mi DRA) / 4GB) = 738
				"c1": {lowOOMScoreAdj: 738, highOOMScoreAdj: 738},
			},
			enableDRANodeAllocatableResources: true,
		},
		"burstable-pod-with-shared-dra-claim": {
			pod: makeTestPodForDRA(podParams{
				c1Mem: "500Mi",
				c2Mem: "500Mi",
				draAlloc: draMemAllocation{
					containers:     []string{"c1", "c2"},
					perPodOverhead: "1000Mi",
				},
			}),
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				// OOMScoreAdj = 1000 - (1000 * (500Mi Spec + 1000Mi DRA / 2 references) / 4GB) = 738
				"c1": {lowOOMScoreAdj: 738, highOOMScoreAdj: 738},
				"c2": {lowOOMScoreAdj: 738, highOOMScoreAdj: 738},
			},
			enableDRANodeAllocatableResources: true,
		},
		"burstable-pod-with-plr-and-dra-claim": {
			pod: makeTestPodForDRA(podParams{
				c1Mem:       "1Gi",
				podLevelMem: "5Gi",
				draAlloc: draMemAllocation{
					containers:           []string{"c1"},
					perContainerOverhead: "2Gi",
				},
			}),
			memoryCapacity: 8000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				// PLR Remaining Request = (5Gi Pod - 1Gi c1 Spec - 2Gi c1 DRA) = 2Gi
				// OOMScoreAdj = 1000 - (1000 * (1Gi Spec + 2Gi DRA + 2Gi PLR remaining) / 8GB) = 329
				"c1": {lowOOMScoreAdj: 329, highOOMScoreAdj: 329},
			},
			podLevelResourcesFeatureEnabled:   true,
			enableDRANodeAllocatableResources: true,
		},
		"burstable-pod-with-plr-and-shared-dra-claim": {
			pod: makeTestPodForDRA(podParams{
				c1Mem:       "1Gi",
				c2Mem:       "1Gi",
				podLevelMem: "6Gi",
				draAlloc: draMemAllocation{
					containers:     []string{"c1", "c2"},
					perPodOverhead: "2Gi",
				},
			}),
			memoryCapacity: 8000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				// c1, c2 effective request = 1Gi Spec + 2Gi DRA / 2 refs = 2Gi
				// PLR Remaining Request = (6Gi Pod - 4Gi sum of effective requests) = 2Gi
				// Per-container PLR share = 2Gi / 2 containers = 1Gi
				// OOMScoreAdj = 1000 - (1000 * (2Gi effective + 1Gi PLR share) / 8GB) = 598
				"c1": {lowOOMScoreAdj: 598, highOOMScoreAdj: 598},
				"c2": {lowOOMScoreAdj: 598, highOOMScoreAdj: 598},
			},
			podLevelResourcesFeatureEnabled:   true,
			enableDRANodeAllocatableResources: true,
		},
		"burstable-pod-with-dra-direct-device": {
			pod: makeTestPodForDRA(podParams{
				c1Mem: "500Mi",
				draAlloc: draMemAllocation{
					containers: []string{"c1"},
					mapping:    "500Mi",
				},
			}),
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				// OOMScoreAdj = 1000 - (1000 * (500Mi Spec + 500Mi direct DRA) / 4GB) = 738
				"c1": {lowOOMScoreAdj: 738, highOOMScoreAdj: 738},
			},
			enableDRANodeAllocatableResources: true,
		},
		"burstable-pod-with-combined-dra-direct-and-overhead-device": {
			pod: makeTestPodForDRA(podParams{
				c1Mem: "500Mi",
				draAlloc: draMemAllocation{
					containers:           []string{"c1"},
					mapping:              "500Mi",
					perContainerOverhead: "300Mi",
					perPodOverhead:       "200Mi",
				},
			}),
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				// OOMScoreAdj = 1000 - (1000 * (500Mi Spec + 500Mi mapping + 300Mi container + 200Mi pod) / 4GB) = 607
				"c1": {lowOOMScoreAdj: 607, highOOMScoreAdj: 607},
			},
			enableDRANodeAllocatableResources: true,
		},
		"burstable-pod-with-sidecar-and-dra-enabled": {
			pod: makeTestPodForDRA(podParams{
				c1Mem:      "500Mi",
				sidecarMem: "200Mi",
				draAlloc: draMemAllocation{
					containers:           []string{"c1"},
					perContainerOverhead: "500Mi",
				},
			}),
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				// c1 OOMScoreAdj = 1000 - (1000 * (500Mi Spec + 500Mi DRA) / 4GB) = 738
				"c1": {lowOOMScoreAdj: 738, highOOMScoreAdj: 738},
				// s1 OOMScoreAdj = 1000 - (1000 * 200Mi Spec / 4GB) = 948 capped to min regular container score (738)
				"s1": {lowOOMScoreAdj: 738, highOOMScoreAdj: 738},
			},
			enableDRANodeAllocatableResources: true,
		},
	}
	for name, test := range oomTests {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, test.podLevelResourcesFeatureEnabled)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRANodeAllocatableResources, test.enableDRANodeAllocatableResources)
			listContainers := test.pod.Spec.InitContainers
			listContainers = append(listContainers, test.pod.Spec.Containers...)
			for _, container := range listContainers {
				oomScoreAdj := GetContainerOOMScoreAdjust(test.pod, &container, test.memoryCapacity)
				if oomScoreAdj < test.lowHighOOMScoreAdj[container.Name].lowOOMScoreAdj || oomScoreAdj > test.lowHighOOMScoreAdj[container.Name].highOOMScoreAdj {
					t.Errorf("oom_score_adj %s should be between %d and %d, but was %d", container.Name, test.lowHighOOMScoreAdj[container.Name].lowOOMScoreAdj, test.lowHighOOMScoreAdj[container.Name].highOOMScoreAdj, oomScoreAdj)
				}
			}
		})

	}
}

const (
	tieBreakMi = int64(1) << 20
	tieBreakGi = int64(1) << 30
	tieBreakTi = int64(1) << 40
)

// tieBreakContainer returns a container with the given memory request and
// limit in bytes. A zero limit means unlimited.
func tieBreakContainer(name string, request, limit int64) v1.Container {
	res := v1.ResourceRequirements{
		Requests: v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(request, resource.BinarySI)},
	}
	if limit > 0 {
		res.Limits = v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(limit, resource.BinarySI)}
	}
	return v1.Container{Name: name, Resources: res}
}

func tieBreakPod(containers ...v1.Container) *v1.Pod {
	return &v1.Pod{Spec: v1.PodSpec{Containers: containers}}
}

func TestGetContainerOOMScoreAdjustTieBreak(t *testing.T) {
	sidecar := tieBreakContainer("sidecar", 50*tieBreakMi, 0)
	sidecar.RestartPolicy = &restartPolicyAlways
	sidecarPod := tieBreakPod(
		tieBreakContainer("small-bounded", 50*tieBreakMi, 50*tieBreakMi),
		tieBreakContainer("large-unbounded", 16*tieBreakGi, 0),
	)
	sidecarPod.Spec.InitContainers = []v1.Container{sidecar}

	// Burstable through the container CPU request; memory is bounded only at
	// the pod level. The single container's share of the request is 2Gi.
	podLevelPod := func(limit string) *v1.Pod {
		return &v1.Pod{Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("2Gi")},
				Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse(limit)},
			},
			Containers: []v1.Container{{Name: "c", Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			}}},
		}}
	}

	// A 500Mi DRA memory claim shared by two containers adds 250Mi to each
	// effective request but the full 500Mi to each cgroup limit.
	sharedClaimPod := tieBreakPod(
		tieBreakContainer("c", 500*tieBreakMi, 500*tieBreakMi),
		tieBreakContainer("other", 500*tieBreakMi, 0),
	)
	sharedClaimPod.Status.NodeAllocatableResourceClaimStatuses = []v1.NodeAllocatableResourceClaimStatus{{
		ResourceClaimName: "claim",
		Containers:        []string{"c", "other"},
		Mapping: []v1.NodeAllocatableMappedResources{{
			Name: v1.ResourceMemory, Quantity: resource.NewQuantity(500*tieBreakMi, resource.BinarySI),
		}},
	}}

	tests := []struct {
		name      string
		capacity  int64
		pod       *v1.Pod
		container *v1.Container
		gateOff   int
		gateOn    int
	}{
		// 16Gi: one adj point is ~16Mi, below the 64Mi floor, so no nudge.
		{
			name:     "16Gi node, 500Mi unlimited",
			capacity: 16 * tieBreakGi,
			pod:      tieBreakPod(tieBreakContainer("c", 500*tieBreakMi, 0)),
			gateOff:  970,
			gateOn:   970,
		},
		{
			name:     "16Gi node, 500Mi request equals limit",
			capacity: 16 * tieBreakGi,
			pod:      tieBreakPod(tieBreakContainer("c", 500*tieBreakMi, 500*tieBreakMi)),
			gateOff:  970,
			gateOn:   969,
		},
		// 1Ti: one adj point is ~1Gi, nudge capped at 4 points.
		{
			name:     "1Ti node, 50Mi unlimited is below the floor",
			capacity: tieBreakTi,
			pod:      tieBreakPod(tieBreakContainer("c", 50*tieBreakMi, 0)),
			gateOff:  999,
			gateOn:   999,
		},
		{
			name:     "1Ti node, 500Mi unlimited",
			capacity: tieBreakTi,
			pod:      tieBreakPod(tieBreakContainer("c", 500*tieBreakMi, 0)),
			gateOff:  999,
			gateOn:   998,
		},
		{
			name:     "1Ti node, 4Gi unlimited hits the nudge cap",
			capacity: tieBreakTi,
			pod:      tieBreakPod(tieBreakContainer("c", 4*tieBreakGi, 0)),
			gateOff:  997,
			gateOn:   993,
		},
		{
			name:     "1Ti node, 16Gi unlimited",
			capacity: tieBreakTi,
			pod:      tieBreakPod(tieBreakContainer("c", 16*tieBreakGi, 0)),
			gateOff:  985,
			gateOn:   981,
		},
		{
			name:     "1Ti node, 500Mi request equals limit",
			capacity: tieBreakTi,
			pod:      tieBreakPod(tieBreakContainer("c", 500*tieBreakMi, 500*tieBreakMi)),
			gateOff:  999,
			gateOn:   993,
		},
		{
			// Effective request 750Mi, cgroup limit 1000Mi.
			name:     "1Ti node, shared DRA memory claim loses the bonus",
			capacity: tieBreakTi,
			pod:      sharedClaimPod,
			gateOff:  999,
			gateOn:   997,
		},
		// 4Ti: one adj point is ~4Gi, nudge capped at 6 points.
		{
			name:     "4Ti node, 500Mi unlimited",
			capacity: 4 * tieBreakTi,
			pod:      tieBreakPod(tieBreakContainer("c", 500*tieBreakMi, 0)),
			gateOff:  999,
			gateOn:   998,
		},
		{
			name:     "4Ti node, 64Gi unlimited",
			capacity: 4 * tieBreakTi,
			pod:      tieBreakPod(tieBreakContainer("c", 64*tieBreakGi, 0)),
			gateOff:  985,
			gateOn:   979,
		},
		{
			// Regulars score 995 (50Mi bounded) and 981 (16Gi). Legacy caps
			// against the smallest request (1000 -> 999); the tie-break caps
			// against the highest regular score instead.
			name:      "1Ti node, sidecar capped at the highest regular container",
			capacity:  tieBreakTi,
			pod:       sidecarPod,
			container: &sidecarPod.Spec.InitContainers[0],
			gateOff:   999,
			gateOn:    995,
		},
		{
			name:     "1Ti node, pod-level limit above the container's share",
			capacity: tieBreakTi,
			pod:      podLevelPod("4Gi"),
			gateOff:  999,
			gateOn:   995,
		},
		{
			name:     "1Ti node, pod-level limit equal to the container's share",
			capacity: tieBreakTi,
			pod:      podLevelPod("2Gi"),
			gateOff:  999,
			gateOn:   990,
		},
	}
	for _, gate := range []bool{false, true} {
		for _, tc := range tests {
			t.Run(fmt.Sprintf("tieBreak=%v/%s", gate, tc.name), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletOOMScoreAdjTieBreak, gate)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRANodeAllocatableResources, true)
				container := tc.container
				if container == nil {
					container = &tc.pod.Spec.Containers[0]
				}
				want := tc.gateOff
				if gate {
					want = tc.gateOn
				}
				if got := GetContainerOOMScoreAdjust(tc.pod, container, tc.capacity); got != want {
					t.Errorf("oom_score_adj = %d, want %d", got, want)
				}
				if got := PossibleContainerOOMScoreAdjusts(tc.pod, container, tc.capacity); got[0] != tc.gateOff || got[1] != tc.gateOn {
					t.Errorf("PossibleContainerOOMScoreAdjusts = %v, want [%d %d]", got, tc.gateOff, tc.gateOn)
				}
			})
		}
	}
}

// TestOOMScoreAdjTieBreakKillOrder checks the documented node OOM behavior
// with the tie-break enabled: containers using the most memory relative to
// their request are killed first. The kernel kills the process with the
// highest oom_score_adj + usage*1000/capacity.
func TestOOMScoreAdjTieBreakKillOrder(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletOOMScoreAdjTieBreak, true)

	points := func(c v1.Container, usage, capacity int64) float64 {
		pod := tieBreakPod(c)
		adj := GetContainerOOMScoreAdjust(pod, &pod.Spec.Containers[0], capacity)
		return float64(adj) + float64(usage)*1000/float64(capacity)
	}

	t.Run("over-request container is killed before one at its request", func(t *testing.T) {
		requests := []int64{
			16 * tieBreakMi, 50 * tieBreakMi, 100 * tieBreakMi, 250 * tieBreakMi, 500 * tieBreakMi,
			tieBreakGi, 2 * tieBreakGi, 4 * tieBreakGi, 16 * tieBreakGi, 64 * tieBreakGi, 256 * tieBreakGi,
		}
		for _, capacity := range []int64{16 * tieBreakGi, 256 * tieBreakGi, tieBreakTi, 4 * tieBreakTi} {
			// The tie-break may shift a container by at most maxNudge points,
			// plus one point of integer truncation that legacy also has.
			maxNudge := log2Floor(capacity / 1000 / oomScoreAdjTieBreakFloor)
			slack := (maxNudge + 1) * capacity / 1000
			for _, big := range requests {
				for _, small := range requests {
					if big <= small || big >= capacity {
						continue
					}
					over := points(tieBreakContainer("big", big, 0), big+slack+tieBreakMi, capacity)
					atRequest := points(tieBreakContainer("small", small, 0), small, capacity)
					if over <= atRequest {
						t.Errorf("capacity %dGi: %dMi request %dMi over (%.1f points) not killed before %dMi at request (%.1f points)",
							capacity/tieBreakGi, big/tieBreakMi, (slack+tieBreakMi)/tieBreakMi, over, small/tieBreakMi, atRequest)
					}
				}
			}
		}
	})

	t.Run("cases from #142235 on a 1Ti node", func(t *testing.T) {
		for _, tc := range []struct{ bigReq, bigUsage, smallReq int64 }{
			{16 * tieBreakGi, 48 * tieBreakGi, 500 * tieBreakMi},
			{64 * tieBreakGi, 84 * tieBreakGi, 500 * tieBreakMi},
			{100 * tieBreakGi, 130 * tieBreakGi, 50 * tieBreakMi},
		} {
			over := points(tieBreakContainer("A", tc.bigReq, 0), tc.bigUsage, tieBreakTi)
			atRequest := points(tieBreakContainer("B", tc.smallReq, 0), tc.smallReq, tieBreakTi)
			if over <= atRequest {
				t.Errorf("A (%dGi request, %dGi used) = %.1f points, B (%dMi at request) = %.1f points: B would be killed",
					tc.bigReq/tieBreakGi, tc.bigUsage/tieBreakGi, over, tc.smallReq/tieBreakMi, atRequest)
			}
		}
	})

	t.Run("bounded orchestrator outlasts an executor worker on a 1Ti node", func(t *testing.T) {
		// From #142230: same 500Mi request, the orchestrator cannot exceed
		// it, while an executor worker process uses less than its request.
		orchestrator := points(tieBreakContainer("orchestrator", 500*tieBreakMi, 500*tieBreakMi), 200*tieBreakMi, tieBreakTi)
		worker := points(tieBreakContainer("executor", 500*tieBreakMi, 0), 100*tieBreakMi, tieBreakTi)
		if orchestrator >= worker {
			t.Errorf("orchestrator = %.1f points, executor worker = %.1f points: orchestrator would be killed", orchestrator, worker)
		}
	})
}
