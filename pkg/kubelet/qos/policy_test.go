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
	sampleDefaultMemRequest = resource.MustParse(strconv.FormatInt(standardMemoryAmount/8, 10))
	sampleDefaultMemLimit   = resource.MustParse(strconv.FormatInt(1000+(standardMemoryAmount/8), 10))

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

	burstableMixedMultiContainerSmallSideCarePod = v1.Pod{
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
)

type lowHighOOMScoreAdjTest struct {
	lowOOMScoreAdj  int
	highOOMScoreAdj int
}
type oomTest struct {
	pod                             *v1.Pod
	memoryCapacity                  int64
	lowHighOOMScoreAdj              map[string]lowHighOOMScoreAdjTest // [container-name] : min and max oom_score_adj score the container should be assigned.
	sidecarContainersFeatureEnabled bool
}

func TestGetContainerOOMScoreAdjust(t *testing.T) {
	oomTests := []oomTest{
		{
			pod:            &cpuLimit,
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"cpu-limit": {lowOOMScoreAdj: 999, highOOMScoreAdj: 999},
			},
		},
		{
			pod:            &memoryLimitCPURequest,
			memoryCapacity: 8000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"memory-limit-cpu-request": {lowOOMScoreAdj: 999, highOOMScoreAdj: 999},
			},
		},
		{
			pod:            &zeroMemoryLimit,
			memoryCapacity: 7230457451,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"zero-memory-limit": {lowOOMScoreAdj: 1000, highOOMScoreAdj: 1000},
			},
		},
		{
			pod:            &noRequestLimit,
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"no-request-limit": {lowOOMScoreAdj: 1000, highOOMScoreAdj: 1000},
			},
		},
		{
			pod:            &equalRequestLimitCPUMemory,
			memoryCapacity: 123456789,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"equal-request-limit-cpu-memory": {lowOOMScoreAdj: -997, highOOMScoreAdj: -997},
			},
		},
		{
			pod:            &cpuUnlimitedMemoryLimitedWithRequests,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"cpu-unlimited-memory-limited-with-requests": {lowOOMScoreAdj: 495, highOOMScoreAdj: 505},
			},
		},
		{
			pod:            &requestNoLimit,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"request-no-limit": {lowOOMScoreAdj: 3, highOOMScoreAdj: 3},
			},
		},
		{
			pod:            &clusterCritical,
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"cluster-critical": {lowOOMScoreAdj: 1000, highOOMScoreAdj: 1000},
			},
		},
		{
			pod:            &nodeCritical,
			memoryCapacity: 4000000000,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"node-critical": {lowOOMScoreAdj: -997, highOOMScoreAdj: -997},
			},
		},
		{
			pod:            &burstableUniqueContainerPod,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"burstable-unique-container": {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
			},
			sidecarContainersFeatureEnabled: true,
		},
		{
			pod:            &burstableMixedUniqueMainContainerPod,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"init-container": {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
				"main-1":         {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
			},
			sidecarContainersFeatureEnabled: true,
		},
		{
			pod:            &burstableMixedMultiContainerSmallSideCarePod,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"init-container":          {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
				"sidecar-small-container": {lowOOMScoreAdj: 951, highOOMScoreAdj: 951}, // first to kill
				"main-1":                  {lowOOMScoreAdj: 875, highOOMScoreAdj: 875},
			},
			sidecarContainersFeatureEnabled: true,
		},
		{
			pod:            &burstableMixedMultiContainerSameRequestPod,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"init-container":    {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
				"sidecar-container": {lowOOMScoreAdj: 876, highOOMScoreAdj: 876}, // first to kill
				"main-1":            {lowOOMScoreAdj: 875, highOOMScoreAdj: 875},
			},
			sidecarContainersFeatureEnabled: true,
		},
		{
			pod:            &burstableMixedMultiContainerBigSidecarContainerPod,
			memoryCapacity: standardMemoryAmount,
			lowHighOOMScoreAdj: map[string]lowHighOOMScoreAdjTest{
				"init-container":        {lowOOMScoreAdj: 875, highOOMScoreAdj: 880},
				"sidecar-big-container": {lowOOMScoreAdj: 876, highOOMScoreAdj: 876}, // first to kill
				"main-1":                {lowOOMScoreAdj: 875, highOOMScoreAdj: 875},
			},
			sidecarContainersFeatureEnabled: true,
		},
	}
	for _, test := range oomTests {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SidecarContainers, test.sidecarContainersFeatureEnabled)
		listContainers := test.pod.Spec.InitContainers
		listContainers = append(listContainers, test.pod.Spec.Containers...)
		for _, container := range listContainers {
			oomScoreAdj := GetContainerOOMScoreAdjust(test.pod, &container, test.memoryCapacity)
			if oomScoreAdj < test.lowHighOOMScoreAdj[container.Name].lowOOMScoreAdj || oomScoreAdj > test.lowHighOOMScoreAdj[container.Name].highOOMScoreAdj {
				t.Errorf("oom_score_adj %s should be between %d and %d, but was %d", container.Name, test.lowHighOOMScoreAdj[container.Name].lowOOMScoreAdj, test.lowHighOOMScoreAdj[container.Name].highOOMScoreAdj, oomScoreAdj)
			}
		}
	}
}
