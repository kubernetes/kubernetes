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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	standardMemoryAmount = 8000000000
)

var (
	cpuLimit = api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Resources: api.ResourceRequirements{
						Limits: api.ResourceList{
							api.ResourceName(api.ResourceCPU): resource.MustParse("10"),
						},
					},
				},
			},
		},
	}

	memoryLimitCPURequest = api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceCPU): resource.MustParse("0"),
						},
						Limits: api.ResourceList{
							api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
						},
					},
				},
			},
		},
	}

	zeroMemoryLimit = api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Resources: api.ResourceRequirements{
						Limits: api.ResourceList{
							api.ResourceName(api.ResourceMemory): resource.MustParse("0"),
						},
					},
				},
			},
		},
	}

	noRequestLimit = api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Resources: api.ResourceRequirements{},
				},
			},
		},
	}

	equalRequestLimitCPUMemory = api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
							api.ResourceName(api.ResourceCPU):    resource.MustParse("5m"),
						},
						Limits: api.ResourceList{
							api.ResourceName(api.ResourceCPU):    resource.MustParse("5m"),
							api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
						},
					},
				},
			},
		},
	}

	cpuUnlimitedMemoryLimitedWithRequests = api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceMemory): resource.MustParse(strconv.Itoa(standardMemoryAmount / 2)),
							api.ResourceName(api.ResourceCPU):    resource.MustParse("5m"),
						},
						Limits: api.ResourceList{
							api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
						},
					},
				},
			},
		},
	}

	requestNoLimit = api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceMemory): resource.MustParse(strconv.Itoa(standardMemoryAmount - 1)),
							api.ResourceName(api.ResourceCPU):    resource.MustParse("5m"),
						},
					},
				},
			},
		},
	}
	criticalPodWithNoLimit = api.Pod{
		ObjectMeta: api.ObjectMeta{
			Annotations: map[string]string{
				kubetypes.CriticalPodAnnotationKey: "",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceMemory): resource.MustParse(strconv.Itoa(standardMemoryAmount - 1)),
							api.ResourceName(api.ResourceCPU):    resource.MustParse("5m"),
						},
					},
				},
			},
		},
	}
)

type oomTest struct {
	pod             *api.Pod
	memoryCapacity  int64
	lowOOMScoreAdj  int // The max oom_score_adj score the container should be assigned.
	highOOMScoreAdj int // The min oom_score_adj score the container should be assigned.
}

func TestGetContainerOOMScoreAdjust(t *testing.T) {
	oomTests := []oomTest{
		{
			pod:             &cpuLimit,
			memoryCapacity:  4000000000,
			lowOOMScoreAdj:  999,
			highOOMScoreAdj: 999,
		},
		{
			pod:             &memoryLimitCPURequest,
			memoryCapacity:  8000000000,
			lowOOMScoreAdj:  999,
			highOOMScoreAdj: 999,
		},
		{
			pod:             &zeroMemoryLimit,
			memoryCapacity:  7230457451,
			lowOOMScoreAdj:  1000,
			highOOMScoreAdj: 1000,
		},
		{
			pod:             &noRequestLimit,
			memoryCapacity:  4000000000,
			lowOOMScoreAdj:  1000,
			highOOMScoreAdj: 1000,
		},
		{
			pod:             &equalRequestLimitCPUMemory,
			memoryCapacity:  123456789,
			lowOOMScoreAdj:  -998,
			highOOMScoreAdj: -998,
		},
		{
			pod:             &cpuUnlimitedMemoryLimitedWithRequests,
			memoryCapacity:  standardMemoryAmount,
			lowOOMScoreAdj:  495,
			highOOMScoreAdj: 505,
		},
		{
			pod:             &requestNoLimit,
			memoryCapacity:  standardMemoryAmount,
			lowOOMScoreAdj:  2,
			highOOMScoreAdj: 2,
		},
		{
			pod:             &criticalPodWithNoLimit,
			memoryCapacity:  standardMemoryAmount,
			lowOOMScoreAdj:  -998,
			highOOMScoreAdj: -998,
		},
	}
	for _, test := range oomTests {
		oomScoreAdj := GetContainerOOMScoreAdjust(test.pod, &test.pod.Spec.Containers[0], test.memoryCapacity)
		if oomScoreAdj < test.lowOOMScoreAdj || oomScoreAdj > test.highOOMScoreAdj {
			t.Errorf("oom_score_adj should be between %d and %d, but was %d", test.lowOOMScoreAdj, test.highOOMScoreAdj, oomScoreAdj)
		}
	}
}
