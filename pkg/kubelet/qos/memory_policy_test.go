/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
)

const (
	standardMemoryAmount = 8000000000
)

var (
	zeroRequestMemoryBestEffort = api.Container{
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceCPU):    resource.MustParse("5m"),
				api.ResourceName(api.ResourceMemory): resource.MustParse("0G"),
			},
			Limits: api.ResourceList{
				api.ResourceName(api.ResourceCPU):    resource.MustParse("5m"),
				api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
			},
		},
	}

	edgeMemoryBestEffort = api.Container{
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse("0G"),
			},
			Limits: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse("0G"),
			},
		},
	}

	noRequestMemoryBestEffort = api.Container{
		Resources: api.ResourceRequirements{
			Limits: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
			},
		},
	}

	noLimitMemoryBestEffort = api.Container{}

	memoryGuaranteed = api.Container{
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
			},
			Limits: api.ResourceList{
				api.ResourceName(api.ResourceCPU):    resource.MustParse("5m"),
				api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
			},
		},
	}

	memoryBurstable = api.Container{
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse(strconv.Itoa(standardMemoryAmount / 2)),
			},
			Limits: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
			},
		},
	}

	memoryBurstableNoLimit = api.Container{
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse(strconv.Itoa(standardMemoryAmount - 1)),
			},
		},
	}
)

func TestIsMemoryBestEffort(t *testing.T) {
	validCases := []api.Container{zeroRequestMemoryBestEffort, noRequestMemoryBestEffort, noLimitMemoryBestEffort, edgeMemoryBestEffort}
	for _, container := range validCases {
		if !isMemoryBestEffort(&container) {
			t.Errorf("container %+v is memory best-effort", container)
		}
	}
	invalidCases := []api.Container{memoryGuaranteed, memoryBurstable}
	for _, container := range invalidCases {
		if isMemoryBestEffort(&container) {
			t.Errorf("container %+v is not memory best-effort", container)
		}
	}
}

func TestIsMemoryGuaranteed(t *testing.T) {
	validCases := []api.Container{memoryGuaranteed}
	for _, container := range validCases {
		if !isMemoryGuaranteed(&container) {
			t.Errorf("container %+v is memory guaranteed", container)
		}
	}
	invalidCases := []api.Container{zeroRequestMemoryBestEffort, noRequestMemoryBestEffort, noLimitMemoryBestEffort, edgeMemoryBestEffort, memoryBurstable}
	for _, container := range invalidCases {
		if isMemoryGuaranteed(&container) {
			t.Errorf("container %+v is not memory guaranteed", container)
		}
	}
}

type oomTest struct {
	container       *api.Container
	memoryCapacity  int64
	lowOOMScoreAdj  int // The max oom_score_adj score the container should be assigned.
	highOOMScoreAdj int // The min oom_score_adj score the container should be assigned.
}

func TestGetContainerOOMScoreAdjust(t *testing.T) {

	oomTests := []oomTest{
		{
			container:       &zeroRequestMemoryBestEffort,
			memoryCapacity:  4000000000,
			lowOOMScoreAdj:  1000,
			highOOMScoreAdj: 1000,
		},
		{
			container:       &edgeMemoryBestEffort,
			memoryCapacity:  8000000000,
			lowOOMScoreAdj:  1000,
			highOOMScoreAdj: 1000,
		},
		{
			container:       &noRequestMemoryBestEffort,
			memoryCapacity:  7230457451,
			lowOOMScoreAdj:  1000,
			highOOMScoreAdj: 1000,
		},
		{
			container:       &noLimitMemoryBestEffort,
			memoryCapacity:  4000000000,
			lowOOMScoreAdj:  1000,
			highOOMScoreAdj: 1000,
		},
		{
			container:       &memoryGuaranteed,
			memoryCapacity:  123456789,
			lowOOMScoreAdj:  -999,
			highOOMScoreAdj: -999,
		},
		{
			container:       &memoryBurstable,
			memoryCapacity:  standardMemoryAmount,
			lowOOMScoreAdj:  495,
			highOOMScoreAdj: 505,
		},
		{
			container:       &memoryBurstableNoLimit,
			memoryCapacity:  standardMemoryAmount,
			lowOOMScoreAdj:  2,
			highOOMScoreAdj: 2,
		},
	}
	for _, test := range oomTests {
		oomScoreAdj := GetContainerOOMScoreAdjust(test.container, test.memoryCapacity)
		if oomScoreAdj < test.lowOOMScoreAdj || oomScoreAdj > test.highOOMScoreAdj {
			t.Errorf("oom_score_adj should be between %d and %d, but was %d", test.lowOOMScoreAdj, test.highOOMScoreAdj, oomScoreAdj)
		}
	}
}
