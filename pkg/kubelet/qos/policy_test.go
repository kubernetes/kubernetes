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
	zeroRequestBestEffort = api.Container{
		Resources: api.ResourceRequirements{
			Limits: api.ResourceList{
				api.ResourceName(api.ResourceCPU): resource.MustParse("10"),
			},
		},
	}

	edgeBestEffort = api.Container{
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceCPU): resource.MustParse("0"),
			},
			Limits: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
			},
		},
	}

	noRequestBestEffort = api.Container{
		Resources: api.ResourceRequirements{
			Limits: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse("0"),
			},
		},
	}

	noLimitBestEffort = api.Container{}

	guaranteed = api.Container{
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
	}

	burstable = api.Container{
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse(strconv.Itoa(standardMemoryAmount / 2)),
				api.ResourceName(api.ResourceCPU):    resource.MustParse("5m"),
			},
			Limits: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
			},
		},
	}

	burstableNoLimit = api.Container{
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceMemory): resource.MustParse(strconv.Itoa(standardMemoryAmount - 1)),
				api.ResourceName(api.ResourceCPU):    resource.MustParse("5m"),
			},
		},
	}
)

func TestIsBestEffort(t *testing.T) {
	validCases := []api.Container{zeroRequestBestEffort, noRequestBestEffort, noLimitBestEffort, edgeBestEffort}
	for _, container := range validCases {
		if !isBestEffort(&container) {
			t.Errorf("container %+v is best-effort", container)
		}
	}
	invalidCases := []api.Container{guaranteed, burstable}
	for _, container := range invalidCases {
		if isBestEffort(&container) {
			t.Errorf("container %+v is not best-effort", container)
		}
	}
}

func TestIsGuaranteed(t *testing.T) {
	validCases := []api.Container{guaranteed}
	for _, container := range validCases {
		if !isGuaranteed(&container) {
			t.Errorf("container %+v is guaranteed", container)
		}
	}
	invalidCases := []api.Container{zeroRequestBestEffort, noRequestBestEffort, noLimitBestEffort, edgeBestEffort, burstable}
	for _, container := range invalidCases {
		if isGuaranteed(&container) {
			t.Errorf("container %+v is not guaranteed", container)
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
			container:       &zeroRequestBestEffort,
			memoryCapacity:  4000000000,
			lowOOMScoreAdj:  1000,
			highOOMScoreAdj: 1000,
		},
		{
			container:       &edgeBestEffort,
			memoryCapacity:  8000000000,
			lowOOMScoreAdj:  1000,
			highOOMScoreAdj: 1000,
		},
		{
			container:       &noRequestBestEffort,
			memoryCapacity:  7230457451,
			lowOOMScoreAdj:  1000,
			highOOMScoreAdj: 1000,
		},
		{
			container:       &noLimitBestEffort,
			memoryCapacity:  4000000000,
			lowOOMScoreAdj:  1000,
			highOOMScoreAdj: 1000,
		},
		{
			container:       &guaranteed,
			memoryCapacity:  123456789,
			lowOOMScoreAdj:  -999,
			highOOMScoreAdj: -999,
		},
		{
			container:       &burstable,
			memoryCapacity:  standardMemoryAmount,
			lowOOMScoreAdj:  495,
			highOOMScoreAdj: 505,
		},
		{
			container:       &burstableNoLimit,
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
