// +build linux

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
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
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
	minShares := int64(MinShares)
	burstableShares := MilliCPUToShares(100)
	memoryQuantity := resource.MustParse("200Mi")
	burstableMemory := memoryQuantity.Value()
	burstablePartialShares := MilliCPUToShares(200)
	burstableQuota, burstablePeriod := MilliCPUToQuota(200)
	guaranteedShares := MilliCPUToShares(100)
	guaranteedQuota, guaranteedPeriod := MilliCPUToQuota(100)
	memoryQuantity = resource.MustParse("100Mi")
	guaranteedMemory := memoryQuantity.Value()
	testCases := map[string]struct {
		pod      *v1.Pod
		expected *ResourceConfig
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
			expected: &ResourceConfig{CpuShares: &minShares},
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
			expected: &ResourceConfig{CpuShares: &burstableShares},
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
			expected: &ResourceConfig{CpuShares: &burstableShares, CpuQuota: &burstableQuota, CpuPeriod: &burstablePeriod, Memory: &burstableMemory},
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
			expected: &ResourceConfig{CpuShares: &burstablePartialShares},
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
			expected: &ResourceConfig{CpuShares: &guaranteedShares, CpuQuota: &guaranteedQuota, CpuPeriod: &guaranteedPeriod, Memory: &guaranteedMemory},
		},
	}
	for testName, testCase := range testCases {
		actual := ResourceConfigForPod(testCase.pod)
		if !reflect.DeepEqual(actual.CpuPeriod, testCase.expected.CpuPeriod) {
			t.Errorf("unexpected result, test: %v, cpu period not as expected", testName)
		}
		if !reflect.DeepEqual(actual.CpuQuota, testCase.expected.CpuQuota) {
			t.Errorf("unexpected result, test: %v, cpu quota not as expected", testName)
		}
		if !reflect.DeepEqual(actual.CpuShares, testCase.expected.CpuShares) {
			t.Errorf("unexpected result, test: %v, cpu shares not as expected", testName)
		}
		if !reflect.DeepEqual(actual.Memory, testCase.expected.Memory) {
			t.Errorf("unexpected result, test: %v, memory not as expected", testName)
		}
	}
}

func TestMilliCPUToQuota(t *testing.T) {
	testCases := []struct {
		input  int64
		quota  int64
		period int64
	}{
		{
			input:  int64(0),
			quota:  int64(0),
			period: int64(0),
		},
		{
			input:  int64(5),
			quota:  int64(1000),
			period: int64(100000),
		},
		{
			input:  int64(9),
			quota:  int64(1000),
			period: int64(100000),
		},
		{
			input:  int64(10),
			quota:  int64(1000),
			period: int64(100000),
		},
		{
			input:  int64(200),
			quota:  int64(20000),
			period: int64(100000),
		},
		{
			input:  int64(500),
			quota:  int64(50000),
			period: int64(100000),
		},
		{
			input:  int64(1000),
			quota:  int64(100000),
			period: int64(100000),
		},
		{
			input:  int64(1500),
			quota:  int64(150000),
			period: int64(100000),
		},
	}
	for _, testCase := range testCases {
		quota, period := MilliCPUToQuota(testCase.input)
		if quota != testCase.quota || period != testCase.period {
			t.Errorf("Input %v, expected quota %v period %v, but got quota %v period %v", testCase.input, testCase.quota, testCase.period, quota, period)
		}
	}
}
