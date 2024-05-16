//go:build linux
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
	"strconv"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
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
			expected:         &ResourceConfig{CPUShares: &burstableShares},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &burstableQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &cpuNoLimit, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory},
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
			expected:         &ResourceConfig{CPUShares: &burstablePartialShares},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &burstableQuota, CPUPeriod: &tunedQuotaPeriod, Memory: &burstableMemory},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &cpuNoLimit, CPUPeriod: &tunedQuotaPeriod, Memory: &burstableMemory},
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
			expected:         &ResourceConfig{CPUShares: &burstablePartialShares},
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
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &guaranteedQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &guaranteedMemory},
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
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &cpuNoLimit, CPUPeriod: &defaultQuotaPeriod, Memory: &guaranteedMemory},
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
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &guaranteedTunedQuota, CPUPeriod: &tunedQuotaPeriod, Memory: &guaranteedMemory},
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
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &cpuNoLimit, CPUPeriod: &tunedQuotaPeriod, Memory: &guaranteedMemory},
		},
		"burstable-partial-limits-with-init-containers": {
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
	}

	for testName, testCase := range testCases {

		actual := ResourceConfigForPod(testCase.pod, testCase.enforceCPULimits, testCase.quotaPeriod, false)

		if !reflect.DeepEqual(actual.CPUPeriod, testCase.expected.CPUPeriod) {
			t.Errorf("unexpected result, test: %v, cpu period not as expected. Expected: %v, Actual:%v", testName, *testCase.expected.CPUPeriod, *actual.CPUPeriod)
		}
		if !reflect.DeepEqual(actual.CPUQuota, testCase.expected.CPUQuota) {
			t.Errorf("unexpected result, test: %v, cpu quota not as expected. Expected: %v, Actual:%v", testName, *testCase.expected.CPUQuota, *actual.CPUQuota)
		}
		if !reflect.DeepEqual(actual.CPUShares, testCase.expected.CPUShares) {
			t.Errorf("unexpected result, test: %v, cpu shares not as expected. Expected: %v, Actual:%v", testName, *testCase.expected.CPUShares, &actual.CPUShares)
		}
		if !reflect.DeepEqual(actual.Memory, testCase.expected.Memory) {
			t.Errorf("unexpected result, test: %v, memory not as expected. Expected: %v, Actual:%v", testName, *testCase.expected.Memory, *actual.Memory)
		}
	}
}

func TestResourceConfigForPodWithCustomCPUCFSQuotaPeriod(t *testing.T) {
	defaultQuotaPeriod := uint64(100 * time.Millisecond / time.Microsecond) // in microseconds
	tunedQuotaPeriod := uint64(5 * time.Millisecond / time.Microsecond)     // in microseconds
	tunedQuota := int64(1 * time.Millisecond / time.Microsecond)

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.CPUCFSQuotaPeriod, true)

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
			expected:         &ResourceConfig{CPUShares: &burstableShares},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &burstableQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &cpuNoLimit, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory},
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
			expected:         &ResourceConfig{CPUShares: &burstablePartialShares},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &tunedQuota, CPUPeriod: &tunedQuotaPeriod, Memory: &burstableMemory},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &cpuNoLimit, CPUPeriod: &tunedQuotaPeriod, Memory: &burstableMemory},
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
			expected:         &ResourceConfig{CPUShares: &burstablePartialShares},
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
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &guaranteedQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &guaranteedMemory},
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
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &cpuNoLimit, CPUPeriod: &defaultQuotaPeriod, Memory: &guaranteedMemory},
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
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &guaranteedTunedQuota, CPUPeriod: &tunedQuotaPeriod, Memory: &guaranteedMemory},
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
			expected:         &ResourceConfig{CPUShares: &guaranteedShares, CPUQuota: &cpuNoLimit, CPUPeriod: &tunedQuotaPeriod, Memory: &guaranteedMemory},
		},
	}

	for testName, testCase := range testCases {

		actual := ResourceConfigForPod(testCase.pod, testCase.enforceCPULimits, testCase.quotaPeriod, false)

		if !reflect.DeepEqual(actual.CPUPeriod, testCase.expected.CPUPeriod) {
			t.Errorf("unexpected result, test: %v, cpu period not as expected", testName)
		}
		if !reflect.DeepEqual(actual.CPUQuota, testCase.expected.CPUQuota) {
			t.Errorf("unexpected result, test: %v, cpu quota not as expected", testName)
		}
		if !reflect.DeepEqual(actual.CPUShares, testCase.expected.CPUShares) {
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
			resultValuePod := ResourceConfigForPod(&p, false, 0, false)
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, Unified: map[string]string{"memory.min": "104857600"}},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &burstableQuota, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory, Unified: map[string]string{"memory.min": "104857600"}},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &cpuNoLimit, CPUPeriod: &defaultQuotaPeriod, Memory: &burstableMemory, Unified: map[string]string{"memory.min": "104857600"}},
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
			expected:         &ResourceConfig{CPUShares: &burstablePartialShares, Unified: map[string]string{"memory.min": "209715200"}},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &burstableQuota, CPUPeriod: &tunedQuotaPeriod, Memory: &burstableMemory, Unified: map[string]string{"memory.min": "104857600"}},
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
			expected:         &ResourceConfig{CPUShares: &burstableShares, CPUQuota: &cpuNoLimit, CPUPeriod: &tunedQuotaPeriod, Memory: &burstableMemory, Unified: map[string]string{"memory.min": "104857600"}},
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
			expected:         &ResourceConfig{CPUShares: &burstablePartialShares, Unified: map[string]string{"memory.min": "209715200"}},
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

		actual := ResourceConfigForPod(testCase.pod, testCase.enforceCPULimits, testCase.quotaPeriod, true)

		if !reflect.DeepEqual(actual.Unified, testCase.expected.Unified) {
			t.Errorf("unexpected result, test: %v, unified not as expected", testName)
		}
	}
}
