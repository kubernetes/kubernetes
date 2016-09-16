/*
Copyright 2016 The Kubernetes Authors.

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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

func getResourceList(cpu, memory string) api.ResourceList {
	res := api.ResourceList{}
	if cpu != "" {
		res[api.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[api.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

func getResourceRequirements(requests, limits api.ResourceList) api.ResourceRequirements {
	res := api.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

func newContainer(name string, requests api.ResourceList, limits api.ResourceList) api.Container {
	return api.Container{
		Name:      name,
		Resources: getResourceRequirements(requests, limits),
	}
}

func newPod(name string, containers []api.Container) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
	}
}

func TestGetPodQOS(t *testing.T) {
	testCases := []struct {
		pod      *api.Pod
		expected QOSClass
	}{
		{
			pod: newPod("guaranteed", []api.Container{
				newContainer("guaranteed", getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
			}),
			expected: Guaranteed,
		},
		{
			pod: newPod("guaranteed-guaranteed", []api.Container{
				newContainer("guaranteed", getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
				newContainer("guaranteed", getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
			}),
			expected: Guaranteed,
		},
		{
			pod: newPod("best-effort-best-effort", []api.Container{
				newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
				newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
			}),
			expected: BestEffort,
		},
		{
			pod: newPod("best-effort", []api.Container{
				newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
			}),
			expected: BestEffort,
		},
		{
			pod: newPod("best-effort-burstable", []api.Container{
				newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
				newContainer("burstable", getResourceList("1", ""), getResourceList("2", "")),
			}),
			expected: Burstable,
		},
		{
			pod: newPod("best-effort-guaranteed", []api.Container{
				newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
				newContainer("guaranteed", getResourceList("10m", "100Mi"), getResourceList("10m", "100Mi")),
			}),
			expected: Burstable,
		},
		{
			pod: newPod("burstable-cpu-guaranteed-memory", []api.Container{
				newContainer("burstable", getResourceList("", "100Mi"), getResourceList("", "100Mi")),
			}),
			expected: Burstable,
		},
		{
			pod: newPod("burstable-guaranteed", []api.Container{
				newContainer("burstable", getResourceList("1", "100Mi"), getResourceList("2", "100Mi")),
				newContainer("guaranteed", getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
			}),
			expected: Burstable,
		},
		{
			pod: newPod("burstable", []api.Container{
				newContainer("burstable", getResourceList("10m", "100Mi"), getResourceList("100m", "200Mi")),
			}),
			expected: Burstable,
		},
		{
			pod: newPod("burstable", []api.Container{
				newContainer("burstable", getResourceList("0", "0"), getResourceList("100m", "200Mi")),
			}),
			expected: Burstable,
		},
	}
	for _, testCase := range testCases {
		if actual := GetPodQOS(testCase.pod); testCase.expected != actual {
			t.Errorf("invalid qos pod %s, expected: %s, actual: %s", testCase.pod.Name, testCase.expected, actual)
		}
	}
}
