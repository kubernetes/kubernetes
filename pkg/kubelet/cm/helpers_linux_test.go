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

package cm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

// getResourceRequirements returns a ResourceRequirements object
func getResourceRequirements(requests, limits api.ResourceList) api.ResourceRequirements {
	res := api.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

// newContainer creates and returns a new container
// with the specified configuration
func newContainer(name string, requests api.ResourceList, limits api.ResourceList) api.Container {
	return api.Container{
		Name:      name,
		Resources: getResourceRequirements(requests, limits),
	}
}

// newPod creates and returns a new pod with
// the specified pod name and containers
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

// getResourceList returns a ResourceList with the
// specified cpu and memory resource values
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

// getNode returns a Node object with the cpu and memory allocatable
// set to the specified values
func getNode(cpu, memory string) *api.Node {
	cpuCores := resource.MustParse(cpu)
	memoryAllocatable := resource.MustParse(memory)
	allocatable := api.ResourceList{}
	allocatable[api.ResourceCPU] = cpuCores
	allocatable[api.ResourceMemory] = memoryAllocatable
	return &api.Node{
		Status: api.NodeStatus{
			Allocatable: allocatable,
		},
	}
}

func TestGetPodResourceRequests(t *testing.T) {
	testCases := []struct {
		pod      *api.Pod
		expected api.ResourceList
	}{
		{
			pod: newPod("foo", []api.Container{
				newContainer("foo", getResourceList("", ""), getResourceList("", "")),
			}),
			expected: getResourceList("", ""),
		},
		{
			pod: newPod("foo", []api.Container{
				newContainer("foo", getResourceList("100m", ""), getResourceList("", "")),
			}),
			expected: getResourceList("100m", ""),
		},
		{
			pod: newPod("foo", []api.Container{
				newContainer("foo", getResourceList("100m", "100Mi"), getResourceList("", "")),
			}),
			expected: getResourceList("100m", "100Mi"),
		},
		{
			pod: newPod("bar", []api.Container{
				newContainer("foo", getResourceList("100m", "100Mi"), getResourceList("", "")),
				newContainer("bar", getResourceList("200m", "100Mi"), getResourceList("", "")),
				newContainer("foobar", getResourceList("50m", "100Mi"), getResourceList("", "")),
			}),
			expected: getResourceList("350m", "300Mi"),
		},
		{
			pod: newPod("bar", []api.Container{
				newContainer("foo", getResourceList("", ""), getResourceList("", "")),
				newContainer("bar", getResourceList("", "100Mi"), getResourceList("", "")),
				newContainer("foobar", getResourceList("50m", "100Mi"), getResourceList("", "")),
			}),
			expected: getResourceList("50m", "200Mi"),
		},
	}
	as := assert.New(t)
	for idx, tc := range testCases {
		actual := GetPodResourceRequests(tc.pod)
		as.Equal(tc.expected.Cpu().Value(), actual.Cpu().Value(), "expected test case [%d] to return %v; got %v instead", idx, tc.expected, actual)
		as.Equal(tc.expected.Memory().Value(), actual.Memory().Value(), "expected test case [%d] to return %v; got %v instead", idx, tc.expected, actual)
	}
}

func TestGetPodResourceLimits(t *testing.T) {
	nodeInfo := getNode("10", "10Gi")
	testCases := []struct {
		pod      *api.Pod
		expected api.ResourceList
	}{
		{
			pod: newPod("foo", []api.Container{
				newContainer("foo", getResourceList("", ""), getResourceList("", "")),
			}),
			expected: getResourceList("10", "10Gi"),
		},
		{
			pod: newPod("foo", []api.Container{
				newContainer("foo", getResourceList("", ""), getResourceList("100m", "")),
			}),
			expected: getResourceList("100m", "10Gi"),
		},
		{
			pod: newPod("foo", []api.Container{
				newContainer("foo", getResourceList("", ""), getResourceList("", "2Gi")),
			}),
			expected: getResourceList("10", "2Gi"),
		},
		{
			pod: newPod("foo", []api.Container{
				newContainer("foo", getResourceList("", ""), getResourceList("100m", "100Mi")),
			}),
			expected: getResourceList("100m", "100Mi"),
		},
		{
			pod: newPod("bar", []api.Container{
				newContainer("foo", getResourceList("", ""), getResourceList("100m", "100Mi")),
				newContainer("bar", getResourceList("", ""), getResourceList("200m", "100Mi")),
				newContainer("foobar", getResourceList("", ""), getResourceList("50m", "100Mi")),
			}),
			expected: getResourceList("350m", "300Mi"),
		},
		{
			pod: newPod("bar", []api.Container{
				newContainer("foo", getResourceList("", ""), getResourceList("", "100Mi")),
				newContainer("bar", getResourceList("", ""), getResourceList("", "100Mi")),
				newContainer("foobar", getResourceList("", ""), getResourceList("50m", "100Mi")),
			}),
			expected: getResourceList("10", "300Mi"),
		},
		{
			pod: newPod("bar", []api.Container{
				newContainer("foo", getResourceList("", ""), getResourceList("100m", "")),
				newContainer("bar", getResourceList("", ""), getResourceList("200m", "100Mi")),
				newContainer("foobar", getResourceList("", ""), getResourceList("50m", "100Mi")),
			}),
			expected: getResourceList("350m", "10Gi"),
		},
		{
			pod: newPod("bar", []api.Container{
				newContainer("foo", getResourceList("", ""), getResourceList("", "100Mi")),
				newContainer("bar", getResourceList("", ""), getResourceList("200m", "100Mi")),
				newContainer("foobar", getResourceList("", ""), getResourceList("50m", "")),
			}),
			expected: getResourceList("10", "10Gi"),
		},
	}
	as := assert.New(t)
	for idx, tc := range testCases {
		actual := GetPodResourceLimits(tc.pod, nodeInfo)
		as.Equal(tc.expected.Cpu().Value(), actual.Cpu().Value(), "expected test case [%d] to return %v; got %v instead", idx, tc.expected.Memory().Value(), actual.Memory().Value())
		as.Equal(tc.expected.Memory().Value(), actual.Memory().Value(), "expected test case [%d] to return %v; got %v instead", idx, tc.expected.Memory().Value(), actual.Memory().Value())
	}
}
