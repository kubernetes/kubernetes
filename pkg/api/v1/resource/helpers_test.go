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

package resource

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestResourceHelpers(t *testing.T) {
	cpuLimit := resource.MustParse("10")
	memoryLimit := resource.MustParse("10G")
	resourceSpec := v1.ResourceRequirements{
		Limits: v1.ResourceList{
			v1.ResourceCPU:    cpuLimit,
			v1.ResourceMemory: memoryLimit,
		},
	}
	if res := resourceSpec.Limits.Cpu(); res.Cmp(cpuLimit) != 0 {
		t.Errorf("expected cpulimit %v, got %v", cpuLimit, res)
	}
	if res := resourceSpec.Limits.Memory(); res.Cmp(memoryLimit) != 0 {
		t.Errorf("expected memorylimit %v, got %v", memoryLimit, res)
	}
	resourceSpec = v1.ResourceRequirements{
		Limits: v1.ResourceList{
			v1.ResourceMemory: memoryLimit,
		},
	}
	if res := resourceSpec.Limits.Cpu(); res.Value() != 0 {
		t.Errorf("expected cpulimit %v, got %v", 0, res)
	}
	if res := resourceSpec.Limits.Memory(); res.Cmp(memoryLimit) != 0 {
		t.Errorf("expected memorylimit %v, got %v", memoryLimit, res)
	}
}

func TestDefaultResourceHelpers(t *testing.T) {
	resourceList := v1.ResourceList{}
	if resourceList.Cpu().Format != resource.DecimalSI {
		t.Errorf("expected %v, actual %v", resource.DecimalSI, resourceList.Cpu().Format)
	}
	if resourceList.Memory().Format != resource.BinarySI {
		t.Errorf("expected %v, actual %v", resource.BinarySI, resourceList.Memory().Format)
	}
}

func TestGetResourceRequest(t *testing.T) {
	cases := []struct {
		pod           *v1.Pod
		cName         string
		resourceName  v1.ResourceName
		expectedValue int64
	}{
		{
			pod:           getPod("foo", podResources{cpuRequest: "9"}),
			resourceName:  v1.ResourceCPU,
			expectedValue: 9000,
		},
		{
			pod:           getPod("foo", podResources{memoryRequest: "90Mi"}),
			resourceName:  v1.ResourceMemory,
			expectedValue: 94371840,
		},
		{
			cName:         "just-overhead for cpu",
			pod:           getPod("foo", podResources{cpuOverhead: "5", memoryOverhead: "5"}),
			resourceName:  v1.ResourceCPU,
			expectedValue: 0,
		},
		{
			cName:         "just-overhead for memory",
			pod:           getPod("foo", podResources{memoryOverhead: "5"}),
			resourceName:  v1.ResourceMemory,
			expectedValue: 0,
		},
		{
			cName:         "cpu overhead and req",
			pod:           getPod("foo", podResources{cpuRequest: "2", cpuOverhead: "5", memoryOverhead: "5"}),
			resourceName:  v1.ResourceCPU,
			expectedValue: 7000,
		},
		{
			cName:         "mem overhead and req",
			pod:           getPod("foo", podResources{cpuRequest: "2", memoryRequest: "1024", cpuOverhead: "5", memoryOverhead: "5"}),
			resourceName:  v1.ResourceMemory,
			expectedValue: 1029,
		},
	}
	as := assert.New(t)
	for idx, tc := range cases {
		actual := GetResourceRequest(tc.pod, tc.resourceName)
		as.Equal(actual, tc.expectedValue, "expected test case [%d] %v: to return %q; got %q instead", idx, tc.cName, tc.expectedValue, actual)
	}
}

func TestExtractResourceValue(t *testing.T) {
	cases := []struct {
		fs            *v1.ResourceFieldSelector
		pod           *v1.Pod
		cName         string
		expectedValue string
		expectedError error
	}{
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{cpuLimit: "9"}),
			expectedValue: "9",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{}),
			expectedValue: "0",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{cpuRequest: "8"}),
			expectedValue: "8",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{cpuRequest: "100m"}),
			expectedValue: "1",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
				Divisor:  resource.MustParse("100m"),
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{cpuRequest: "1200m"}),
			expectedValue: "12",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{memoryRequest: "100Mi"}),
			expectedValue: "104857600",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
				Divisor:  resource.MustParse("1Mi"),
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{memoryRequest: "100Mi", memoryLimit: "1Gi"}),
			expectedValue: "100",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.memory",
			},
			cName:         "foo",
			pod:           getPod("foo", podResources{memoryRequest: "10Mi", memoryLimit: "100Mi"}),
			expectedValue: "104857600",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.cpu",
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{cpuLimit: "9"}),
			expectedValue: "9",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{}),
			expectedValue: "0",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{cpuRequest: "8"}),
			expectedValue: "8",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{cpuRequest: "100m"}),
			expectedValue: "1",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
				Divisor:  resource.MustParse("100m"),
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{cpuRequest: "1200m"}),
			expectedValue: "12",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{memoryRequest: "100Mi"}),
			expectedValue: "104857600",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
				Divisor:  resource.MustParse("1Mi"),
			},
			cName:         "init-foo",
			pod:           getPod("foo", podResources{memoryRequest: "100Mi", memoryLimit: "1Gi"}),
			expectedValue: "100",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.memory",
			},
			cName: "init-foo",
			pod:   getPod("foo", podResources{memoryRequest: "10Mi", memoryLimit: "100Mi"}),

			expectedValue: "104857600",
		},
	}
	as := assert.New(t)
	for idx, tc := range cases {
		actual, err := ExtractResourceValueByContainerName(tc.fs, tc.pod, tc.cName)
		if tc.expectedError != nil {
			as.Equal(tc.expectedError, err, "expected test case [%d] to fail with error %v; got %v", idx, tc.expectedError, err)
		} else {
			as.Nil(err, "expected test case [%d] to not return an error; got %v", idx, err)
			as.Equal(tc.expectedValue, actual, "expected test case [%d] to return %q; got %q instead", idx, tc.expectedValue, actual)
		}
	}
}

func TestPodRequestsAndLimits(t *testing.T) {
	cases := []struct {
		pod              *v1.Pod
		cName            string
		expectedRequests v1.ResourceList
		expectedLimits   v1.ResourceList
	}{
		{
			cName:            "just-limit-no-overhead",
			pod:              getPod("foo", podResources{cpuLimit: "9"}),
			expectedRequests: v1.ResourceList{},
			expectedLimits: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU): resource.MustParse("9"),
			},
		},
		{
			cName: "just-overhead",
			pod:   getPod("foo", podResources{cpuOverhead: "5", memoryOverhead: "5"}),
			expectedRequests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("5"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("5"),
			},
			expectedLimits: v1.ResourceList{},
		},
		{
			cName: "req-and-overhead",
			pod:   getPod("foo", podResources{cpuRequest: "1", memoryRequest: "10", cpuOverhead: "5", memoryOverhead: "5"}),
			expectedRequests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("6"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("15"),
			},
			expectedLimits: v1.ResourceList{},
		},
		{
			cName: "all-req-lim-and-overhead",
			pod:   getPod("foo", podResources{cpuRequest: "1", cpuLimit: "2", memoryRequest: "10", memoryLimit: "12", cpuOverhead: "5", memoryOverhead: "5"}),
			expectedRequests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("6"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("15"),
			},
			expectedLimits: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("7"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("17"),
			},
		},
		{
			cName: "req-some-lim-and-overhead",
			pod:   getPod("foo", podResources{cpuRequest: "1", cpuLimit: "2", memoryRequest: "10", cpuOverhead: "5", memoryOverhead: "5"}),
			expectedRequests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("6"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("15"),
			},
			expectedLimits: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU): resource.MustParse("7"),
			},
		},
	}
	for idx, tc := range cases {
		resRequests, resLimits := PodRequestsAndLimits(tc.pod)

		if !equality.Semantic.DeepEqual(tc.expectedRequests, resRequests) {
			t.Errorf("test case failure[%d]: %v, requests:\n expected:\t%v\ngot\t\t%v", idx, tc.cName, tc.expectedRequests, resRequests)
		}

		if !equality.Semantic.DeepEqual(tc.expectedLimits, resLimits) {
			t.Errorf("test case failure[%d]: %v, limits:\n expected:\t%v\ngot\t\t%v", idx, tc.cName, tc.expectedLimits, resLimits)
		}
	}
}

type podResources struct {
	cpuRequest, cpuLimit, memoryRequest, memoryLimit, cpuOverhead, memoryOverhead string
}

func getPod(cname string, resources podResources) *v1.Pod {
	r := v1.ResourceRequirements{
		Limits:   make(v1.ResourceList),
		Requests: make(v1.ResourceList),
	}

	overhead := make(v1.ResourceList)

	if resources.cpuLimit != "" {
		r.Limits[v1.ResourceCPU] = resource.MustParse(resources.cpuLimit)
	}
	if resources.memoryLimit != "" {
		r.Limits[v1.ResourceMemory] = resource.MustParse(resources.memoryLimit)
	}
	if resources.cpuRequest != "" {
		r.Requests[v1.ResourceCPU] = resource.MustParse(resources.cpuRequest)
	}
	if resources.memoryRequest != "" {
		r.Requests[v1.ResourceMemory] = resource.MustParse(resources.memoryRequest)
	}
	if resources.cpuOverhead != "" {
		overhead[v1.ResourceCPU] = resource.MustParse(resources.cpuOverhead)
	}
	if resources.memoryOverhead != "" {
		overhead[v1.ResourceMemory] = resource.MustParse(resources.memoryOverhead)
	}

	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      cname,
					Resources: r,
				},
			},
			InitContainers: []v1.Container{
				{
					Name:      "init-" + cname,
					Resources: r,
				},
			},
			Overhead: overhead,
		},
	}
}
