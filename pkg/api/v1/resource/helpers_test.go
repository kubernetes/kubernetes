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
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestResourceHelpers(t *testing.T) {
	cpuLimit := resource.MustParse("10")
	memoryLimit := resource.MustParse("10G")
	resourceSpec := v1.ResourceRequirements{
		Limits: v1.ResourceList{
			"cpu":             cpuLimit,
			"memory":          memoryLimit,
			"kube.io/storage": memoryLimit,
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
			"memory":          memoryLimit,
			"kube.io/storage": memoryLimit,
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
			pod:           getPod("foo", "", "9", "", ""),
			expectedValue: "9",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", "", "", "", ""),
			expectedValue: "0",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", "8", "", "", ""),
			expectedValue: "8",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
			},
			cName:         "foo",
			pod:           getPod("foo", "100m", "", "", ""),
			expectedValue: "1",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.cpu",
				Divisor:  resource.MustParse("100m"),
			},
			cName:         "foo",
			pod:           getPod("foo", "1200m", "", "", ""),
			expectedValue: "12",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
			},
			cName:         "foo",
			pod:           getPod("foo", "", "", "100Mi", ""),
			expectedValue: "104857600",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "requests.memory",
				Divisor:  resource.MustParse("1Mi"),
			},
			cName:         "foo",
			pod:           getPod("foo", "", "", "100Mi", "1Gi"),
			expectedValue: "100",
		},
		{
			fs: &v1.ResourceFieldSelector{
				Resource: "limits.memory",
			},
			cName:         "foo",
			pod:           getPod("foo", "", "", "10Mi", "100Mi"),
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

func getPod(cname, cpuRequest, cpuLimit, memoryRequest, memoryLimit string) *v1.Pod {
	resources := v1.ResourceRequirements{
		Limits:   make(v1.ResourceList),
		Requests: make(v1.ResourceList),
	}
	if cpuLimit != "" {
		resources.Limits[v1.ResourceCPU] = resource.MustParse(cpuLimit)
	}
	if memoryLimit != "" {
		resources.Limits[v1.ResourceMemory] = resource.MustParse(memoryLimit)
	}
	if cpuRequest != "" {
		resources.Requests[v1.ResourceCPU] = resource.MustParse(cpuRequest)
	}
	if memoryRequest != "" {
		resources.Requests[v1.ResourceMemory] = resource.MustParse(memoryRequest)
	}
	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      cname,
					Resources: resources,
				},
			},
		},
	}
}
