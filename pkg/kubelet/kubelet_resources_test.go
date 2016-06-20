/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"testing"

	"github.com/stretchr/testify/assert"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

func TestPodResourceLimitsDefaulting(t *testing.T) {
	cpuCores := resource.MustParse("10")
	memoryCapacity := resource.MustParse("10Gi")
	tk := newTestKubelet(t, true)
	tk.fakeCadvisor.On("VersionInfo").Return(&cadvisorapi.VersionInfo{}, nil)
	tk.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{
		NumCores:       int(cpuCores.Value()),
		MemoryCapacity: uint64(memoryCapacity.Value()),
	}, nil)
	tk.fakeCadvisor.On("ImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	tk.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	cases := []struct {
		pod      *api.Pod
		expected *api.Pod
	}{
		{
			pod:      getPod("0", "0"),
			expected: getPod("10", "10Gi"),
		},
		{
			pod:      getPod("1", "0"),
			expected: getPod("1", "10Gi"),
		},
		{
			pod:      getPod("", ""),
			expected: getPod("10", "10Gi"),
		},
		{
			pod:      getPod("0", "1Mi"),
			expected: getPod("10", "1Mi"),
		},
	}
	as := assert.New(t)
	for idx, tc := range cases {
		actual, _, err := tk.kubelet.defaultPodLimitsForDownwardApi(tc.pod, nil)
		as.Nil(err, "failed to default pod limits: %v", err)
		if !api.Semantic.DeepEqual(tc.expected, actual) {
			as.Fail("test case [%d] failed.  Expected: %+v, Got: %+v", idx, tc.expected, actual)
		}
	}
}

func getPod(cpuLimit, memoryLimit string) *api.Pod {
	resources := api.ResourceRequirements{}
	if cpuLimit != "" || memoryLimit != "" {
		resources.Limits = make(api.ResourceList)
	}
	if cpuLimit != "" {
		resources.Limits[api.ResourceCPU] = resource.MustParse(cpuLimit)
	}
	if memoryLimit != "" {
		resources.Limits[api.ResourceMemory] = resource.MustParse(memoryLimit)
	}
	return &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:      "foo",
					Resources: resources,
				},
			},
		},
	}
}
