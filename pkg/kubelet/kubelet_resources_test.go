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

package kubelet

import (
	"testing"

	"github.com/stretchr/testify/assert"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPodResourceLimitsDefaulting(t *testing.T) {
	cpuCores := resource.MustParse("10")
	memoryCapacity := resource.MustParse("10Gi")
	tk := newTestKubelet(t, true)
	defer tk.Cleanup()
	tk.fakeCadvisor.On("VersionInfo").Return(&cadvisorapi.VersionInfo{}, nil)
	tk.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{
		NumCores:       int(cpuCores.Value()),
		MemoryCapacity: uint64(memoryCapacity.Value()),
	}, nil)
	tk.fakeCadvisor.On("ImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	tk.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	tk.kubelet.nodeInfo = &testNodeInfo{
		nodes: []*v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: string(tk.kubelet.nodeName),
				},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("6"),
						v1.ResourceMemory: resource.MustParse("4Gi"),
					},
				},
			},
		},
	}
	cases := []struct {
		pod      *v1.Pod
		expected *v1.Pod
	}{
		{
			pod:      getPod("0", "0"),
			expected: getPod("6", "4Gi"),
		},
		{
			pod:      getPod("1", "0"),
			expected: getPod("1", "4Gi"),
		},
		{
			pod:      getPod("", ""),
			expected: getPod("6", "4Gi"),
		},
		{
			pod:      getPod("0", "1Mi"),
			expected: getPod("6", "1Mi"),
		},
	}
	as := assert.New(t)
	for idx, tc := range cases {
		actual, _, err := tk.kubelet.defaultPodLimitsForDownwardAPI(tc.pod, nil)
		as.Nil(err, "failed to default pod limits: %v", err)
		if !apiequality.Semantic.DeepEqual(tc.expected, actual) {
			as.Fail("test case [%d] failed.  Expected: %+v, Got: %+v", idx, tc.expected, actual)
		}
	}
}

func getPod(cpuLimit, memoryLimit string) *v1.Pod {
	resources := v1.ResourceRequirements{}
	if cpuLimit != "" || memoryLimit != "" {
		resources.Limits = make(v1.ResourceList)
	}
	if cpuLimit != "" {
		resources.Limits[v1.ResourceCPU] = resource.MustParse(cpuLimit)
	}
	if memoryLimit != "" {
		resources.Limits[v1.ResourceMemory] = resource.MustParse(memoryLimit)
	}
	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      "foo",
					Resources: resources,
				},
			},
		},
	}
}
