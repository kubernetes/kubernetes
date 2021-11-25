//go:build linux
// +build linux

/*
Copyright 2021 The Kubernetes Authors.

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

package kuberuntime

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

func TestApplySandboxResources(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	m.cpuCFSQuota = true

	config := &runtimeapi.PodSandboxConfig{
		Linux: &runtimeapi.LinuxPodSandboxConfig{},
	}

	require.NoError(t, err)

	tests := []struct {
		description      string
		pod              *v1.Pod
		expectedResource *runtimeapi.LinuxContainerResources
		expectedOverhead *runtimeapi.LinuxContainerResources
	}{
		{
			description: "pod with overhead defined",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					UID:       "12345678",
					Name:      "bar",
					Namespace: "new",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceMemory: resource.MustParse("128Mi"),
									v1.ResourceCPU:    resource.MustParse("2"),
								},
								Limits: v1.ResourceList{
									v1.ResourceMemory: resource.MustParse("256Mi"),
									v1.ResourceCPU:    resource.MustParse("4"),
								},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse("128Mi"),
						v1.ResourceCPU:    resource.MustParse("1"),
					},
				},
			},
			expectedResource: &runtimeapi.LinuxContainerResources{
				MemoryLimitInBytes: 268435456,
				CpuPeriod:          100000,
				CpuQuota:           400000,
				CpuShares:          2048,
			},
			expectedOverhead: &runtimeapi.LinuxContainerResources{
				MemoryLimitInBytes: 134217728,
				CpuPeriod:          100000,
				CpuQuota:           100000,
				CpuShares:          1024,
			},
		},
		{
			description: "pod without overhead defined",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					UID:       "12345678",
					Name:      "bar",
					Namespace: "new",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceMemory: resource.MustParse("128Mi"),
								},
								Limits: v1.ResourceList{
									v1.ResourceMemory: resource.MustParse("256Mi"),
								},
							},
						},
					},
				},
			},
			expectedResource: &runtimeapi.LinuxContainerResources{
				MemoryLimitInBytes: 268435456,
				CpuPeriod:          100000,
				CpuQuota:           0,
				CpuShares:          2,
			},
			expectedOverhead: &runtimeapi.LinuxContainerResources{},
		},
	}

	for i, test := range tests {
		m.applySandboxResources(test.pod, config)
		assert.Equal(t, test.expectedResource, config.Linux.Resources, "TestCase[%d]: %s", i, test.description)
		assert.Equal(t, test.expectedOverhead, config.Linux.Overhead, "TestCase[%d]: %s", i, test.description)
	}
}
