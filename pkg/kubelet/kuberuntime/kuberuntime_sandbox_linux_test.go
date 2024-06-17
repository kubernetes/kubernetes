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

	getPodWithOverhead := func() *v1.Pod {
		return &v1.Pod{
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
		}
	}
	getPodWithoutOverhead := func() *v1.Pod {
		return &v1.Pod{
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
		}
	}

	require.NoError(t, err)

	tests := []struct {
		description      string
		pod              *v1.Pod
		expectedResource *runtimeapi.LinuxContainerResources
		expectedOverhead *runtimeapi.LinuxContainerResources
		cgroupVersion    CgroupVersion
	}{
		{
			description: "pod with overhead defined",
			pod:         getPodWithOverhead(),
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
			cgroupVersion: cgroupV1,
		},
		{
			description: "pod without overhead defined",
			pod:         getPodWithoutOverhead(),
			expectedResource: &runtimeapi.LinuxContainerResources{
				MemoryLimitInBytes: 268435456,
				CpuPeriod:          100000,
				CpuQuota:           0,
				CpuShares:          2,
			},
			expectedOverhead: &runtimeapi.LinuxContainerResources{},
			cgroupVersion:    cgroupV1,
		},
		{
			description: "pod with overhead defined",
			pod:         getPodWithOverhead(),
			expectedResource: &runtimeapi.LinuxContainerResources{
				MemoryLimitInBytes: 268435456,
				CpuPeriod:          100000,
				CpuQuota:           400000,
				CpuShares:          2048,
				Unified:            map[string]string{"memory.oom.group": "1"},
			},
			expectedOverhead: &runtimeapi.LinuxContainerResources{
				MemoryLimitInBytes: 134217728,
				CpuPeriod:          100000,
				CpuQuota:           100000,
				CpuShares:          1024,
				Unified:            map[string]string{"memory.oom.group": "1"},
			},
			cgroupVersion: cgroupV2,
		},
		{
			description: "pod without overhead defined",
			pod:         getPodWithoutOverhead(),
			expectedResource: &runtimeapi.LinuxContainerResources{
				MemoryLimitInBytes: 268435456,
				CpuPeriod:          100000,
				CpuQuota:           0,
				CpuShares:          2,
				Unified:            map[string]string{"memory.oom.group": "1"},
			},
			expectedOverhead: &runtimeapi.LinuxContainerResources{},
			cgroupVersion:    cgroupV2,
		},
	}

	for i, test := range tests {
		setCgroupVersionDuringTest(test.cgroupVersion)

		m.applySandboxResources(test.pod, config)
		assert.Equal(t, test.expectedResource, config.Linux.Resources, "TestCase[%d]: %s", i, test.description)
		assert.Equal(t, test.expectedOverhead, config.Linux.Overhead, "TestCase[%d]: %s", i, test.description)
	}
}

func TestGeneratePodSandboxConfigWithLinuxSecurityContext(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)
	pod := newTestPodWithLinuxSecurityContext()

	expectedLinuxPodSandboxConfig := &runtimeapi.LinuxPodSandboxConfig{
		SecurityContext: &runtimeapi.LinuxSandboxSecurityContext{
			SelinuxOptions: &runtimeapi.SELinuxOption{
				User: "qux",
			},
			RunAsUser:  &runtimeapi.Int64Value{Value: 1000},
			RunAsGroup: &runtimeapi.Int64Value{Value: 10},
		},
	}

	podSandboxConfig, err := m.generatePodSandboxConfig(pod, 1)
	assert.NoError(t, err)
	assert.Equal(t, expectedLinuxPodSandboxConfig.SecurityContext.SelinuxOptions, podSandboxConfig.Linux.SecurityContext.SelinuxOptions)
	assert.Equal(t, expectedLinuxPodSandboxConfig.SecurityContext.RunAsUser, podSandboxConfig.Linux.SecurityContext.RunAsUser)
	assert.Equal(t, expectedLinuxPodSandboxConfig.SecurityContext.RunAsGroup, podSandboxConfig.Linux.SecurityContext.RunAsGroup)
}

func newTestPodWithLinuxSecurityContext() *v1.Pod {
	anyGroup := int64(10)
	anyUser := int64(1000)
	pod := newTestPod()

	pod.Spec.SecurityContext = &v1.PodSecurityContext{
		SELinuxOptions: &v1.SELinuxOptions{
			User: "qux",
		},
		RunAsUser:  &anyUser,
		RunAsGroup: &anyGroup,
	}

	return pod
}

func newSupplementalGroupsPolicyPod(supplementalGroupsPolicy *v1.SupplementalGroupsPolicy) *v1.Pod {
	pod := newTestPod()
	if pod.Spec.SecurityContext == nil {
		pod.Spec.SecurityContext = &v1.PodSecurityContext{}
	}
	pod.Spec.SecurityContext.SupplementalGroupsPolicy = supplementalGroupsPolicy
	return pod
}

func TestGeneratePodSandboxLinuxConfigSupplementalGroupsPolicy(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	tests := []struct {
		description    string
		pod            *v1.Pod
		expected       string
		expectedErr    bool
		expectedErrMsg string
	}{{
		description: "SupplementalGroups=nil should convert to Merge",
		pod:         newSupplementalGroupsPolicyPod(nil),
		expected:    runtimeapi.SupplementalGroupsPolicy_Merge.String(),
	}, {
		description: "SupplementalGroups=Merge should convert to Merge",
		pod:         newSupplementalGroupsPolicyPod(&supplementalGroupsPolicyMerge),
		expected:    runtimeapi.SupplementalGroupsPolicy_Merge.String(),
	}, {
		description: "SupplementalGroups=Strict should convert to Strict",
		pod:         newSupplementalGroupsPolicyPod(&supplementalGroupsPolicyStrict),
		expected:    runtimeapi.SupplementalGroupsPolicy_Strict.String(),
	}, {
		description:    "SupplementalGroups=Unsupported should raise an error",
		pod:            newSupplementalGroupsPolicyPod(&supplementalGroupsPolicyUnSupported),
		expectedErr:    true,
		expectedErrMsg: "unsupported supplementalGroupsPolicy: UnSupported",
	},
	}

	for i, test := range tests {
		config, err := m.generatePodSandboxLinuxConfig(test.pod)
		if test.expectedErr {
			assert.NotEmptyf(t, err, "TestCase[%d]: %s", i, test.description)
			assert.Emptyf(t, config, "TestCase[%d]: %s", i, test.description)
			assert.Containsf(t, err.Error(), test.expectedErrMsg, "TestCase[%d]: %s", i, test.description)
		} else {
			actualPolicy := config.SecurityContext.SupplementalGroupsPolicy.String()
			assert.EqualValues(t, test.expected, actualPolicy, "TestCase[%d]: %s", i, test.description)
		}
	}
}
