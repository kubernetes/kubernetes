//go:build linux

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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestApplySandboxResources(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	m.cpuCFSQuota = true
	m.singleProcessOOMKill = ptr.To(false)

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
		description              string
		pod                      *v1.Pod
		draEnabled               bool
		podLevelResourcesEnabled bool
		expectedResource         *runtimeapi.LinuxContainerResources
		expectedOverhead         *runtimeapi.LinuxContainerResources
		cgroupVersion            CgroupVersion
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
		{
			description: "pod with DRA memory direct claims",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
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
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "direct-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("200m"))},
								{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("128Mi"))},
							},
						},
					},
				},
			},
			draEnabled: true,
			expectedResource: &runtimeapi.LinuxContainerResources{
				MemoryLimitInBytes: 402653184, // 256Mi spec + 128Mi DRA = 384Mi
				CpuPeriod:          100000,
				CpuQuota:           420000, // 4 CPUs spec + 0.2 CPU DRA = 4.2 CPUs
				CpuShares:          2252,   // 2 CPUs spec + 0.2 CPU DRA = 2.2 CPUs
				Unified:            map[string]string{"memory.oom.group": "1"},
			},
			expectedOverhead: &runtimeapi.LinuxContainerResources{},
			cgroupVersion:    cgroupV2,
		},
		{
			description: "pod with DRA direct claims and no container limits",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceMemory: resource.MustParse("128Mi"),
									v1.ResourceCPU:    resource.MustParse("2"),
								},
							},
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "direct-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("200m"))},
								{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("128Mi"))},
							},
						},
					},
				},
			},
			draEnabled: true,
			expectedResource: &runtimeapi.LinuxContainerResources{
				MemoryLimitInBytes: 0, // no container limits declared, DRA is not added to limits and the sandbox stays unlimited
				CpuPeriod:          100000,
				CpuQuota:           0,
				CpuShares:          2252, // 2 CPUs spec + 0.2 CPU DRA = 2.2 CPUs
				Unified:            map[string]string{"memory.oom.group": "1"},
			},
			expectedOverhead: &runtimeapi.LinuxContainerResources{},
			cgroupVersion:    cgroupV2,
		},
		{
			description: "pod with DRA memory direct claims and feature disabled",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
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
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "direct-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("200m"))},
								{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("128Mi"))},
							},
						},
					},
				},
			},
			draEnabled: false,
			expectedResource: &runtimeapi.LinuxContainerResources{
				MemoryLimitInBytes: 268435456, // DRA ignored, only 256Mi spec limit is used
				CpuPeriod:          100000,
				CpuQuota:           400000, // DRA ignored, only 4 CPUs spec limit is used
				CpuShares:          2048,   // DRA ignored, only 2 CPUs spec request is used
				Unified:            map[string]string{"memory.oom.group": "1"},
			},
			expectedOverhead: &runtimeapi.LinuxContainerResources{},
			cgroupVersion:    cgroupV2,
		},
		{
			description: "pod with pod-level resources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("4"),
							v1.ResourceMemory: resource.MustParse("512Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("8"),
							v1.ResourceMemory: resource.MustParse("1Gi"),
						},
					},
					Containers: []v1.Container{
						{
							Name: "c1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("1"),
									v1.ResourceMemory: resource.MustParse("128Mi"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("2"),
									v1.ResourceMemory: resource.MustParse("256Mi"),
								},
							},
						},
					},
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "direct-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("200m"))},
								{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("128Mi"))},
							},
						},
					},
				},
			},
			draEnabled:               true,
			podLevelResourcesEnabled: true,
			expectedResource: &runtimeapi.LinuxContainerResources{
				MemoryLimitInBytes: 1073741824, // Pod-level limit of 1Gi = 1073741824 bytes (overrides container limits and DRA)
				CpuPeriod:          100000,
				CpuQuota:           800000, // Pod-level limit of 8 CPUs = 800000 quota
				CpuShares:          4096,   // Pod-level request of 4 CPUs = 4096 shares
				Unified:            map[string]string{"memory.oom.group": "1"},
			},
			expectedOverhead: &runtimeapi.LinuxContainerResources{},
			cgroupVersion:    cgroupV2,
		},
		{
			description: "pod with DRA memory combined mapping and overhead claims",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
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
				},
				Status: v1.PodStatus{
					NodeAllocatableResourceClaimStatuses: []v1.NodeAllocatableResourceClaimStatus{
						{
							ResourceClaimName: "combined-claim",
							Containers:        []string{"c1"},
							Mapping: []v1.NodeAllocatableMappedResources{
								{Name: v1.ResourceCPU, Quantity: new(resource.MustParse("200m"))},
								{Name: v1.ResourceMemory, Quantity: new(resource.MustParse("128Mi"))},
							},
							Overhead: []v1.NodeAllocatableOverheadResources{
								{
									Name:         v1.ResourceCPU,
									PerPod:       new(resource.MustParse("100m")),
									PerContainer: new(resource.MustParse("50m")),
								},
								{
									Name:         v1.ResourceMemory,
									PerPod:       new(resource.MustParse("100Mi")),
									PerContainer: new(resource.MustParse("50Mi")),
								},
							},
						},
					},
				},
			},
			draEnabled: true,
			expectedResource: &runtimeapi.LinuxContainerResources{
				// spec 256Mi + mapping 128Mi + perPod 100Mi + perContainer 50Mi = 534Mi = 560000000 bytes approx.
				MemoryLimitInBytes: (256 + 128 + 100 + 50) * 1024 * 1024,
				CpuPeriod:          100000,
				// limit: spec 4 + mapping 200m + perPod 100m + perContainer 50m = 4.35 CPUs
				CpuQuota: 435000,
				// request: spec 2 + mapping 200m + perPod 100m + perContainer 50m  = 2.35 CPUs
				CpuShares: 2406,
				Unified:   map[string]string{"memory.oom.group": "1"},
			},
			expectedOverhead: &runtimeapi.LinuxContainerResources{},
			cgroupVersion:    cgroupV2,
		},
	}

	for i, test := range tests {
		setCgroupVersionDuringTest(test.cgroupVersion)
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRANodeAllocatableResources, test.draEnabled)
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, test.podLevelResourcesEnabled)

		err = m.applySandboxResources(tCtx, test.pod, config)
		require.NoError(t, err)
		assert.Equal(t, test.expectedResource, config.Linux.Resources, "TestCase[%d]: %s", i, test.description)
		assert.Equal(t, test.expectedOverhead, config.Linux.Overhead, "TestCase[%d]: %s", i, test.description)
	}
}

func TestGeneratePodSandboxConfigWithLinuxSecurityContext(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
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

	podSandboxConfig, err := m.generatePodSandboxConfig(tCtx, pod, 1)
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
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
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
		config, err := m.generatePodSandboxLinuxConfig(tCtx, test.pod)
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
