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

package kuberuntime

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

func seccompLocalhostRef(profileName string) string {
	return filepath.Join(fakeSeccompProfileRoot, profileName)
}

func TestGetSeccompProfile(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	unconfinedProfile := &runtimeapi.SecurityProfile{
		ProfileType: runtimeapi.SecurityProfile_Unconfined,
	}

	runtimeDefaultProfile := &runtimeapi.SecurityProfile{
		ProfileType: runtimeapi.SecurityProfile_RuntimeDefault,
	}

	tests := []struct {
		description     string
		annotation      map[string]string
		podSc           *v1.PodSecurityContext
		containerSc     *v1.SecurityContext
		containerName   string
		expectedProfile *runtimeapi.SecurityProfile
		expectedError   string
	}{
		{
			description:     "no seccomp should return unconfined",
			expectedProfile: unconfinedProfile,
		},
		{
			description:     "pod seccomp profile set to unconfined returns unconfined",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: unconfinedProfile,
		},
		{
			description:     "container seccomp profile set to unconfined returns unconfined",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: unconfinedProfile,
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: runtimeDefaultProfile,
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: runtimeDefaultProfile,
		},
		{
			description: "pod seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			podSc:       &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename")}},
			expectedProfile: &runtimeapi.SecurityProfile{
				ProfileType:  runtimeapi.SecurityProfile_Localhost,
				LocalhostRef: seccompLocalhostRef("filename"),
			},
		},
		{
			description:   "pod seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns error",
			podSc:         &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedError: "localhostProfile must be set if seccompProfile type is Localhost.",
		},
		{
			description:   "container seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns error",
			containerSc:   &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedError: "localhostProfile must be set if seccompProfile type is Localhost.",
		},
		{
			description: "container seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			containerSc: &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename2")}},
			expectedProfile: &runtimeapi.SecurityProfile{
				ProfileType:  runtimeapi.SecurityProfile_Localhost,
				LocalhostRef: seccompLocalhostRef("filename2"),
			},
		},
		{
			description:     "prioritise container field over pod field",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: runtimeDefaultProfile,
		},
		{
			description:   "prioritise container field over pod field",
			podSc:         &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("field-pod-profile.json")}},
			containerSc:   &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("field-cont-profile.json")}},
			containerName: "container1",
			expectedProfile: &runtimeapi.SecurityProfile{
				ProfileType:  runtimeapi.SecurityProfile_Localhost,
				LocalhostRef: seccompLocalhostRef("field-cont-profile.json"),
			},
		},
	}

	for i, test := range tests {
		seccompProfile, err := m.getSeccompProfile(test.annotation, test.containerName, test.podSc, test.containerSc, false)
		if test.expectedError != "" {
			assert.EqualError(t, err, test.expectedError, "TestCase[%d]: %s", i, test.description)
		} else {
			assert.NoError(t, err, "TestCase[%d]: %s", i, test.description)
			assert.Equal(t, test.expectedProfile, seccompProfile, "TestCase[%d]: %s", i, test.description)
		}
	}
}

func TestGetSeccompProfileDefaultSeccomp(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	unconfinedProfile := &runtimeapi.SecurityProfile{
		ProfileType: runtimeapi.SecurityProfile_Unconfined,
	}

	runtimeDefaultProfile := &runtimeapi.SecurityProfile{
		ProfileType: runtimeapi.SecurityProfile_RuntimeDefault,
	}

	tests := []struct {
		description     string
		annotation      map[string]string
		podSc           *v1.PodSecurityContext
		containerSc     *v1.SecurityContext
		containerName   string
		expectedProfile *runtimeapi.SecurityProfile
		expectedError   string
	}{
		{
			description:     "no seccomp should return RuntimeDefault",
			expectedProfile: runtimeDefaultProfile,
		},
		{
			description:     "pod seccomp profile set to unconfined returns unconfined",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: unconfinedProfile,
		},
		{
			description:     "container seccomp profile set to unconfined returns unconfined",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: unconfinedProfile,
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: runtimeDefaultProfile,
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: runtimeDefaultProfile,
		},
		{
			description: "pod seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			podSc:       &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename")}},
			expectedProfile: &runtimeapi.SecurityProfile{
				ProfileType:  runtimeapi.SecurityProfile_Localhost,
				LocalhostRef: seccompLocalhostRef("filename"),
			},
		},
		{
			description:   "pod seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns error",
			podSc:         &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedError: "localhostProfile must be set if seccompProfile type is Localhost.",
		},
		{
			description:   "container seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns error",
			containerSc:   &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedError: "localhostProfile must be set if seccompProfile type is Localhost.",
		},
		{
			description: "container seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			containerSc: &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename2")}},
			expectedProfile: &runtimeapi.SecurityProfile{
				ProfileType:  runtimeapi.SecurityProfile_Localhost,
				LocalhostRef: seccompLocalhostRef("filename2"),
			},
		},
		{
			description:     "prioritise container field over pod field",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: runtimeDefaultProfile,
		},
		{
			description:   "prioritise container field over pod field",
			podSc:         &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("field-pod-profile.json")}},
			containerSc:   &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("field-cont-profile.json")}},
			containerName: "container1",
			expectedProfile: &runtimeapi.SecurityProfile{
				ProfileType:  runtimeapi.SecurityProfile_Localhost,
				LocalhostRef: seccompLocalhostRef("field-cont-profile.json"),
			},
		},
	}

	for i, test := range tests {
		seccompProfile, err := m.getSeccompProfile(test.annotation, test.containerName, test.podSc, test.containerSc, true)
		if test.expectedError != "" {
			assert.EqualError(t, err, test.expectedError, "TestCase[%d]: %s", i, test.description)
		} else {
			assert.NoError(t, err, "TestCase[%d]: %s", i, test.description)
			assert.Equal(t, test.expectedProfile, seccompProfile, "TestCase[%d]: %s", i, test.description)
		}
	}
}

func getLocal(v string) *string {
	return &v
}

func TestSharesToMilliCPU(t *testing.T) {
	knownMilliCPUToShares := map[int64]int64{
		0:    2,
		1:    2,
		2:    2,
		3:    3,
		4:    4,
		32:   32,
		64:   65,
		100:  102,
		250:  256,
		500:  512,
		1000: 1024,
		1500: 1536,
		2000: 2048,
	}

	t.Run("sharesToMilliCPUTest", func(t *testing.T) {
		var testMilliCPU int64
		for testMilliCPU = 0; testMilliCPU <= 2000; testMilliCPU++ {
			shares := int64(cm.MilliCPUToShares(testMilliCPU))
			if expectedShares, found := knownMilliCPUToShares[testMilliCPU]; found {
				if shares != expectedShares {
					t.Errorf("Test milliCPIToShares: Input milliCPU %v, expected shares %v, but got %v", testMilliCPU, expectedShares, shares)
				}
			}
			expectedMilliCPU := testMilliCPU
			if testMilliCPU < 2 {
				expectedMilliCPU = 2
			}
			milliCPU := sharesToMilliCPU(shares)
			if milliCPU != expectedMilliCPU {
				t.Errorf("Test sharesToMilliCPU: Input shares %v, expected milliCPU %v, but got %v", shares, expectedMilliCPU, milliCPU)
			}
		}
	})
}

func TestQuotaToMilliCPU(t *testing.T) {
	for _, tc := range []struct {
		name     string
		quota    int64
		period   int64
		expected int64
	}{
		{
			name:     "50m",
			quota:    int64(5000),
			period:   int64(100000),
			expected: int64(50),
		},
		{
			name:     "750m",
			quota:    int64(75000),
			period:   int64(100000),
			expected: int64(750),
		},
		{
			name:     "1000m",
			quota:    int64(100000),
			period:   int64(100000),
			expected: int64(1000),
		},
		{
			name:     "1500m",
			quota:    int64(150000),
			period:   int64(100000),
			expected: int64(1500),
		}} {
		t.Run(tc.name, func(t *testing.T) {
			milliCPU := quotaToMilliCPU(tc.quota, tc.period)
			if milliCPU != tc.expected {
				t.Errorf("Test %s: Input quota %v and period %v, expected milliCPU %v, but got %v", tc.name, tc.quota, tc.period, tc.expected, milliCPU)
			}
		})
	}
}

func TestSubtractOverheadFromResourceConfig(t *testing.T) {
	podCPUMilli := resource.MustParse("200m")
	podMemory := resource.MustParse("256Mi")
	podOverheadCPUMilli := resource.MustParse("100m")
	podOverheadMemory := resource.MustParse("64Mi")

	resCfg := &cm.ResourceConfig{
		Memory:    int64Ptr(335544320),
		CPUShares: uint64Ptr(306),
		CPUPeriod: uint64Ptr(100000),
		CPUQuota:  int64Ptr(30000),
	}

	for _, tc := range []struct {
		name     string
		cfgInput *cm.ResourceConfig
		pod      *v1.Pod
		expected *cm.ResourceConfig
	}{
		{
			name:     "withoutOverhead",
			cfgInput: resCfg,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: podCPUMilli,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podCPUMilli,
									v1.ResourceMemory: podMemory,
								},
							},
						},
					},
				},
			},
			expected: &cm.ResourceConfig{
				Memory:    int64Ptr(335544320),
				CPUShares: uint64Ptr(306),
				CPUPeriod: uint64Ptr(100000),
				CPUQuota:  int64Ptr(30000),
			},
		},
		{
			name:     "withoutCPUOverhead",
			cfgInput: resCfg,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: podCPUMilli,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podCPUMilli,
									v1.ResourceMemory: podMemory,
								},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceMemory: podOverheadMemory,
					},
				},
			},
			expected: &cm.ResourceConfig{
				Memory:    int64Ptr(268435456),
				CPUShares: uint64Ptr(306),
				CPUPeriod: uint64Ptr(100000),
				CPUQuota:  int64Ptr(30000),
			},
		},
		{
			name:     "withoutMemoryOverhead",
			cfgInput: resCfg,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: podCPUMilli,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podCPUMilli,
									v1.ResourceMemory: podMemory,
								},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceCPU: podOverheadCPUMilli,
					},
				},
			},
			expected: &cm.ResourceConfig{
				Memory:    int64Ptr(335544320),
				CPUShares: uint64Ptr(203),
				CPUPeriod: uint64Ptr(100000),
				CPUQuota:  int64Ptr(20000),
			},
		},
		{
			name: "withoutCPUPeriod",
			cfgInput: &cm.ResourceConfig{
				Memory:    int64Ptr(335544320),
				CPUShares: uint64Ptr(306),
			},
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: podCPUMilli,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podCPUMilli,
									v1.ResourceMemory: podMemory,
								},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceCPU: podOverheadCPUMilli,
					},
				},
			},
			expected: &cm.ResourceConfig{
				Memory:    int64Ptr(335544320),
				CPUShares: uint64Ptr(203),
			},
		},
		{
			name: "withoutCPUShares",
			cfgInput: &cm.ResourceConfig{
				Memory:    int64Ptr(335544320),
				CPUPeriod: uint64Ptr(100000),
				CPUQuota:  int64Ptr(30000),
			},
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: podCPUMilli,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podCPUMilli,
									v1.ResourceMemory: podMemory,
								},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceCPU: podOverheadCPUMilli,
					},
				},
			},
			expected: &cm.ResourceConfig{
				Memory:    int64Ptr(335544320),
				CPUPeriod: uint64Ptr(100000),
				CPUQuota:  int64Ptr(20000),
			},
		},
		{
			name:     "withOverhead",
			cfgInput: resCfg,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: podCPUMilli,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podCPUMilli,
									v1.ResourceMemory: podMemory,
								},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    podOverheadCPUMilli,
						v1.ResourceMemory: podOverheadMemory,
					},
				},
			},
			expected: &cm.ResourceConfig{
				Memory:    int64Ptr(268435456),
				CPUShares: uint64Ptr(203),
				CPUPeriod: uint64Ptr(100000),
				CPUQuota:  int64Ptr(20000),
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotCfg := subtractOverheadFromResourceConfig(tc.cfgInput, tc.pod)

			if tc.expected.CPUPeriod != nil && *gotCfg.CPUPeriod != *tc.expected.CPUPeriod {
				t.Errorf("Test %s: expected CPUPeriod %v, but got %v", tc.name, *tc.expected.CPUPeriod, *gotCfg.CPUPeriod)
			}
			if tc.expected.CPUQuota != nil && *gotCfg.CPUQuota != *tc.expected.CPUQuota {
				t.Errorf("Test %s: expected CPUQuota %v, but got %v", tc.name, *tc.expected.CPUQuota, *gotCfg.CPUQuota)
			}
			if tc.expected.CPUShares != nil && *gotCfg.CPUShares != *tc.expected.CPUShares {
				t.Errorf("Test %s: expected CPUShares %v, but got %v", tc.name, *tc.expected.CPUShares, *gotCfg.CPUShares)
			}
			if tc.expected.Memory != nil && *gotCfg.Memory != *tc.expected.Memory {
				t.Errorf("Test %s: expected Memory %v, but got %v", tc.name, *tc.expected.Memory, *gotCfg.Memory)
			}
		})
	}
}
