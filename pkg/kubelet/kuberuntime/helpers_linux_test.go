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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
	utilpointer "k8s.io/utils/pointer"
)

func seccompLocalhostRef(profileName string) string {
	return filepath.Join(fakeSeccompProfileRoot, profileName)
}

func seccompLocalhostPath(profileName string) string {
	return "localhost/" + seccompLocalhostRef(profileName)
}

func TestMilliCPUToQuota(t *testing.T) {
	for _, testCase := range []struct {
		msg      string
		input    int64
		expected int64
		period   uint64
	}{
		{
			msg:      "all-zero",
			input:    int64(0),
			expected: int64(0),
			period:   uint64(0),
		},
		{
			msg:      "5 input default quota and period",
			input:    int64(5),
			expected: int64(1000),
			period:   uint64(100000),
		},
		{
			msg:      "9 input default quota and period",
			input:    int64(9),
			expected: int64(1000),
			period:   uint64(100000),
		},
		{
			msg:      "10 input default quota and period",
			input:    int64(10),
			expected: int64(1000),
			period:   uint64(100000),
		},
		{
			msg:      "200 input 20k quota and default period",
			input:    int64(200),
			expected: int64(20000),
			period:   uint64(100000),
		},
		{
			msg:      "500 input 50k quota and default period",
			input:    int64(500),
			expected: int64(50000),
			period:   uint64(100000),
		},
		{
			msg:      "1k input 100k quota and default period",
			input:    int64(1000),
			expected: int64(100000),
			period:   uint64(100000),
		},
		{
			msg:      "1500 input 150k quota and default period",
			input:    int64(1500),
			expected: int64(150000),
			period:   uint64(100000),
		}} {
		t.Run(testCase.msg, func(t *testing.T) {
			quota := milliCPUToQuota(testCase.input, int64(testCase.period))
			if quota != testCase.expected {
				t.Errorf("Input %v and %v, expected quota %v, but got quota %v", testCase.input, testCase.period, testCase.expected, quota)
			}
		})
	}
}

func TestMilliCPUToQuotaWithCustomCPUCFSQuotaPeriod(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUCFSQuotaPeriod, true)()

	for _, testCase := range []struct {
		msg      string
		input    int64
		expected int64
		period   uint64
	}{
		{
			msg:      "all-zero",
			input:    int64(0),
			expected: int64(0),
			period:   uint64(0),
		},
		{
			msg:      "5 input default quota and period",
			input:    int64(5),
			expected: minQuotaPeriod,
			period:   uint64(100000),
		},
		{
			msg:      "9 input default quota and period",
			input:    int64(9),
			expected: minQuotaPeriod,
			period:   uint64(100000),
		},
		{
			msg:      "10 input default quota and period",
			input:    int64(10),
			expected: minQuotaPeriod,
			period:   uint64(100000),
		},
		{
			msg:      "200 input 20k quota and default period",
			input:    int64(200),
			expected: int64(20000),
			period:   uint64(100000),
		},
		{
			msg:      "500 input 50k quota and default period",
			input:    int64(500),
			expected: int64(50000),
			period:   uint64(100000),
		},
		{
			msg:      "1k input 100k quota and default period",
			input:    int64(1000),
			expected: int64(100000),
			period:   uint64(100000),
		},
		{
			msg:      "1500 input 150k quota and default period",
			input:    int64(1500),
			expected: int64(150000),
			period:   uint64(100000),
		},
		{
			msg:      "5 input 10k period and default quota expected",
			input:    int64(5),
			period:   uint64(10000),
			expected: minQuotaPeriod,
		},
		{
			msg:      "5 input 5k period and default quota expected",
			input:    int64(5),
			period:   uint64(5000),
			expected: minQuotaPeriod,
		},
		{
			msg:      "9 input 10k period and default quota expected",
			input:    int64(9),
			period:   uint64(10000),
			expected: minQuotaPeriod,
		},
		{
			msg:      "10 input 200k period and 2000 quota expected",
			input:    int64(10),
			period:   uint64(200000),
			expected: int64(2000),
		},
		{
			msg:      "200 input 200k period and 40k quota",
			input:    int64(200),
			period:   uint64(200000),
			expected: int64(40000),
		},
		{
			msg:      "500 input 20k period and 20k expected quota",
			input:    int64(500),
			period:   uint64(20000),
			expected: int64(10000),
		},
		{
			msg:      "1000 input 10k period and 10k expected quota",
			input:    int64(1000),
			period:   uint64(10000),
			expected: int64(10000),
		},
		{
			msg:      "1500 input 5000 period and 7500 expected quota",
			input:    int64(1500),
			period:   uint64(5000),
			expected: int64(7500),
		}} {
		t.Run(testCase.msg, func(t *testing.T) {
			quota := milliCPUToQuota(testCase.input, int64(testCase.period))
			if quota != testCase.expected {
				t.Errorf("Input %v and %v, expected quota %v, but got quota %v", testCase.input, testCase.period, testCase.expected, quota)
			}
		})
	}
}

func TestFieldProfile(t *testing.T) {
	tests := []struct {
		description     string
		scmpProfile     *v1.SeccompProfile
		rootPath        string
		expectedProfile string
	}{
		{
			description:     "no seccompProfile should return empty",
			expectedProfile: "",
		},
		{
			description: "type localhost without profile should return empty",
			scmpProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeLocalhost,
			},
			expectedProfile: "",
		},
		{
			description: "unknown type should return empty",
			scmpProfile: &v1.SeccompProfile{
				Type: "",
			},
			expectedProfile: "",
		},
		{
			description: "SeccompProfileTypeRuntimeDefault should return runtime/default",
			scmpProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeRuntimeDefault,
			},
			expectedProfile: "runtime/default",
		},
		{
			description: "SeccompProfileTypeUnconfined should return unconfined",
			scmpProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeUnconfined,
			},
			expectedProfile: "unconfined",
		},
		{
			description: "SeccompProfileTypeLocalhost should return localhost",
			scmpProfile: &v1.SeccompProfile{
				Type:             v1.SeccompProfileTypeLocalhost,
				LocalhostProfile: utilpointer.StringPtr("profile.json"),
			},
			rootPath:        "/test/",
			expectedProfile: "localhost//test/profile.json",
		},
	}

	for i, test := range tests {
		seccompProfile := fieldProfile(test.scmpProfile, test.rootPath, false)
		assert.Equal(t, test.expectedProfile, seccompProfile, "TestCase[%d]: %s", i, test.description)
	}
}

func TestFieldProfileDefaultSeccomp(t *testing.T) {
	tests := []struct {
		description     string
		scmpProfile     *v1.SeccompProfile
		rootPath        string
		expectedProfile string
	}{
		{
			description:     "no seccompProfile should return runtime/default",
			expectedProfile: v1.SeccompProfileRuntimeDefault,
		},
		{
			description: "type localhost without profile should return runtime/default",
			scmpProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeLocalhost,
			},
			expectedProfile: v1.SeccompProfileRuntimeDefault,
		},
		{
			description: "unknown type should return runtime/default",
			scmpProfile: &v1.SeccompProfile{
				Type: "",
			},
			expectedProfile: v1.SeccompProfileRuntimeDefault,
		},
		{
			description: "SeccompProfileTypeRuntimeDefault should return runtime/default",
			scmpProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeRuntimeDefault,
			},
			expectedProfile: "runtime/default",
		},
		{
			description: "SeccompProfileTypeUnconfined should return unconfined",
			scmpProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeUnconfined,
			},
			expectedProfile: "unconfined",
		},
		{
			description: "SeccompProfileTypeLocalhost should return localhost",
			scmpProfile: &v1.SeccompProfile{
				Type:             v1.SeccompProfileTypeLocalhost,
				LocalhostProfile: utilpointer.StringPtr("profile.json"),
			},
			rootPath:        "/test/",
			expectedProfile: "localhost//test/profile.json",
		},
	}

	for i, test := range tests {
		seccompProfile := fieldProfile(test.scmpProfile, test.rootPath, true)
		assert.Equal(t, test.expectedProfile, seccompProfile, "TestCase[%d]: %s", i, test.description)
	}
}

func TestGetSeccompProfilePath(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	tests := []struct {
		description     string
		annotation      map[string]string
		podSc           *v1.PodSecurityContext
		containerSc     *v1.SecurityContext
		containerName   string
		expectedProfile string
	}{
		{
			description:     "no seccomp should return empty",
			expectedProfile: "",
		},
		{
			description:     "annotations: no seccomp with containerName should return empty",
			containerName:   "container1",
			expectedProfile: "",
		},
		{
			description:     "pod seccomp profile set to unconfined returns unconfined",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: "unconfined",
		},
		{
			description:     "container seccomp profile set to unconfined returns unconfined",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: "unconfined",
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: "runtime/default",
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: "runtime/default",
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename")}},
			expectedProfile: seccompLocalhostPath("filename"),
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns empty",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: "",
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns empty",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: "",
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename2")}},
			expectedProfile: seccompLocalhostPath("filename2"),
		},
		{
			description:     "prioritise container field over pod field",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: "runtime/default",
		},
	}

	for i, test := range tests {
		seccompProfile := m.getSeccompProfilePath(test.annotation, test.containerName, test.podSc, test.containerSc, false)
		assert.Equal(t, test.expectedProfile, seccompProfile, "TestCase[%d]: %s", i, test.description)
	}
}

func TestGetSeccompProfilePathDefaultSeccomp(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	tests := []struct {
		description     string
		annotation      map[string]string
		podSc           *v1.PodSecurityContext
		containerSc     *v1.SecurityContext
		containerName   string
		expectedProfile string
	}{
		{
			description:     "no seccomp should return runtime/default",
			expectedProfile: v1.SeccompProfileRuntimeDefault,
		},
		{
			description:     "annotations: no seccomp with containerName should return runtime/default",
			containerName:   "container1",
			expectedProfile: v1.SeccompProfileRuntimeDefault,
		},
		{
			description:     "pod seccomp profile set to unconfined returns unconfined",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: "unconfined",
		},
		{
			description:     "container seccomp profile set to unconfined returns unconfined",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: "unconfined",
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: "runtime/default",
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: "runtime/default",
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename")}},
			expectedProfile: seccompLocalhostPath("filename"),
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns runtime/default",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: v1.SeccompProfileRuntimeDefault,
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns runtime/default",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: v1.SeccompProfileRuntimeDefault,
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename2")}},
			expectedProfile: seccompLocalhostPath("filename2"),
		},
		{
			description:     "prioritise container field over pod field",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: "runtime/default",
		},
	}

	for i, test := range tests {
		seccompProfile := m.getSeccompProfilePath(test.annotation, test.containerName, test.podSc, test.containerSc, true)
		assert.Equal(t, test.expectedProfile, seccompProfile, "TestCase[%d]: %s", i, test.description)
	}
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
			description:     "pod seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns unconfined",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: unconfinedProfile,
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns unconfined",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: unconfinedProfile,
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
		seccompProfile := m.getSeccompProfile(test.annotation, test.containerName, test.podSc, test.containerSc, false)
		assert.Equal(t, test.expectedProfile, seccompProfile, "TestCase[%d]: %s", i, test.description)
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
			description:     "pod seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns unconfined",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: unconfinedProfile,
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns unconfined",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: unconfinedProfile,
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
		seccompProfile := m.getSeccompProfile(test.annotation, test.containerName, test.podSc, test.containerSc, true)
		assert.Equal(t, test.expectedProfile, seccompProfile, "TestCase[%d]: %s", i, test.description)
	}
}

func getLocal(v string) *string {
	return &v
}
