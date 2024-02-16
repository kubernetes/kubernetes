/*
Copyright 2022 The Kubernetes Authors.

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

package userns

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
)

// Test all public methods behave ok when the feature gate is disabled.

func TestMakeUserNsManagerDisabled(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, false)()

	testUserNsPodsManager := &testUserNsPodsManager{}
	_, err := MakeUserNsManager(testUserNsPodsManager)
	assert.NoError(t, err)
}

func TestReleaseDisabled(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, false)()

	testUserNsPodsManager := &testUserNsPodsManager{}
	m, err := MakeUserNsManager(testUserNsPodsManager)
	require.NoError(t, err)

	m.Release("some-pod")
}

func TestGetOrCreateUserNamespaceMappingsDisabled(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, false)()

	trueVal := true
	falseVal := false

	tests := []struct {
		name    string
		pod     *v1.Pod
		success bool
	}{
		{
			name:    "pod is nil",
			pod:     nil,
			success: true,
		},
		{
			name: "hostUsers is nil",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostUsers: nil,
				},
			},
			success: true,
		},
		{
			name: "hostUsers is true",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostUsers: &trueVal,
				},
			},
		},
		{
			name: "hostUsers is false",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					HostUsers: &falseVal,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testUserNsPodsManager := &testUserNsPodsManager{}
			m, err := MakeUserNsManager(testUserNsPodsManager)
			require.NoError(t, err)

			userns, err := m.GetOrCreateUserNamespaceMappings(test.pod, "")
			assert.Nil(t, userns)
			if test.success {
				assert.NoError(t, err)
			} else {
				assert.Error(t, err)
			}
		})
	}
}

func TestCleanupOrphanedPodUsernsAllocationsDisabled(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, false)()

	testUserNsPodsManager := &testUserNsPodsManager{}
	m, err := MakeUserNsManager(testUserNsPodsManager)
	require.NoError(t, err)

	err = m.CleanupOrphanedPodUsernsAllocations(nil, nil)
	assert.NoError(t, err)
}
