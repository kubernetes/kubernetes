/*
Copyright 2025 The Kubernetes Authors.

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

package inplacepodresize

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures"
)

func TestGuaranteedQoSPodCPUResizeFeature_Discover(t *testing.T) {
	testCases := []struct {
		name          string
		config        *nodedeclaredfeatures.NodeConfiguration
		expected      bool
		expectErr     bool
		expectedError string
	}{
		{
			name: "feature gate enabled, static policy",
			config: &nodedeclaredfeatures.NodeConfiguration{
				FeatureGates:  map[string]bool{IPPRExclusiveCPUsFeatureGate: true},
				KubeletConfig: map[string]string{CPUManagerPolicyConfigName: CPUManagerPolicyStatic},
			},
			expected:  true,
			expectErr: false,
		},
		{
			name: "feature gate disabled, static policy",
			config: &nodedeclaredfeatures.NodeConfiguration{
				FeatureGates:  map[string]bool{IPPRExclusiveCPUsFeatureGate: false},
				KubeletConfig: map[string]string{CPUManagerPolicyConfigName: CPUManagerPolicyStatic},
			},
			expected:  false,
			expectErr: false,
		},
		{
			name: "none policy",
			config: &nodedeclaredfeatures.NodeConfiguration{
				FeatureGates:  map[string]bool{IPPRExclusiveCPUsFeatureGate: true}, // gate doesn't matter
				KubeletConfig: map[string]string{CPUManagerPolicyConfigName: CPUManagerPolicyNone},
			},
			expected:  true,
			expectErr: false,
		},
		{
			name: "feature gate missing",
			config: &nodedeclaredfeatures.NodeConfiguration{
				FeatureGates:  map[string]bool{},
				KubeletConfig: map[string]string{CPUManagerPolicyConfigName: CPUManagerPolicyStatic},
			},
			expectErr:     true,
			expectedError: "feature gate \"InPlacePodVerticalScalingExclusiveCPUs\" not found",
		},
		{
			name: "config missing",
			config: &nodedeclaredfeatures.NodeConfiguration{
				FeatureGates:  map[string]bool{IPPRExclusiveCPUsFeatureGate: true},
				KubeletConfig: map[string]string{},
			},
			expectErr:     true,
			expectedError: "config \"cpuManagerPolicy\" not found",
		},
	}

	feature := &guaranteedQoSPodCPUResizeFeature{}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			enabled, err := feature.Discover(tc.config)
			if tc.expectErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tc.expectedError)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected, enabled)
			}
		})
	}
}

func TestGuaranteedQoSPodCPUResizeFeature_InferFromCreate(t *testing.T) {
	feature := &guaranteedQoSPodCPUResizeFeature{}
	assert.False(t, feature.InferFromCreate(&nodedeclaredfeatures.PodInfo{Pod: &v1.Pod{}}), "InferFromCreate should always be false")
}

func TestGuaranteedQoSPodCPUResizeFeature_InferFromUpdate(t *testing.T) {
	basePod := func() *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "test-pod"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "c1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
							Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
						},
					},
				},
			},
			Status: v1.PodStatus{
				QOSClass: v1.PodQOSGuaranteed,
			},
		}
	}

	testCases := []struct {
		name     string
		oldPod   *v1.Pod
		newPod   *v1.Pod
		expected bool
	}{
		{
			name:     "not guaranteed QOS",
			oldPod:   func() *v1.Pod { p := basePod(); p.Status.QOSClass = v1.PodQOSBurstable; return p }(),
			newPod:   func() *v1.Pod { p := basePod(); p.Status.QOSClass = v1.PodQOSBurstable; return p }(),
			expected: false,
		},
		{
			name:     "no CPU request change",
			oldPod:   basePod(),
			newPod:   basePod(),
			expected: false,
		},
		{
			name:   "CPU request changed",
			oldPod: basePod(),
			newPod: func() *v1.Pod {
				p := basePod()
				p.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("2")
				return p
			}(),
			expected: true,
		},
		{
			name:   "CPU limit changed, but not request",
			oldPod: basePod(),
			newPod: func() *v1.Pod {
				p := basePod()
				p.Spec.Containers[0].Resources.Limits[v1.ResourceCPU] = resource.MustParse("2")
				return p
			}(),
			expected: false, // Only request changes matter for this feature
		},
	}

	feature := &guaranteedQoSPodCPUResizeFeature{}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			oldPodInfo := &nodedeclaredfeatures.PodInfo{Pod: tc.oldPod}
			newPodInfo := &nodedeclaredfeatures.PodInfo{Pod: tc.newPod}
			assert.Equal(t, tc.expected, feature.InferFromUpdate(oldPodInfo, newPodInfo))
		})
	}
}
