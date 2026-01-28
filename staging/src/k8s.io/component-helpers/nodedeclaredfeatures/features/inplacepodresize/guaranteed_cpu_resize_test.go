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
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures"
	test "k8s.io/component-helpers/nodedeclaredfeatures/testing"
)

func TestGuaranteedQoSPodCPUResizeFeature_Requirements(t *testing.T) {
	feature := &guaranteedQoSPodCPUResizeFeature{}
	reqs := feature.Requirements()
	assert.NotNil(t, reqs)
	assert.Equal(t, []string{IPPRExclusiveCPUsFeatureGate}, reqs.EnabledFeatureGates)
	assert.Equal(t, map[string]string{"CPUManagerPolicy": "static"}, reqs.StaticConfig)
}

func TestGuaranteedQoSPodCPUResizeFeature_Discover(t *testing.T) {
	testCases := []struct {
		name             string
		cpuManagerPolicy string
		gateEnabled      bool
		expected         bool
	}{
		{
			name:             "feature gate enabled, static policy",
			cpuManagerPolicy: CPUManagerPolicyStatic,
			gateEnabled:      true,
			expected:         true,
		},
		{
			name:             "feature gate disabled, static policy",
			cpuManagerPolicy: CPUManagerPolicyStatic,
			gateEnabled:      false,
			expected:         false,
		},
		{
			name:             "none policy",
			cpuManagerPolicy: CPUManagerPolicyNone,
			gateEnabled:      true,
			expected:         true,
		},
		{
			name:             "feature gate missing",
			cpuManagerPolicy: CPUManagerPolicyStatic,
			gateEnabled:      false, // Effectively disabled if not found
			expected:         false,
		},
	}

	feature := &guaranteedQoSPodCPUResizeFeature{}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockFG := test.NewMockFeatureGate(t)
			mockFG.EXPECT().CheckEnabled(IPPRExclusiveCPUsFeatureGate).Return(tc.gateEnabled, nil)

			config := &nodedeclaredfeatures.NodeConfiguration{
				FeatureGates: mockFG,
				StaticConfig: nodedeclaredfeatures.StaticConfiguration{CPUManagerPolicy: tc.cpuManagerPolicy},
			}
			enabled, err := feature.Discover(config)
			require.NoError(t, err)
			assert.Equal(t, tc.expected, enabled)
		})
	}
}

func TestGuaranteedQoSPodCPUResizeFeature_InferForScheduling(t *testing.T) {
	feature := &guaranteedQoSPodCPUResizeFeature{}
	assert.False(t, feature.InferForScheduling(&nodedeclaredfeatures.PodInfo{Spec: &v1.PodSpec{}, Status: &v1.PodStatus{}}), "InferForScheduling should always be false")
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
			oldPodInfo := &nodedeclaredfeatures.PodInfo{Spec: &tc.oldPod.Spec, Status: &tc.oldPod.Status}
			newPodInfo := &nodedeclaredfeatures.PodInfo{Spec: &tc.newPod.Spec, Status: &tc.newPod.Status}
			assert.Equal(t, tc.expected, feature.InferForUpdate(oldPodInfo, newPodInfo))
		})
	}
}
