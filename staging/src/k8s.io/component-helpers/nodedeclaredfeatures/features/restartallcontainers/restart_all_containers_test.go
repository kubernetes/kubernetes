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

package restartallcontainers

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures"
	test "k8s.io/component-helpers/nodedeclaredfeatures/testing"
)

func TestDiscover(t *testing.T) {
	tests := []struct {
		name               string
		featureGateEnabled bool
		expected           bool
	}{
		{
			name:               "both feature enabled",
			featureGateEnabled: true,
			expected:           true,
		},
		{
			name:               "restartAllContainers feature disabled",
			featureGateEnabled: false,
			expected:           false,
		},
	}

	feature := &restartAllContainersFeature{}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mockFG := test.NewMockFeatureGate(t)
			mockFG.EXPECT().Enabled(RestartAllContainersOnContainerExits).Return(tc.featureGateEnabled)

			config := &nodedeclaredfeatures.NodeConfiguration{
				FeatureGates: mockFG,
			}
			enabled := feature.Discover(config)
			assert.Equal(t, tc.expected, enabled)
		})
	}
}

func TestInferForScheduling(t *testing.T) {
	tests := []struct {
		name     string
		pod      *v1.PodSpec
		expected bool
	}{
		{
			name: "init container with rules",
			pod: &v1.PodSpec{
				InitContainers: []v1.Container{
					containerWithRestartAllContainersAction(),
				},
				Containers: []v1.Container{{
					Name:  "name",
					Image: "image",
				}},
			},
			expected: true,
		},
		{
			name: "regular container with rules",
			pod: &v1.PodSpec{
				InitContainers: []v1.Container{{
					Name:  "name",
					Image: "image",
				}},
				Containers: []v1.Container{containerWithRestartAllContainersAction()},
			},
			expected: true,
		},
		{
			name: "no rules",
			pod: &v1.PodSpec{
				InitContainers: []v1.Container{{
					Name:  "name",
					Image: "image",
				}},
				Containers: []v1.Container{{
					Name:  "name2",
					Image: "image",
				}},
			},
			expected: false,
		},
	}

	feature := &restartAllContainersFeature{}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			podInfo := &nodedeclaredfeatures.PodInfo{Spec: tc.pod}
			assert.Equal(t, tc.expected, feature.InferForScheduling(podInfo))
		})
	}
}

func TestInferForUpdate(t *testing.T) {
	feature := &restartAllContainersFeature{}
	podInfo := &nodedeclaredfeatures.PodInfo{Spec: &v1.PodSpec{}}
	assert.False(t, feature.InferForUpdate(nil, podInfo), "expect InferForUpdate to be false")
}

func containerWithRestartAllContainersAction() v1.Container {
	restartPolicy := v1.ContainerRestartPolicyNever
	return v1.Container{
		Name:          "container",
		Image:         "image",
		RestartPolicy: &restartPolicy,
		RestartPolicyRules: []v1.ContainerRestartRule{
			{
				Action: v1.ContainerRestartRuleActionRestartAllContainers,
				ExitCodes: &v1.ContainerRestartRuleOnExitCodes{
					Operator: v1.ContainerRestartRuleOnExitCodesOpIn,
					Values:   []int32{1},
				},
			},
		},
	}
}
