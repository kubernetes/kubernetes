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

package nodedeclaredfeatures

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
)

// mockFeature is a mock implementation of the Feature interface for testing.
type mockFeature struct {
	name        string
	discover    func(cfg *NodeConfiguration) (bool, error)
	inferCreate func(podInfo *PodInfo) bool
	inferUpdate func(oldPodInfo, newPodInfo *PodInfo) bool
	minVersion  *version.Version
	maxVersion  *version.Version
}

func (f *mockFeature) Name() string                                  { return f.name }
func (f *mockFeature) Discover(cfg *NodeConfiguration) (bool, error) { return f.discover(cfg) }
func (f *mockFeature) InferFromCreate(podInfo *PodInfo) bool         { return f.inferCreate(podInfo) }
func (f *mockFeature) InferFromUpdate(oldPodInfo, newPodInfo *PodInfo) bool {
	return f.inferUpdate(oldPodInfo, newPodInfo)
}
func (f *mockFeature) MinVersion() *version.Version { return f.minVersion }
func (f *mockFeature) MaxVersion() *version.Version { return f.maxVersion }

func TestNewHelper(t *testing.T) {
	_, err := NewHelper(nil)
	require.Error(t, err, "NewHelper should return an error with a nil registry")

	_, err = NewHelper([]Feature{})
	require.NoError(t, err, "NewHelper should not return an error with an empty registry")
}

func TestDiscoverNodeFeatures(t *testing.T) {
	registry := []Feature{
		&mockFeature{
			name: "FeatureA",
			discover: func(cfg *NodeConfiguration) (bool, error) {
				return cfg.FeatureGates["feature-a"], nil
			},
		},
		&mockFeature{
			name: "FeatureBWithKubeletConfig",
			discover: func(cfg *NodeConfiguration) (bool, error) {
				return cfg.FeatureGates["feature-b"] && cfg.KubeletConfig["config-b"] == "xyz", nil
			},
		},
		&mockFeature{
			name: "ErrorFeature",
			discover: func(cfg *NodeConfiguration) (bool, error) {
				if cfg.FeatureGates["error-feature"] {
					return false, fmt.Errorf("discovery error")
				}
				return false, nil
			},
		},
	}

	helper, _ := NewHelper(registry)

	testCases := []struct {
		name          string
		config        *NodeConfiguration
		expected      []string
		expectErr     bool
		expectedError string
	}{
		{
			name: "FeatureAEnabled",
			config: &NodeConfiguration{
				FeatureGates: map[string]bool{"feature-a": true},
			},
			expected:  []string{"FeatureA"},
			expectErr: false,
		},
		{
			name: "feature-b enabled",
			config: &NodeConfiguration{
				FeatureGates:  map[string]bool{"feature-b": true},
				KubeletConfig: map[string]string{"config-b": "xyz"},
			},
			expected:  []string{"FeatureBWithKubeletConfig"},
			expectErr: false,
		},
		{
			name: "both features enabled",
			config: &NodeConfiguration{
				FeatureGates:  map[string]bool{"feature-a": true, "feature-b": true},
				KubeletConfig: map[string]string{"config-b": "xyz"},
			},
			expected:  []string{"FeatureA", "FeatureBWithKubeletConfig"}, // Should be sorted
			expectErr: false,
		},
		{
			name: "no features enabled",
			config: &NodeConfiguration{
				FeatureGates:  map[string]bool{"feature-a": false, "feature-b": true},
				KubeletConfig: map[string]string{"config-b": "abc"},
			},
			expected:  []string{},
			expectErr: false,
		},
		{
			name: "discovery error",
			config: &NodeConfiguration{
				FeatureGates: map[string]bool{"error-feature": true},
			},
			expectErr:     true,
			expectedError: "discovery error",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			features, err := helper.DiscoverNodeFeatures(tc.config)
			if tc.expectErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.expectedError)
			} else {
				require.NoError(t, err)
				if len(tc.expected) == 0 {
					assert.Empty(t, features)
				} else {
					assert.Equal(t, tc.expected, features)
				}
			}
		})
	}
}

func TestInferForPodCreate(t *testing.T) {
	inferPodlevelRes := func(p *PodInfo) bool {
		return p.Pod.Spec.Resources != nil
	}

	podWithPodLevelResources := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-with-podlevelresources"},
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("500m"),
				},
			},
		},
	}
	podWithoutPodLevelResources := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-without-podlevelresources"},
		Spec:       v1.PodSpec{},
	}

	testCases := []struct {
		name          string
		registry      []Feature
		newPod        *v1.Pod
		targetVersion string
		expectedReqs  []string
		expectErr     bool
		errContains   string
	}{
		{
			name: "pod with feature, inferred furing create",
			registry: []Feature{
				&mockFeature{
					name:        "PodLevelResources",
					inferCreate: inferPodlevelRes,
					minVersion:  version.MustParseSemantic("1.30.0"),
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: "1.30.0",
			expectedReqs:  []string{"PodLevelResources"},
			expectErr:     false,
		},
		{
			name: "pod without feature",
			registry: []Feature{
				&mockFeature{
					name:        "PodLevelResources",
					inferCreate: inferPodlevelRes,
					minVersion:  version.MustParseSemantic("1.30.0"),
				},
			},
			newPod:        podWithoutPodLevelResources,
			targetVersion: "1.30.0",
			expectedReqs:  []string{},
			expectErr:     false,
		},
		{
			name: "incompatible feature version",
			registry: []Feature{
				&mockFeature{
					name:        "PodLevelResources",
					inferCreate: inferPodlevelRes,
					minVersion:  version.MustParseSemantic("1.31.0"),
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: "1.30.0",
			expectErr:     true,
			errContains:   "feature \"PodLevelResources\" is not available",
		},
		{
			name: "feature universally available, not inferred during create",
			registry: []Feature{
				&mockFeature{
					name:        "PodLevelResources",
					inferCreate: inferPodlevelRes,
					minVersion:  version.MustParseSemantic("1.27.0"),
					maxVersion:  version.MustParseSemantic("1.30.0"),
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: "1.31.0",
			expectedReqs:  []string{},
			expectErr:     false,
		},
		{
			name: "invalid target version",
			registry: []Feature{
				&mockFeature{
					name:        "PodLevelResources",
					inferCreate: inferPodlevelRes,
					minVersion:  version.MustParseSemantic("1.30.0"),
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: "invalid-version",
			expectErr:     true,
			errContains:   "failed to parse target version",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			helper, _ := NewHelper(tc.registry)
			reqs, err := helper.InferForPodCreate(&PodInfo{Pod: tc.newPod}, tc.targetVersion)

			if tc.expectErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errContains)
			} else {
				require.NoError(t, err)
				if len(tc.expectedReqs) == 0 {
					assert.Empty(t, reqs)
				} else {
					assert.Equal(t, tc.expectedReqs, reqs)
				}
			}
		})
	}
}
func TestInferForPodUpdate(t *testing.T) {
	inferResize := func(oldPodInfo, newPodInfo *PodInfo) bool {
		oldCPU := oldPodInfo.Pod.Spec.Containers[0].Resources.Requests.Cpu()
		newCPU := newPodInfo.Pod.Spec.Containers[0].Resources.Requests.Cpu()
		if oldCPU != nil && newCPU != nil && !oldCPU.Equal(*newCPU) {
			return true
		}
		return false
	}

	basePod := func(cpuRequest string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "test-pod"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "c1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(cpuRequest)},
						},
					},
				},
			},
		}
	}

	podWith1CPU := basePod("1")
	podWith2CPU := basePod("2")

	testCases := []struct {
		name          string
		registry      []Feature
		oldPod        *v1.Pod
		newPod        *v1.Pod
		targetVersion string
		expectedReqs  []string
		expectErr     bool
		errContains   string
	}{
		{
			name: "pod update requires feature",
			registry: []Feature{
				&mockFeature{
					name:        "InPlacePodResize",
					inferUpdate: inferResize,
					minVersion:  version.MustParseSemantic("1.30.0"),
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: "1.30.0",
			expectedReqs:  []string{"InPlacePodResize"},
			expectErr:     false,
		},
		{
			name: "pod update does not require feature",
			registry: []Feature{
				&mockFeature{
					name:        "InPlacePodResize",
					inferUpdate: inferResize,
					minVersion:  version.MustParseSemantic("1.30.0"),
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith1CPU,
			targetVersion: "1.30.0",
			expectedReqs:  []string{},
			expectErr:     false,
		},
		{
			name: "incompatible feature version",
			registry: []Feature{
				&mockFeature{
					name:        "InPlacePodResize",
					inferUpdate: inferResize,
					minVersion:  version.MustParseSemantic("1.31.0"),
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: "1.30.0",
			expectErr:     true,
			errContains:   "feature \"InPlacePodResize\" is not available",
		},
		{
			name: "feature universally available, not inferred during update",
			registry: []Feature{
				&mockFeature{
					name:        "InPlacePodResize",
					inferUpdate: inferResize,
					minVersion:  version.MustParseSemantic("1.27.0"),
					maxVersion:  version.MustParseSemantic("1.30.0"),
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: "1.31.0",
			expectedReqs:  []string{},
			expectErr:     false,
		},
		{
			name: "invalid target version",
			registry: []Feature{
				&mockFeature{
					name:        "InPlacePodResize",
					inferUpdate: inferResize,
					minVersion:  version.MustParseSemantic("1.30.0"),
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: "invalid-version",
			expectErr:     true,
			errContains:   "failed to parse target version",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			helper, _ := NewHelper(tc.registry)
			reqs, err := helper.InferForPodUpdate(&PodInfo{Pod: tc.oldPod}, &PodInfo{Pod: tc.newPod}, tc.targetVersion)

			if tc.expectErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errContains)
			} else {
				require.NoError(t, err)
				if len(tc.expectedReqs) == 0 {
					assert.Empty(t, reqs)
				} else {
					assert.Equal(t, tc.expectedReqs, reqs)
				}
			}
		})
	}
}

func TestMatchNode(t *testing.T) {
	helper, _ := NewHelper([]Feature{})

	testCases := []struct {
		name                   string
		podFeatureRequirements []string
		nodeFeatures           []string
		expectedMatch          bool
		expectedUnsatisfied    []string
	}{
		{
			name:                   "all features match",
			podFeatureRequirements: []string{"feature-a", "feature-b"},
			nodeFeatures:           []string{"feature-a", "feature-b", "feature-c"},
			expectedMatch:          true,
			expectedUnsatisfied:    nil,
		},
		{
			name:                   "some features missing",
			podFeatureRequirements: []string{"feature-a", "feature-b"},
			nodeFeatures:           []string{"feature-a", "feature-c"},
			expectedMatch:          false,
			expectedUnsatisfied:    []string{"feature-b"},
		},
		{
			name:                   "all features missing",
			podFeatureRequirements: []string{"feature-a", "feature-b"},
			nodeFeatures:           []string{"feature-c"},
			expectedMatch:          false,
			expectedUnsatisfied:    []string{"feature-a", "feature-b"},
		},
		{
			name:                   "no node features",
			podFeatureRequirements: []string{"feature-a", "feature-b"},
			nodeFeatures:           []string{},
			expectedMatch:          false,
			expectedUnsatisfied:    []string{"feature-a", "feature-b"},
		},
		{
			name:                   "no requirements",
			podFeatureRequirements: []string{},
			nodeFeatures:           []string{"feature-a", "feature-b", "feature-c"},
			expectedMatch:          true,
			expectedUnsatisfied:    nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			matchVariations := []string{"MatchNode", "MatchCurrentNode"}
			for _, variationName := range matchVariations {
				t.Run(variationName, func(t *testing.T) {
					var result *MatchResult
					var err error

					switch variationName {
					case "MatchNode":
						node := &v1.Node{Status: v1.NodeStatus{DeclaredFeatures: tc.nodeFeatures}}
						result, err = helper.MatchNode(tc.podFeatureRequirements, node)
					case "MatchCurrentNode":
						result, err = helper.MatchCurrentNode(tc.podFeatureRequirements, tc.nodeFeatures)
					default:
						t.Fatalf("unknown match variation: %s", variationName)
					}

					require.NoError(t, err)
					assert.Equal(t, tc.expectedMatch, result.IsMatch)
					if !tc.expectedMatch {
						assert.ElementsMatch(t, tc.expectedUnsatisfied, result.UnsatisfiedRequirements)
					}
				})
			}
		})
	}

	// Test nil node
	_, err := helper.MatchNode([]string{"feature-a"}, nil)
	require.Error(t, err, "MatchNode should return an error for a nil node")
}
