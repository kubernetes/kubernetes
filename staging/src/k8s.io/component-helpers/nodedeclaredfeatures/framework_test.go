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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// mockFeature is a mock implementation of the Feature interface for testing.
type mockFeature struct {
	name               string
	discover           func(cfg *types.NodeConfiguration) bool
	inferForScheduling func(podInfo *types.PodInfo) bool
	inferForUpdate     func(oldPodInfo, newPodInfo *types.PodInfo) bool
	maxVersion         *version.Version
}

func (f *mockFeature) Name() string                               { return f.name }
func (f *mockFeature) Discover(cfg *types.NodeConfiguration) bool { return f.discover(cfg) }
func (f *mockFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	return f.inferForScheduling(podInfo)
}
func (f *mockFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	return f.inferForUpdate(oldPodInfo, newPodInfo)
}
func (f *mockFeature) MaxVersion() *version.Version { return f.maxVersion }

type mockFeatureGate struct {
	features map[string]bool
}

func (m *mockFeatureGate) Enabled(key string) bool {
	if m.features == nil {
		return false
	}
	return m.features[key]
}

func newMockFeatureGate(features map[string]bool) *mockFeatureGate {
	return &mockFeatureGate{features: features}
}

func TestNewFramework(t *testing.T) {
	_, err := New(nil)
	require.Error(t, err, "NewFramework should return an error with a nil registry")

	_, err = New([]types.Feature{})
	require.NoError(t, err, "NewFramework should not return an error with an empty registry")
}

func TestDiscoverNodeFeatures(t *testing.T) {
	featureMaxVersion := version.MustParse("1.38.0")
	registry := []types.Feature{
		&mockFeature{
			name: "FeatureA",
			discover: func(cfg *types.NodeConfiguration) bool {
				return cfg.FeatureGates.Enabled("feature-a")
			},
			maxVersion: featureMaxVersion,
		},
		&mockFeature{
			name: "FeatureBWithStaticConfig",
			discover: func(cfg *types.NodeConfiguration) bool {
				return cfg.FeatureGates.Enabled("feature-b") && cfg.StaticConfig.CPUManagerPolicy == "static"
			},
			maxVersion: featureMaxVersion,
		},
	}

	framework, _ := New(registry)

	testCases := []struct {
		name     string
		config   *types.NodeConfiguration
		expected []string
	}{
		{
			name: "Feature Enabled",
			config: &types.NodeConfiguration{
				FeatureGates: newMockFeatureGate(map[string]bool{string("feature-a"): true}),
				StaticConfig: types.StaticConfiguration{},
			},
			expected: []string{"FeatureA"},
		},
		{
			name: "multiple features enabled",
			config: &types.NodeConfiguration{
				FeatureGates: newMockFeatureGate(map[string]bool{
					string("feature-a"): true,
					string("feature-b"): true,
				}),
				StaticConfig: types.StaticConfiguration{CPUManagerPolicy: "static"},
			},
			expected: []string{"FeatureA", "FeatureBWithStaticConfig"}, // Should be sorted
		},
		{
			name: "no features enabled",
			config: &types.NodeConfiguration{
				FeatureGates: newMockFeatureGate(map[string]bool{
					string("feature-a"): false,
					string("feature-b"): true,
				}),
				StaticConfig: types.StaticConfiguration{CPUManagerPolicy: "none"},
			},
			expected: []string{},
		},
		{
			name: "feature past max version",
			config: &types.NodeConfiguration{
				FeatureGates: newMockFeatureGate(map[string]bool{string("feature-a"): true}),
				StaticConfig: types.StaticConfiguration{},
				Version:      featureMaxVersion.AddMinor(1),
			},
			expected: []string{}, // Not published
		},
		{
			name: "feature past max version - pre-release version",
			config: &types.NodeConfiguration{
				FeatureGates: newMockFeatureGate(map[string]bool{string("feature-a"): true}),
				StaticConfig: types.StaticConfiguration{},
				Version:      version.MustParse("1.39.0-alpha.2.39+049eafd34dfbd2"),
			},
			expected: []string{}, // Not published
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			features := framework.DiscoverNodeFeatures(tc.config)
			if len(tc.expected) == 0 {
				assert.Empty(t, features)
			} else {
				assert.Equal(t, tc.expected, features)
			}
		})
	}
}

func TestInferForPodScheduling(t *testing.T) {
	inferPodlevelResources := func(p *types.PodInfo) bool {
		return p.Spec.Resources != nil
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
		registry      []types.Feature
		newPod        *v1.Pod
		targetVersion *version.Version
		expectedReqs  FeatureSet
		expectErr     bool
		errContains   string
	}{
		{
			name: "pod with feature, inferred during scheduling",
			registry: []types.Feature{
				&mockFeature{
					name:               "PodLevelResources",
					inferForScheduling: inferPodlevelResources,
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: version.MustParse("1.30.0"),
			expectedReqs:  NewFeatureSet("PodLevelResources"),
			expectErr:     false,
		},
		{
			name: "pod without feature",
			registry: []types.Feature{
				&mockFeature{
					name:               "PodLevelResources",
					inferForScheduling: inferPodlevelResources,
				},
			},
			newPod:        podWithoutPodLevelResources,
			targetVersion: version.MustParse("1.30.0"),
			expectedReqs:  NewFeatureSet(),
			expectErr:     false,
		},
		{
			name: "feature universally available, not inferred during create",
			registry: []types.Feature{
				&mockFeature{
					name:               "PodLevelResources",
					inferForScheduling: inferPodlevelResources,
					maxVersion:         version.MustParse("1.30.0"),
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: version.MustParse("1.31.0"),
			expectedReqs:  NewFeatureSet(),
			expectErr:     false,
		},
		{
			name: "pre-release target version",
			registry: []types.Feature{
				&mockFeature{
					name:               "PodLevelResources",
					inferForScheduling: inferPodlevelResources,
					maxVersion:         version.MustParse("1.30.0"),
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: version.MustParse("0.0.0-alpha.2.39+049eafd34dfbd2"),
			expectedReqs:  NewFeatureSet("PodLevelResources"),
			expectErr:     false,
		},
		{
			name: "target version nil",
			registry: []types.Feature{
				&mockFeature{
					name:               "PodLevelResources",
					inferForScheduling: inferPodlevelResources,
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: nil,
			expectedReqs:  NewFeatureSet(),
			expectErr:     true,
			errContains:   "target version cannot be nil",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			framework, _ := New(tc.registry)
			reqs, err := framework.InferForPodScheduling(&types.PodInfo{Spec: &tc.newPod.Spec, Status: &tc.newPod.Status}, tc.targetVersion)

			if tc.expectErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errContains)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tc.expectedReqs, reqs)
			}
		})
	}
}

func TestInferForPodUpdate(t *testing.T) {
	inferResize := func(oldPodInfo, newPodInfo *types.PodInfo) bool {
		oldCPU := oldPodInfo.Spec.Containers[0].Resources.Requests.Cpu()
		newCPU := newPodInfo.Spec.Containers[0].Resources.Requests.Cpu()
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
		registry      []types.Feature
		oldPod        *v1.Pod
		newPod        *v1.Pod
		targetVersion *version.Version
		expectedReqs  FeatureSet
		expectErr     bool
		errContains   string
	}{
		{
			name: "pod update requires feature",
			registry: []types.Feature{
				&mockFeature{
					name:           "InPlacePodResize",
					inferForUpdate: inferResize,
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: version.MustParse("1.30.0"),
			expectedReqs:  NewFeatureSet("InPlacePodResize"),
			expectErr:     false,
		},
		{
			name: "pod update requires feature with pre-release version",
			registry: []types.Feature{
				&mockFeature{
					name:           "InPlacePodResize",
					inferForUpdate: inferResize,
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: version.MustParse("1.35.0-alpha.2.39+049eafd34dfbd2"),
			expectedReqs:  NewFeatureSet("InPlacePodResize"),
			expectErr:     false,
		},
		{
			name: "pod update does not require feature",
			registry: []types.Feature{
				&mockFeature{
					name:           "InPlacePodResize",
					inferForUpdate: inferResize,
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith1CPU,
			targetVersion: version.MustParse("1.30.0"),
			expectedReqs:  NewFeatureSet(),
			expectErr:     false,
		},
		{
			name: "feature universally available, not inferred during update",
			registry: []types.Feature{
				&mockFeature{
					name:           "InPlacePodResize",
					inferForUpdate: inferResize,
					maxVersion:     version.MustParse("1.30.0"),
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: version.MustParse("1.31.0"),
			expectedReqs:  NewFeatureSet(),
			expectErr:     false,
		},
		{
			name: "target version nil",
			registry: []types.Feature{
				&mockFeature{
					name:           "InPlacePodResize",
					inferForUpdate: inferResize,
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: nil,
			expectedReqs:  NewFeatureSet(),
			expectErr:     true,
			errContains:   "target version cannot be nil",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			framework, _ := New(tc.registry)
			reqs, err := framework.InferForPodUpdate(&types.PodInfo{Spec: &tc.oldPod.Spec, Status: &tc.oldPod.Status}, &types.PodInfo{Spec: &tc.newPod.Spec, Status: &tc.newPod.Status}, tc.targetVersion)

			if tc.expectErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errContains)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tc.expectedReqs, reqs)
			}
		})
	}
}

func TestMatchNode(t *testing.T) {
	testCases := []struct {
		name                   string
		podFeatureRequirements FeatureSet
		nodeFeatures           []string
		expectedMatch          bool
		expectedUnsatisfied    []string
	}{
		{
			name:                   "all features match",
			podFeatureRequirements: NewFeatureSet("feature-a", "feature-b"),
			nodeFeatures:           []string{"feature-a", "feature-b", "feature-c"},
			expectedMatch:          true,
			expectedUnsatisfied:    nil,
		},
		{
			name:                   "some features missing",
			podFeatureRequirements: NewFeatureSet("feature-a", "feature-b"),
			nodeFeatures:           []string{"feature-a", "feature-c"},
			expectedMatch:          false,
			expectedUnsatisfied:    []string{"feature-b"},
		},
		{
			name:                   "all features missing",
			podFeatureRequirements: NewFeatureSet("feature-a", "feature-b"),
			nodeFeatures:           []string{"feature-c"},
			expectedMatch:          false,
			expectedUnsatisfied:    []string{"feature-a", "feature-b"},
		},
		{
			name:                   "no node features",
			podFeatureRequirements: NewFeatureSet("feature-a", "feature-b"),
			nodeFeatures:           []string{},
			expectedMatch:          false,
			expectedUnsatisfied:    []string{"feature-a", "feature-b"},
		},
		{
			name:                   "no requirements",
			podFeatureRequirements: NewFeatureSet(),
			nodeFeatures:           []string{"feature-a", "feature-b", "feature-c"},
			expectedMatch:          true,
			expectedUnsatisfied:    nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			matchVariations := []string{"MatchNode", "MatchNodeFeatureSet"}
			for _, variationName := range matchVariations {
				t.Run(variationName, func(t *testing.T) {
					var result *MatchResult
					var err error

					switch variationName {
					case "MatchNode":
						node := &v1.Node{Status: v1.NodeStatus{DeclaredFeatures: tc.nodeFeatures}}
						result, err = MatchNode(tc.podFeatureRequirements, node)
					case "MatchNodeFeatureSet":
						result, err = MatchNodeFeatureSet(tc.podFeatureRequirements, NewFeatureSet(tc.nodeFeatures...))
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
	_, err := MatchNode(NewFeatureSet("feature-a"), nil)
	require.Error(t, err, "MatchNode should return an error for a nil node")
}
