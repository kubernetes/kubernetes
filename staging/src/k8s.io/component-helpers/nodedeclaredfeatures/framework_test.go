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
	"reflect"
	"slices"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
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
	requirements       func() *FeatureRequirements
}

func (f *mockFeature) Name() string { return f.name }
func (f *mockFeature) Discover(cfg *types.NodeConfiguration) bool {
	if f.discover != nil {
		return f.discover(cfg)
	}
	return true
}
func (f *mockFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	if f.inferForScheduling != nil {
		return f.inferForScheduling(podInfo)
	}
	return false
}
func (f *mockFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	if f.inferForUpdate != nil {
		return f.inferForUpdate(oldPodInfo, newPodInfo)
	}
	return false
}
func (f *mockFeature) MaxVersion() *version.Version       { return f.maxVersion }
func (f *mockFeature) Requirements() *FeatureRequirements { return f.requirements() }

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

func newTestFramework(features ...string) *Framework {
	slices.Sort(features)

	allFeatures := make([]Feature, len(features))
	for i, name := range features {
		allFeatures[i] = &mockFeature{name: name}
	}

	return New(allFeatures)
}

func TestDiscoverNodeFeatures(t *testing.T) {
	featureMaxVersion := version.MustParse("1.38.0")
	registry := []types.Feature{
		&mockFeature{
			name: "FeatureA",
			discover: func(cfg *types.NodeConfiguration) bool {
				return cfg.FeatureGates.Enabled("feature-a")
			},
			requirements: func() *FeatureRequirements {
				return &FeatureRequirements{
					EnabledFeatureGates: []string{"feature-a"},
				}
			},
			maxVersion: featureMaxVersion,
		},
		&mockFeature{
			name: "FeatureB",
			discover: func(cfg *types.NodeConfiguration) bool {
				return cfg.FeatureGates.Enabled("feature-b")
			},
			requirements: func() *FeatureRequirements {
				return &FeatureRequirements{
					EnabledFeatureGates: []string{"feature-b"},
				}
			},
			maxVersion: featureMaxVersion,
		},
	}

	framework := New(registry)

	testCases := []struct {
		name     string
		config   *types.NodeConfiguration
		expected []string
	}{
		{
			name: "Feature Enabled",
			config: &types.NodeConfiguration{
				FeatureGates: newMockFeatureGate(map[string]bool{string("feature-a"): true}),
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
			},
			expected: []string{"FeatureA", "FeatureB"}, // Should be sorted
		},
		{
			name: "no features enabled",
			config: &types.NodeConfiguration{
				FeatureGates: newMockFeatureGate(map[string]bool{
					string("feature-a"): false,
					string("feature-b"): false,
				}),
			},
			expected: []string{},
		},
		{
			name: "feature past max version",
			config: &types.NodeConfiguration{
				FeatureGates: newMockFeatureGate(map[string]bool{string("feature-a"): true}),
				Version:      featureMaxVersion.AddMinor(1),
			},
			expected: []string{}, // Not published
		},
		{
			name: "feature past max version - pre-release version",
			config: &types.NodeConfiguration{
				FeatureGates: newMockFeatureGate(map[string]bool{string("feature-a"): true}),
				Version:      version.MustParse("1.39.0-alpha.2.39+049eafd34dfbd2"),
			},
			expected: []string{}, // Not published
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			features := framework.DiscoverNodeFeatures(tc.config)
			if !slices.Equal(tc.expected, features) {
				t.Errorf("expected %#v, got %#v", tc.expected, features)
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

	// defaultMockRequirements is a helper to provide default requirements with a dummy gate.
	defaultMockRequirements := func() *FeatureRequirements {
		return &FeatureRequirements{
			EnabledFeatureGates: []string{"pod-level-resources"},
		}
	}

	testCases := []struct {
		name          string
		registry      []types.Feature
		newPod        *v1.Pod
		targetVersion *version.Version
		expectedReqs  []string
		expectErr     bool
		errContains   string
	}{
		{
			name: "pod with feature, inferred during scheduling",
			registry: []types.Feature{
				&mockFeature{
					name:               "PodLevelResources",
					inferForScheduling: inferPodlevelResources,
					requirements:       defaultMockRequirements,
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: version.MustParse("1.30.0"),
			expectedReqs:  []string{"PodLevelResources"},
			expectErr:     false,
		},
		{
			name: "pod without feature",
			registry: []types.Feature{
				&mockFeature{
					name:               "PodLevelResources",
					inferForScheduling: inferPodlevelResources,
					requirements:       defaultMockRequirements,
				},
			},
			newPod:        podWithoutPodLevelResources,
			targetVersion: version.MustParse("1.30.0"),
			expectedReqs:  nil,
			expectErr:     false,
		},
		{
			name: "feature universally available, not inferred during create",
			registry: []types.Feature{
				&mockFeature{
					name:               "PodLevelResources",
					inferForScheduling: inferPodlevelResources,
					maxVersion:         version.MustParse("1.30.0"),
					requirements:       defaultMockRequirements,
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: version.MustParse("1.31.0"),
			expectedReqs:  nil,
			expectErr:     false,
		},
		{
			name: "pre-release target version",
			registry: []types.Feature{
				&mockFeature{
					name:               "PodLevelResources",
					inferForScheduling: inferPodlevelResources,
					maxVersion:         version.MustParse("1.30.0"),
					requirements:       defaultMockRequirements,
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: version.MustParse("0.0.0-alpha.2.39+049eafd34dfbd2"),
			expectedReqs:  []string{"PodLevelResources"},
			expectErr:     false,
		},
		{
			name: "exceeds max version",
			registry: []types.Feature{
				&mockFeature{
					name:               "PodLevelResources",
					inferForScheduling: inferPodlevelResources,
					maxVersion:         version.MustParse("1.30.0"),
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: version.MustParse("1.40.0"),
			expectedReqs:  nil,
			expectErr:     false,
		},
		{
			name: "target version nil",
			registry: []types.Feature{
				&mockFeature{
					name:               "PodLevelResources",
					inferForScheduling: inferPodlevelResources,
					requirements:       defaultMockRequirements,
				},
			},
			newPod:        podWithPodLevelResources,
			targetVersion: nil,
			expectedReqs:  nil,
			expectErr:     true,
			errContains:   "target version cannot be nil",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			framework := New(tc.registry)
			reqs, err := framework.InferForPodScheduling(&types.PodInfo{Spec: &tc.newPod.Spec, Status: &tc.newPod.Status}, tc.targetVersion)

			if tc.expectErr {
				if err == nil {
					t.Errorf("expected error, got none")
				} else if !strings.Contains(err.Error(), tc.errContains) {
					t.Errorf("expected %q to contain %q", err.Error(), tc.errContains)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error %v", err)
				}
				unmappedReqs, err := framework.Unmap(reqs)
				if err != nil {
					t.Fatalf("unexpected error %v", err)
				}
				if !reflect.DeepEqual(tc.expectedReqs, unmappedReqs) {
					t.Errorf("expected %#v, got %#v", tc.expectedReqs, unmappedReqs)
				}
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

	// defaultMockRequirements is a helper to provide default requirements with a dummy gate.
	defaultMockRequirements := func() *FeatureRequirements {
		return &FeatureRequirements{
			EnabledFeatureGates: []string{"inplace-pod-resize"},
		}
	}

	testCases := []struct {
		name          string
		registry      []types.Feature
		oldPod        *v1.Pod
		newPod        *v1.Pod
		targetVersion *version.Version
		expectedReqs  []string
		expectErr     bool
		errContains   string
	}{
		{
			name: "pod update requires feature",
			registry: []types.Feature{
				&mockFeature{
					name:           "InPlacePodResize",
					inferForUpdate: inferResize,
					requirements:   defaultMockRequirements,
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: version.MustParse("1.30.0"),
			expectedReqs:  []string{"InPlacePodResize"},
			expectErr:     false,
		},
		{
			name: "pod update requires feature with pre-release version",
			registry: []types.Feature{
				&mockFeature{
					name:           "InPlacePodResize",
					inferForUpdate: inferResize,
					requirements:   defaultMockRequirements,
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: version.MustParse("1.35.0-alpha.2.39+049eafd34dfbd2"),
			expectedReqs:  []string{"InPlacePodResize"},
			expectErr:     false,
		},
		{
			name: "pod update does not require feature",
			registry: []types.Feature{
				&mockFeature{
					name:           "InPlacePodResize",
					inferForUpdate: inferResize,
					requirements:   defaultMockRequirements,
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith1CPU,
			targetVersion: version.MustParse("1.30.0"),
			expectedReqs:  nil,
			expectErr:     false,
		},
		{
			name: "feature universally available, not inferred during update",
			registry: []types.Feature{
				&mockFeature{
					name:           "InPlacePodResize",
					inferForUpdate: inferResize,
					maxVersion:     version.MustParse("1.30.0"),
					requirements:   defaultMockRequirements,
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: version.MustParse("1.31.0"),
			expectedReqs:  nil,
			expectErr:     false,
		},
		{
			name: "target version nil",
			registry: []types.Feature{
				&mockFeature{
					name:           "InPlacePodResize",
					inferForUpdate: inferResize,
					requirements:   defaultMockRequirements,
				},
			},
			oldPod:        podWith1CPU,
			newPod:        podWith2CPU,
			targetVersion: nil,
			expectedReqs:  nil,
			expectErr:     true,
			errContains:   "target version cannot be nil",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			framework := New(tc.registry)
			reqs, err := framework.InferForPodUpdate(&types.PodInfo{Spec: &tc.oldPod.Spec, Status: &tc.oldPod.Status}, &types.PodInfo{Spec: &tc.newPod.Spec, Status: &tc.newPod.Status}, tc.targetVersion)

			if tc.expectErr {
				if err == nil {
					t.Errorf("expected error, got none")
				} else if !strings.Contains(err.Error(), tc.errContains) {
					t.Errorf("expected %q to contain %q", err.Error(), tc.errContains)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error %v", err)
				}
				unmappedReqs, err := framework.Unmap(reqs)
				if err != nil {
					t.Fatalf("unexpected error %v", err)
				}
				if !reflect.DeepEqual(tc.expectedReqs, unmappedReqs) {
					t.Errorf("expected %#v, got %#v", tc.expectedReqs, unmappedReqs)
				}
			}
		})
	}
}

func TestMatchNode(t *testing.T) {
	framework := newTestFramework("feature-a", "feature-b", "feature-c")
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
			matchVariations := []string{"MatchNode", "MatchNodeFeatureSet"}
			for _, variationName := range matchVariations {
				t.Run(variationName, func(t *testing.T) {
					var result *MatchResult
					var err error

					switch variationName {
					case "MatchNode":
						node := &v1.Node{Status: v1.NodeStatus{DeclaredFeatures: tc.nodeFeatures}}
						result, err = framework.MatchNode(framework.MustMapSorted(tc.podFeatureRequirements), node)
					case "MatchNodeFeatureSet":
						result, err = framework.MatchNodeFeatureSet(framework.MustMapSorted(tc.podFeatureRequirements), framework.MustMapSorted(tc.nodeFeatures))
					default:
						t.Fatalf("unknown match variation: %s", variationName)
					}

					if err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					if tc.expectedMatch != result.IsMatch {
						t.Fatalf("expected match=%v, got %v", tc.expectedMatch, result.IsMatch)
					}
					if !tc.expectedMatch {
						want := sets.NewString(tc.expectedUnsatisfied...)
						got := sets.NewString(result.UnsatisfiedRequirements...)
						if !want.Equal(got) {
							t.Fatalf("expected unsatisfied=%v, got=%v", want.List(), got.List())
						}
					}
				})
			}
		})
	}

	// Test nil node
	_, err := framework.MatchNode(framework.MustMapSorted([]string{"feature-a"}), nil)
	if err == nil {
		t.Fatalf("MatchNode should return an error for a nil node")
	}
}
