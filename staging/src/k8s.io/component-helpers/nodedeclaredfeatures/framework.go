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
	"slices"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
)

// Framework provides functions for discovering node features and inferring pod feature requirements.
// It is stateful and holds the feature registry.
type Framework struct {
	registry []Feature
}

// FeatureSet is a set of node features.
type FeatureSet struct {
	sets.Set[string]
}

// NewFeatureSet creates a FeatureSet from a list of feature names.
func NewFeatureSet(features ...string) FeatureSet {
	return FeatureSet{Set: sets.New(features...)}
}

// Equal returns true if both the sets have the same features.
func (s *FeatureSet) Equal(other FeatureSet) bool {
	return s.Set.Equal(other.Set)
}

// Clone returns a copy of the FeatureSet.
func (s *FeatureSet) Clone() FeatureSet {
	if s.Set == nil {
		return FeatureSet{Set: nil}
	}
	return FeatureSet{Set: s.Set.Clone()}
}

// New creates a new instance of the Framework.
func New(registry []Feature) (*Framework, error) {
	if registry == nil {
		return nil, fmt.Errorf("registry must not be nil")
	}
	return &Framework{
		registry: registry,
	}, nil
}

// DiscoverNodeFeatures determines which features from the registry are enabled
// for a specific node configuration. It returns a sorted, unique list of feature names.
func (f *Framework) DiscoverNodeFeatures(cfg *NodeConfiguration) []string {
	var enabledFeatures []string
	for _, f := range f.registry {
		if f.Discover(cfg) {
			if cfg.Version != nil && f.MaxVersion() != nil && cfg.Version.GreaterThan(f.MaxVersion()) {
				continue
			}
			enabledFeatures = append(enabledFeatures, f.Name())
		}
	}
	slices.Sort(enabledFeatures)
	return enabledFeatures
}

// InferForPodScheduling determines which features from the registry are required by a pod scheduling for a given target version.
func (f *Framework) InferForPodScheduling(podInfo *PodInfo, targetVersion *version.Version) (FeatureSet, error) {
	if targetVersion == nil {
		return FeatureSet{}, fmt.Errorf("target version cannot be nil")
	}
	reqs := NewFeatureSet()
	for _, f := range f.registry {
		if f.MaxVersion() != nil && targetVersion.GreaterThan(f.MaxVersion()) {
			// If target version is greater than the feature's max version, no need to require the feature
			continue
		}
		if f.InferForScheduling(podInfo) {
			reqs.Insert(f.Name())
		}
	}
	return reqs, nil
}

// InferForPodUpdate determines which features are required by a pod update operation for a given target version.
func (f *Framework) InferForPodUpdate(oldPodInfo, newPodInfo *PodInfo, targetVersion *version.Version) (FeatureSet, error) {
	if targetVersion == nil {
		return FeatureSet{}, fmt.Errorf("target version cannot be nil")
	}
	reqs := NewFeatureSet()
	for _, f := range f.registry {
		if f.MaxVersion() != nil && targetVersion.GreaterThan(f.MaxVersion()) {
			// If target version is greater than the feature's max version, no need to require the feature
			continue
		}
		if f.InferForUpdate(oldPodInfo, newPodInfo) {
			reqs.Insert(f.Name())
		}
	}
	return reqs, nil
}

// MatchResult encapsulates the result of a feature match check.
type MatchResult struct {
	// IsMatch is true if the node satisfies all feature requirements.
	IsMatch bool
	// UnsatisfiedRequirements lists the specific features that were not met.
	// This field is only populated if IsMatch is false.
	UnsatisfiedRequirements []string
}

// MatchNode checks if a node's declared features satisfy the pod's pre-computed requirements.
// It returns a MatchResult:
// - IsMatch is true if all requiredFeatures are present in node.status.declaredFeatures.
// - UnsatisfiedRequirements lists features in requiredFeatures but not in node.status.declaredFeatures.
func MatchNode(requiredFeatures FeatureSet, node *v1.Node) (*MatchResult, error) {
	if node == nil {
		return nil, fmt.Errorf("node cannot be nil")
	}
	return MatchNodeFeatureSet(requiredFeatures, NewFeatureSet(node.Status.DeclaredFeatures...))
}

// MatchNodeFeatureSet compares a set of required features against a set of features present on a node.
// It returns a MatchResult:
// - IsMatch is true if all requiredFeatures are present in nodeFeatures.
// - UnsatisfiedRequirements lists features in requiredFeatures but not in nodeFeatures.
func MatchNodeFeatureSet(requiredFeatures FeatureSet, nodeFeatures FeatureSet) (*MatchResult, error) {
	if requiredFeatures.Len() == 0 {
		return &MatchResult{IsMatch: true}, nil // No requirements to match.
	}
	var mismatched []string
	for req := range requiredFeatures.Set {
		if !nodeFeatures.Has(req) {
			mismatched = append(mismatched, req)
			continue
		}
	}
	if len(mismatched) > 0 {
		return &MatchResult{IsMatch: false, UnsatisfiedRequirements: mismatched}, nil
	}
	return &MatchResult{IsMatch: true}, nil
}
