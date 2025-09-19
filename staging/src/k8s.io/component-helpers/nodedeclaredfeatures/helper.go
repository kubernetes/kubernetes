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
	"k8s.io/apimachinery/pkg/util/version"
)

// Helper provides functions for discovering node features and inferring pod feature requirements.
// It is stateful and holds the feature registry.
type Helper struct {
	registry []Feature
}

// NewHelper creates a new instance of the Helper.
func NewHelper(registry []Feature) (*Helper, error) {
	if registry == nil {
		return nil, fmt.Errorf("registry must not be nil")
	}
	return &Helper{
		registry: registry,
	}, nil
}

// DiscoverNodeFeatures determines which features from the registry are enabled
// for a specific node configuration. It returns a sorted, unique list of feature names.
func (h *Helper) DiscoverNodeFeatures(cfg *NodeConfiguration) ([]string, error) {
	var enabledFeatures []string
	for _, f := range h.registry {
		enabled, err := f.Discover(cfg)
		if err != nil {
			// We return discovery errors as they might indicate a configuration problem.
			return nil, fmt.Errorf("error discovering feature %q: %w", f.Name(), err)
		}
		if enabled {
			enabledFeatures = append(enabledFeatures, f.Name())
		}
	}

	slices.Sort(enabledFeatures)
	return enabledFeatures, nil
}

// InferForPodCreate determines which features from the registry are required by a new pod for a given target version.
func (h *Helper) InferForPodCreate(podInfo *PodInfo, targetVersion *version.Version) ([]string, error) {
	var reqs []string
	for _, f := range h.registry {
		if f.MaxVersion() != nil && targetVersion.GreaterThan(f.MaxVersion()) {
			// If target version is greater than the feature's max version, no need to declare the feature
			continue
		}
		if f.InferFromCreate(podInfo) {
			if targetVersion.LessThan(f.MinVersion()) {
				return nil, &IncompatibleFeatureError{
					FeatureName:        f.Name(),
					TargetVersion:      targetVersion,
					RequiredMinVersion: f.MinVersion(),
				}
			}
			reqs = append(reqs, f.Name())
		}
	}
	return reqs, nil
}

// InferForPodUpdate determines which features are required by a pod update operation for a given target version.
func (h *Helper) InferForPodUpdate(oldPodInfo, newPodInfo *PodInfo, targetVersion *version.Version) ([]string, error) {
	var reqs []string
	for _, f := range h.registry {
		if f.MaxVersion() != nil && targetVersion.GreaterThan(f.MaxVersion()) {
			// If target version is greater than the feature's max version, no need to declare the feature
			continue
		}
		if f.InferFromUpdate(oldPodInfo, newPodInfo) {
			if targetVersion.LessThan(f.MinVersion()) {
				return nil, &IncompatibleFeatureError{
					FeatureName:        f.Name(),
					TargetVersion:      targetVersion,
					RequiredMinVersion: f.MinVersion(),
				}
			}
			reqs = append(reqs, f.Name())
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
func (h *Helper) MatchNode(reqs []string, node *v1.Node) (*MatchResult, error) {
	if node == nil {
		return nil, fmt.Errorf("node cannot be nil")
	}
	return h.match(reqs, node.Status.DeclaredFeatures)
}

// MatchCurrentNode checks if a node's declared features satisfy the pod's pre-computed requirements.
func (h *Helper) MatchCurrentNode(reqs []string, nodeDeclaredFeatures []string) (*MatchResult, error) {
	return h.match(reqs, nodeDeclaredFeatures)
}

func (h *Helper) match(reqs []string, nodeDeclaredFeatures []string) (*MatchResult, error) {
	if len(reqs) == 0 {
		return &MatchResult{IsMatch: true}, nil // No requirements to match.
	}
	var mismatched []string
	for _, req := range reqs {
		found := false
		for _, feature := range nodeDeclaredFeatures {
			if req == feature {
				found = true
				break // Requirement satisfied.
			}
		}
		if !found {
			mismatched = append(mismatched, req)
		}
	}
	if len(mismatched) > 0 {
		return &MatchResult{IsMatch: false, UnsatisfiedRequirements: mismatched}, nil
	}
	return &MatchResult{IsMatch: true}, nil
}
