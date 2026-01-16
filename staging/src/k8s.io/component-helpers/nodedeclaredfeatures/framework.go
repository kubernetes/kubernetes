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
	"strings"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/features"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// Framework provides functions for discovering node features and inferring pod feature requirements.
// It is stateful and holds the feature registry.
type Framework struct {
	*FeatureMapper
	registry []types.Feature
}

var DefaultFramework = New(features.AllFeatures)

// New creates a new instance of the Framework.
func New(registry []types.Feature) *Framework {
	// Ensure the features are sorted.
	slices.SortFunc(registry, func(a, b types.Feature) int {
		return strings.Compare(a.Name(), b.Name())
	})
	featureNames := make([]string, len(registry))
	for i, f := range registry {
		featureNames[i] = f.Name()
	}
	return &Framework{
		registry:      registry,
		FeatureMapper: NewFeatureMapper(featureNames),
	}
}

// DiscoverNodeFeatures determines which features from the registry are enabled
// for a specific node configuration. It returns a sorted, unique list of feature names.
func (f *Framework) DiscoverNodeFeatures(cfg *types.NodeConfiguration) []string {
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
func (f *Framework) InferForPodScheduling(podInfo *types.PodInfo, targetVersion *version.Version) (FeatureSet, error) {
	if targetVersion == nil {
		return FeatureSet{}, fmt.Errorf("target version cannot be nil")
	}
	reqs := f.NewFeatureSet()
	for i, f := range f.registry {
		if f.MaxVersion() != nil && targetVersion.GreaterThan(f.MaxVersion()) {
			// If target version is greater than the feature's max version, no need to require the feature
			continue
		}
		if f.InferForScheduling(podInfo) {
			reqs.Set(i)
		}
	}
	return reqs, nil
}

// InferForPodUpdate determines which features are required by a pod update operation for a given target version.
func (f *Framework) InferForPodUpdate(oldPodInfo, newPodInfo *types.PodInfo, targetVersion *version.Version) (FeatureSet, error) {
	if targetVersion == nil {
		return FeatureSet{}, fmt.Errorf("target version cannot be nil")
	}
	reqs := f.NewFeatureSet()
	for i, f := range f.registry {
		if f.MaxVersion() != nil && targetVersion.GreaterThan(f.MaxVersion()) {
			// If target version is greater than the feature's max version, no need to require the feature
			continue
		}
		if f.InferForUpdate(oldPodInfo, newPodInfo) {
			reqs.Set(i)
		}
	}
	return reqs, nil
}
