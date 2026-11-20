/*
Copyright The Kubernetes Authors.

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

package dynamiccontainers

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// Ensure the feature struct implements the unified Feature interface.
var _ types.Feature = &dynamicContainersFeature{}

// DynamicContainersFeatureGate is the feature gate for DynamicContainers.
const DynamicContainersFeatureGate = "DynamicContainers"

// Feature is the implementation of the `DynamicContainers` feature.
var Feature = &dynamicContainersFeature{}

type dynamicContainersFeature struct{}

func (f *dynamicContainersFeature) Name() string {
	return DynamicContainersFeatureGate
}

func (f *dynamicContainersFeature) Discover(cfg *types.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(DynamicContainersFeatureGate)
}

func (f *dynamicContainersFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{DynamicContainersFeatureGate},
	}
}

func (f *dynamicContainersFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	// Containers can only be added or removed on update.
	return false
}

// InferForUpdate returns true if the update adds or removes a main container. A node without the
// feature would neither start added containers nor kill removed ones.
func (f *dynamicContainersFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	oldContainers, newContainers := oldPodInfo.Spec.Containers, newPodInfo.Spec.Containers
	if len(oldContainers) != len(newContainers) {
		return true
	}
	// At the same length, any name mismatch means containers were added, removed or reordered,
	// all of which need the feature.
	for i := range oldContainers {
		if oldContainers[i].Name != newContainers[i].Name {
			return true
		}
	}
	return false
}

func (f *dynamicContainersFeature) MaxVersion() *version.Version {
	return nil
}
