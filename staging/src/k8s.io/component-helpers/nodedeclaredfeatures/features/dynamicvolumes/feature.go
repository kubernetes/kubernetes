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

package dynamicvolumes

import (
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// Ensure the feature struct implements the unified Feature interface.
var _ types.Feature = &dynamicNodeLocalEphemeralVolumesFeature{}

// DynamicNodeLocalEphemeralVolumesFeatureGate is the feature gate for DynamicNodeLocalEphemeralVolumes.
const DynamicNodeLocalEphemeralVolumesFeatureGate = "DynamicNodeLocalEphemeralVolumes"

// Feature is the implementation of the DynamicNodeLocalEphemeralVolumes feature.
var Feature = &dynamicNodeLocalEphemeralVolumesFeature{}

type dynamicNodeLocalEphemeralVolumesFeature struct{}

func (f *dynamicNodeLocalEphemeralVolumesFeature) Name() string {
	return DynamicNodeLocalEphemeralVolumesFeatureGate
}

func (f *dynamicNodeLocalEphemeralVolumesFeature) Discover(cfg *types.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(DynamicNodeLocalEphemeralVolumesFeatureGate)
}

func (f *dynamicNodeLocalEphemeralVolumesFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{DynamicNodeLocalEphemeralVolumesFeatureGate},
	}
}

func (f *dynamicNodeLocalEphemeralVolumesFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	// Volumes can only be dynamically added or removed on update.
	return false
}

// InferForUpdate returns true if the update adds, removes, or reorders a volume.
func (f *dynamicNodeLocalEphemeralVolumesFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	oldVolumes, newVolumes := oldPodInfo.Spec.Volumes, newPodInfo.Spec.Volumes
	if len(oldVolumes) != len(newVolumes) {
		return true
	}
	// At the same length, any name mismatch means volumes were added, removed, or reordered,
	// all of which need the feature.
	for i := range oldVolumes {
		if oldVolumes[i].Name != newVolumes[i].Name {
			return true
		}
	}
	return false
}

func (f *dynamicNodeLocalEphemeralVolumesFeature) MaxVersion() *version.Version {
	return nil
}
