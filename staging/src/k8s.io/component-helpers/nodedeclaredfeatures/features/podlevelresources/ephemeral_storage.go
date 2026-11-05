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

package podlevelresources

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

var _ types.Feature = &ephemeralStorageFeature{}

// PodLevelResourcesEphemeralStorageFeatureGate is the feature gate and declared feature name
// for pod-level ephemeral-storage requests and limits.
const PodLevelResourcesEphemeralStorageFeatureGate = "PodLevelResourcesEphemeralStorage"

// EphemeralStorageFeature is the implementation of the PodLevelResourcesEphemeralStorage feature.
var EphemeralStorageFeature = &ephemeralStorageFeature{}

type ephemeralStorageFeature struct{}

func (f *ephemeralStorageFeature) Name() string {
	return PodLevelResourcesEphemeralStorageFeatureGate
}

func (f *ephemeralStorageFeature) Requirements() *types.FeatureRequirements {
	return &types.FeatureRequirements{
		EnabledFeatureGates: []string{PodLevelResourcesEphemeralStorageFeatureGate},
	}
}

func (f *ephemeralStorageFeature) Discover(cfg *types.NodeConfiguration) bool {
	return cfg.FeatureGates.Enabled(PodLevelResourcesEphemeralStorageFeatureGate)
}

func (f *ephemeralStorageFeature) InferForScheduling(podInfo *types.PodInfo) bool {
	if podInfo == nil || podInfo.Spec == nil || podInfo.Spec.Resources == nil {
		return false
	}
	if _, ok := podInfo.Spec.Resources.Requests[v1.ResourceEphemeralStorage]; ok {
		return true
	}
	if _, ok := podInfo.Spec.Resources.Limits[v1.ResourceEphemeralStorage]; ok {
		return true
	}
	return false
}

func (f *ephemeralStorageFeature) InferForUpdate(oldPodInfo, newPodInfo *types.PodInfo) bool {
	// Pod-level ephemeral-storage cannot be modified in place.
	return false
}

func (f *ephemeralStorageFeature) MaxVersion() *version.Version {
	return nil
}
